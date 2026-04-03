//! HTTP API server for the distributed inference operator.
//!
//! Routes:
//! - POST /v1/chat/completions    — OpenAI-compatible (head only, with SpendAuth)
//! - POST /v1/pipeline/activations — receive activations from upstream peer
//! - POST /v1/pipeline/forward     — receive activations and return final output
//! - GET  /v1/pipeline/status      — pipeline topology and connected peers
//! - GET  /health                   — operator health check

use blueprint_std::sync::Arc;
use blueprint_std::time::Duration;

use axum::{
    extract::State,
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router as HttpRouter,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use tower_http::cors::CorsLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;

use crate::config::OperatorConfig;
use crate::health;
use crate::network::PipelineNetwork;
use crate::pipeline::{ActivationPayload, PipelineManager, RequestMetadata};

// --- SpendAuth types (only enforced on head operator) ---

#[derive(Debug, Deserialize)]
pub struct SpendAuthPayload {
    pub commitment: String,
    pub service_id: u64,
    pub job_index: u8,
    pub amount: String,
    pub operator: String,
    pub nonce: u64,
    pub expiry: u64,
    pub signature: String,
}

// --- OpenAI-compatible request/response types ---

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub stream: bool,
    pub spend_auth: Option<SpendAuthPayload>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Debug, Serialize)]
struct ErrorDetail {
    message: String,
    r#type: String,
    code: String,
}

fn default_max_tokens() -> u32 {
    512
}
fn default_temperature() -> f32 {
    0.7
}

fn error_response(status: StatusCode, message: String, error_type: &str, code: &str) -> Response {
    let body = ErrorResponse {
        error: ErrorDetail {
            message,
            r#type: error_type.to_string(),
            code: code.to_string(),
        },
    };
    (status, Json(body)).into_response()
}

/// Shared application state.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<OperatorConfig>,
    pub pipeline: Arc<PipelineManager>,
    pub network: Arc<PipelineNetwork>,
    pub semaphore: Arc<Semaphore>,
}

/// Start the HTTP server.
pub async fn start(
    state: AppState,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
) -> anyhow::Result<tokio::task::JoinHandle<()>> {
    let mut app = HttpRouter::new()
        .route("/v1/pipeline/activations", post(receive_activations))
        .route("/v1/pipeline/forward", post(forward_activations))
        .route("/v1/pipeline/status", get(pipeline_status))
        .route("/health", get(health_check));

    // Only expose the chat completions endpoint on the pipeline head
    if state.config.is_pipeline_head() {
        app = app.route("/v1/chat/completions", post(chat_completions));
    }

    let app = app
        .layer(TimeoutLayer::new(Duration::from_secs(
            state.config.server.stream_timeout_secs,
        )))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state.clone());

    let bind = format!("{}:{}", state.config.server.host, state.config.server.port);
    let listener = tokio::net::TcpListener::bind(&bind).await?;
    tracing::info!(bind = %bind, "HTTP server listening");

    let handle = tokio::spawn(async move {
        let shutdown_signal = async move {
            let _ = shutdown_rx.wait_for(|&v| v).await;
            tracing::info!("HTTP server shutting down");
        };
        if let Err(e) = axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal)
            .await
        {
            tracing::error!(error = %e, "HTTP server error");
        }
    });

    Ok(handle)
}

// --- Handlers ---

/// POST /v1/chat/completions — OpenAI-compatible endpoint (head operator only).
///
/// Accepts user requests, runs layers 0..layer_end through local vLLM,
/// forwards activations through the pipeline, collects the final output
/// from the tail operator, and returns it to the user.
async fn chat_completions(
    State(state): State<AppState>,
    _headers: HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    // Semaphore
    let _permit = match state.semaphore.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            return error_response(
                StatusCode::TOO_MANY_REQUESTS,
                "server at capacity".to_string(),
                "rate_limit_error",
                "too_many_requests",
            );
        }
    };

    // SpendAuth validation (head operator only)
    if state.config.billing.billing_required && req.spend_auth.is_none() {
        return error_response(
            StatusCode::PAYMENT_REQUIRED,
            "SpendAuth required for inference".to_string(),
            "billing_error",
            "payment_required",
        );
    }

    if let Some(ref spend_auth) = req.spend_auth {
        // Validate amount
        if spend_auth.amount.parse::<u64>().is_err() {
            return error_response(
                StatusCode::BAD_REQUEST,
                "invalid spend_auth amount".to_string(),
                "billing_error",
                "invalid_amount",
            );
        }

        // Validate expiry
        let now = blueprint_std::time::SystemTime::now()
            .duration_since(blueprint_std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        if spend_auth.expiry < now.saturating_sub(state.config.billing.clock_skew_tolerance_secs) {
            return error_response(
                StatusCode::BAD_REQUEST,
                "spend_auth expired".to_string(),
                "billing_error",
                "expired",
            );
        }
    }

    // Build initial activation from the user's prompt
    let request_id = uuid::Uuid::new_v4().to_string();
    let prompt = req
        .messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    let initial_activation = ActivationPayload {
        request_id: request_id.clone(),
        shape: vec![1, prompt.len() as u64, 1], // placeholder shape
        data: prompt.as_bytes().to_vec(),
        metadata: RequestMetadata {
            prompt: prompt.clone(),
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            stream: req.stream,
        },
    };

    // Process through local layers
    let local_output = match state.pipeline.process_activations(initial_activation).await {
        Ok(output) => output,
        Err(e) => {
            tracing::error!(error = %e, "local layer processing failed");
            return error_response(
                StatusCode::BAD_GATEWAY,
                format!("layer processing failed: {e}"),
                "pipeline_error",
                "layer_processing_failed",
            );
        }
    };

    // If this is also the tail (single-operator pipeline), return directly
    if state.config.is_pipeline_tail() {
        return build_completion_response(
            &request_id,
            &state.config.pipeline.model_id,
            &local_output,
        );
    }

    // Forward through the pipeline and wait for the final result
    let final_output = match state.network.send_activations_and_wait(local_output).await {
        Ok(output) => output,
        Err(e) => {
            tracing::error!(error = %e, "pipeline forwarding failed");
            return error_response(
                StatusCode::BAD_GATEWAY,
                format!("pipeline forwarding failed: {e}"),
                "pipeline_error",
                "forwarding_failed",
            );
        }
    };

    build_completion_response(&request_id, &state.config.pipeline.model_id, &final_output)
}

/// POST /v1/pipeline/activations — receive activations from upstream peer.
/// Fire-and-forget: processes locally and forwards downstream.
async fn receive_activations(
    State(state): State<AppState>,
    Json(payload): Json<ActivationPayload>,
) -> Response {
    state.network.mark_upstream_connected().await;

    let output = match state.pipeline.process_activations(payload).await {
        Ok(o) => o,
        Err(e) => {
            tracing::error!(error = %e, "activation processing failed");
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("activation processing failed: {e}"),
                "pipeline_error",
                "processing_failed",
            );
        }
    };

    if state.config.is_pipeline_tail() {
        // Return the final output as the response
        Json(output).into_response()
    } else {
        // Forward downstream
        if let Err(e) = state.pipeline.forward_downstream(output).await {
            tracing::error!(error = %e, "downstream forwarding failed");
            return error_response(
                StatusCode::BAD_GATEWAY,
                format!("downstream forwarding failed: {e}"),
                "pipeline_error",
                "forwarding_failed",
            );
        }
        StatusCode::OK.into_response()
    }
}

/// POST /v1/pipeline/forward — receive activations and return final output.
/// Blocking: waits for the entire downstream pipeline to complete.
async fn forward_activations(
    State(state): State<AppState>,
    Json(payload): Json<ActivationPayload>,
) -> Response {
    state.network.mark_upstream_connected().await;

    let output = match state.pipeline.process_activations(payload).await {
        Ok(o) => o,
        Err(e) => {
            tracing::error!(error = %e, "activation processing failed");
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("activation processing failed: {e}"),
                "pipeline_error",
                "processing_failed",
            );
        }
    };

    if state.config.is_pipeline_tail() {
        Json(output).into_response()
    } else {
        match state.network.send_activations_and_wait(output).await {
            Ok(final_output) => Json(final_output).into_response(),
            Err(e) => {
                tracing::error!(error = %e, "downstream chain failed");
                error_response(
                    StatusCode::BAD_GATEWAY,
                    format!("downstream pipeline failed: {e}"),
                    "pipeline_error",
                    "chain_failed",
                )
            }
        }
    }
}

/// GET /v1/pipeline/status — show pipeline topology and peer connectivity.
async fn pipeline_status(State(state): State<AppState>) -> Json<serde_json::Value> {
    let status = state.pipeline.status().await;
    Json(serde_json::to_value(status).unwrap_or_default())
}

/// GET /health
async fn health_check(State(state): State<AppState>) -> Response {
    let downstream_ok = state.network.check_downstream_health().await;
    let is_tail = state.config.is_pipeline_tail();

    // Head/middle operators need downstream connectivity; tail just needs vLLM
    let healthy = is_tail || downstream_ok;

    if healthy {
        Json(serde_json::json!({
            "status": "ok",
            "model": state.config.pipeline.model_id,
            "layers": format!("{}-{}", state.config.pipeline.layer_start, state.config.pipeline.layer_end),
            "pipeline_head": state.config.is_pipeline_head(),
            "pipeline_tail": state.config.is_pipeline_tail(),
            "downstream_connected": downstream_ok,
        }))
        .into_response()
    } else {
        error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "downstream peer not connected".to_string(),
            "health_error",
            "downstream_disconnected",
        )
    }
}

/// Build an OpenAI-compatible chat completion response from the final pipeline output.
fn build_completion_response(
    request_id: &str,
    model: &str,
    output: &ActivationPayload,
) -> Response {
    // The final activation data contains the generated text
    let text = String::from_utf8_lossy(&output.data).to_string();

    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{request_id}"),
        object: "chat.completion".to_string(),
        created: blueprint_std::time::SystemTime::now()
            .duration_since(blueprint_std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        model: model.to_string(),
        choices: vec![Choice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: text,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: 0, // populated by vLLM layer output
            completion_tokens: 0,
            total_tokens: 0,
        },
    };

    Json(response).into_response()
}
