//! HTTP API server for the distributed inference operator.
//!
//! Routes exposed depend on the operator's pipeline position:
//!
//! **Head operator (`layer_start == 0`)** — full billing AppState:
//! - `POST /v1/chat/completions` — OpenAI-compatible, x402 SpendAuth required
//! - `POST /v1/pipeline/activations`, `POST /v1/pipeline/forward` — peer endpoints
//! - `GET /v1/pipeline/status`, `/health`, `/metrics`
//!
//! **Intermediate / tail operators** — no billing, lightweight state:
//! - `POST /v1/pipeline/activations`, `POST /v1/pipeline/forward`
//! - `GET /v1/pipeline/status`, `/health`, `/metrics`
//!
//! Only the head validates x402 SpendAuth and calls BillingClient. Payment
//! is split proportionally across the pipeline via on-chain settlement in the
//! BSM contract — intermediate operators never touch customer billing state.

use blueprint_sdk::std::sync::Arc;
use blueprint_sdk::std::time::Duration;

use axum::{
    extract::{DefaultBodyLimit, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router as HttpRouter,
};
use serde::{Deserialize, Serialize};
use tokio::task::JoinHandle;
use tower_http::cors::CorsLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;

use tangle_inference_core::server::{
    acquire_permit, billing_gate, error_response, metrics_handler, settle_billing,
};
use tangle_inference_core::{
    AppState, CostModel, CostParams, PerTokenCostModel, RequestGuard, SpendAuthPayload,
};

use crate::config::OperatorConfig;
use crate::network::PipelineNetwork;
use crate::pipeline::{ActivationPayload, PipelineManager, RequestMetadata};

/// Backend attached to the head operator's `AppState` via `AppStateBuilder`.
///
/// Owns the pipeline coordinator, peer network handle, full operator config,
/// and the per-token cost model. Retrieved in handlers via
/// `state.backend::<DistributedBackend>().unwrap()`.
pub struct DistributedBackend {
    pub config: Arc<OperatorConfig>,
    pub pipeline: Arc<PipelineManager>,
    pub network: Arc<PipelineNetwork>,
    pub cost_model: PerTokenCostModel,
}

impl DistributedBackend {
    pub fn new(
        config: Arc<OperatorConfig>,
        pipeline: Arc<PipelineManager>,
        network: Arc<PipelineNetwork>,
    ) -> Self {
        let cost_model = PerTokenCostModel {
            price_per_input_token: config.pipeline.price_per_input_token,
            price_per_output_token: config.pipeline.price_per_output_token,
        };
        Self {
            config,
            pipeline,
            network,
            cost_model,
        }
    }

    /// Calculate the cost for a request given token counts.
    pub fn calculate_cost(&self, prompt_tokens: u32, completion_tokens: u32) -> u64 {
        self.cost_model.calculate_cost(&CostParams {
            prompt_tokens,
            completion_tokens,
            ..Default::default()
        })
    }
}

/// Lightweight state for intermediate / tail operators. No billing, no nonce
/// store — just the pipeline coordinator and network handle.
#[derive(Clone)]
pub struct PeerState {
    pub config: Arc<OperatorConfig>,
    pub pipeline: Arc<PipelineManager>,
    pub network: Arc<PipelineNetwork>,
}

// --- Request / Response types (OpenAI-compatible) ---

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

    /// ShieldedCredits spend authorization. Can also be provided via x402
    /// headers (X-Payment-Signature).
    pub spend_auth: Option<SpendAuthPayload>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

fn default_max_tokens() -> u32 {
    512
}
fn default_temperature() -> f32 {
    0.7
}

// --- Server startup ---

/// Start the head operator HTTP server (full billing AppState).
pub async fn start_head(
    state: AppState,
    shutdown_rx: tokio::sync::watch::Receiver<bool>,
) -> anyhow::Result<JoinHandle<()>> {
    let backend = state
        .backend::<DistributedBackend>()
        .ok_or_else(|| anyhow::anyhow!("AppState backend is not a DistributedBackend"))?;
    let max_request_body_bytes = state.server_config.max_request_body_bytes;
    let stream_timeout_secs = state.server_config.stream_timeout_secs;
    let bind = format!("{}:{}", state.server_config.host, state.server_config.port);
    let _ = backend;

    let app = HttpRouter::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/pipeline/activations", post(receive_activations_head))
        .route("/v1/pipeline/forward", post(forward_activations_head))
        .route("/v1/pipeline/status", get(pipeline_status_head))
        .route("/health", get(health_check_head))
        .route("/metrics", get(metrics_handler))
        .layer(DefaultBodyLimit::max(max_request_body_bytes))
        .layer(TimeoutLayer::new(Duration::from_secs(stream_timeout_secs)))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    serve(app, bind, shutdown_rx_signal(shutdown_rx)).await
}

/// Start the intermediate/tail operator HTTP server (no billing).
pub async fn start_peer(
    state: PeerState,
    shutdown_rx: tokio::sync::watch::Receiver<bool>,
) -> anyhow::Result<JoinHandle<()>> {
    let max_request_body_bytes = state.config.server.max_request_body_bytes;
    let stream_timeout_secs = state.config.server.stream_timeout_secs;
    let bind = format!("{}:{}", state.config.server.host, state.config.server.port);

    let app = HttpRouter::new()
        .route("/v1/pipeline/activations", post(receive_activations_peer))
        .route("/v1/pipeline/forward", post(forward_activations_peer))
        .route("/v1/pipeline/status", get(pipeline_status_peer))
        .route("/health", get(health_check_peer))
        .route("/metrics", get(metrics_handler))
        .layer(DefaultBodyLimit::max(max_request_body_bytes))
        .layer(TimeoutLayer::new(Duration::from_secs(stream_timeout_secs)))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    serve(app, bind, shutdown_rx_signal(shutdown_rx)).await
}

async fn shutdown_rx_signal(mut rx: tokio::sync::watch::Receiver<bool>) {
    let _ = rx.wait_for(|&v| v).await;
}

async fn serve(
    app: HttpRouter,
    bind: String,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> anyhow::Result<JoinHandle<()>> {
    let listener = tokio::net::TcpListener::bind(&bind).await?;
    tracing::info!(bind = %bind, "HTTP server listening");

    let handle = tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app)
            .with_graceful_shutdown(shutdown)
            .await
        {
            tracing::error!(error = %e, "HTTP server error");
        }
    });

    Ok(handle)
}

// --- Head handlers ---

fn backend_from(state: &AppState) -> &DistributedBackend {
    state
        .backend::<DistributedBackend>()
        .expect("AppState backend is DistributedBackend (checked in start_head)")
}

/// POST /v1/chat/completions — head operator only.
///
/// Validates x402 SpendAuth, authorizes the spend on-chain, runs the prompt
/// through this operator's layers, forwards through the pipeline, and settles
/// billing with the actual metered cost on return.
async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let backend = backend_from(&state);
    let model_name = req.model.as_deref().unwrap_or(&backend.config.pipeline.model_id);
    let mut metrics_guard = RequestGuard::new(model_name);

    // 1. Concurrency gate
    let _permit = match acquire_permit(&state) {
        Ok(p) => p,
        Err(resp) => return resp,
    };

    // 2. Billing gate — extract + validate + authorize in one call
    let estimated = backend.calculate_cost(1000, 512);
    let (spend_auth, preauth_amount) =
        match billing_gate(&state, &headers, req.spend_auth, estimated).await {
            Ok(pair) => pair,
            Err(resp) => return resp,
        };

    // 3. Build initial activation from the prompt
    let request_id = uuid::Uuid::new_v4().to_string();
    let prompt = req
        .messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    let initial_activation = ActivationPayload {
        request_id: request_id.clone(),
        shape: vec![1, prompt.len() as u64, 1],
        data: prompt.as_bytes().to_vec(),
        metadata: RequestMetadata {
            prompt: prompt.clone(),
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            stream: req.stream,
        },
    };

    // 4. Process through local layers
    let local_output = match backend.pipeline.process_activations(initial_activation).await {
        Ok(o) => o,
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

    // 5. Forward through remaining pipeline stages (if any)
    let final_output = if backend.config.is_pipeline_tail() {
        local_output
    } else {
        match backend.network.send_activations_and_wait(local_output).await {
            Ok(o) => o,
            Err(e) => {
                tracing::error!(error = %e, "pipeline forwarding failed");
                return error_response(
                    StatusCode::BAD_GATEWAY,
                    format!("pipeline forwarding failed: {e}"),
                    "pipeline_error",
                    "forwarding_failed",
                );
            }
        }
    };

    // 6. Build OpenAI-compatible response
    let response = build_completion_response(
        &request_id,
        &backend.config.pipeline.model_id,
        &final_output,
    );

    // 7. Record metrics + settle billing
    let prompt_tokens = response.usage.prompt_tokens;
    let completion_tokens = response.usage.completion_tokens;
    metrics_guard.set_tokens(prompt_tokens, completion_tokens);
    metrics_guard.set_success();

    if let (Some(ref sa), Some(preauth)) = (&spend_auth, preauth_amount) {
        let actual_cost = backend.calculate_cost(prompt_tokens, completion_tokens);
        if let Err(e) = settle_billing(&state.billing, sa, preauth, actual_cost).await {
            tracing::error!(error = %e, "on-chain settlement failed — manual recovery required");
        }
    }

    Json(response).into_response()
}

async fn receive_activations_head(
    State(state): State<AppState>,
    Json(payload): Json<ActivationPayload>,
) -> Response {
    let backend = backend_from(&state);
    run_receive_activations(&backend.pipeline, &backend.network, &backend.config, payload).await
}

async fn forward_activations_head(
    State(state): State<AppState>,
    Json(payload): Json<ActivationPayload>,
) -> Response {
    let backend = backend_from(&state);
    run_forward_activations(&backend.pipeline, &backend.network, &backend.config, payload).await
}

async fn pipeline_status_head(State(state): State<AppState>) -> Json<serde_json::Value> {
    let backend = backend_from(&state);
    let status = backend.pipeline.status().await;
    Json(serde_json::to_value(status).unwrap_or_default())
}

async fn health_check_head(State(state): State<AppState>) -> Response {
    let backend = backend_from(&state);
    run_health_check(&backend.network, &backend.config).await
}

// --- Peer handlers (intermediate / tail — no billing) ---

async fn receive_activations_peer(
    State(state): State<PeerState>,
    Json(payload): Json<ActivationPayload>,
) -> Response {
    run_receive_activations(&state.pipeline, &state.network, &state.config, payload).await
}

async fn forward_activations_peer(
    State(state): State<PeerState>,
    Json(payload): Json<ActivationPayload>,
) -> Response {
    run_forward_activations(&state.pipeline, &state.network, &state.config, payload).await
}

async fn pipeline_status_peer(State(state): State<PeerState>) -> Json<serde_json::Value> {
    let status = state.pipeline.status().await;
    Json(serde_json::to_value(status).unwrap_or_default())
}

async fn health_check_peer(State(state): State<PeerState>) -> Response {
    run_health_check(&state.network, &state.config).await
}

// --- Shared handler bodies ---

async fn run_receive_activations(
    pipeline: &Arc<PipelineManager>,
    network: &Arc<PipelineNetwork>,
    config: &Arc<OperatorConfig>,
    payload: ActivationPayload,
) -> Response {
    network.mark_upstream_connected().await;

    let mut guard = RequestGuard::new(&config.pipeline.model_id);

    let output = match pipeline.process_activations(payload).await {
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

    if config.is_pipeline_tail() {
        guard.set_success();
        Json(output).into_response()
    } else if let Err(e) = pipeline.forward_downstream(output).await {
        tracing::error!(error = %e, "downstream forwarding failed");
        error_response(
            StatusCode::BAD_GATEWAY,
            format!("downstream forwarding failed: {e}"),
            "pipeline_error",
            "forwarding_failed",
        )
    } else {
        guard.set_success();
        StatusCode::OK.into_response()
    }
}

async fn run_forward_activations(
    pipeline: &Arc<PipelineManager>,
    network: &Arc<PipelineNetwork>,
    config: &Arc<OperatorConfig>,
    payload: ActivationPayload,
) -> Response {
    network.mark_upstream_connected().await;
    let mut guard = RequestGuard::new(&config.pipeline.model_id);

    let output = match pipeline.process_activations(payload).await {
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

    if config.is_pipeline_tail() {
        guard.set_success();
        Json(output).into_response()
    } else {
        match network.send_activations_and_wait(output).await {
            Ok(final_output) => {
                guard.set_success();
                Json(final_output).into_response()
            }
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

async fn run_health_check(
    network: &Arc<PipelineNetwork>,
    config: &Arc<OperatorConfig>,
) -> Response {
    let downstream_ok = network.check_downstream_health().await;
    let is_tail = config.is_pipeline_tail();
    let healthy = is_tail || downstream_ok;

    if healthy {
        Json(serde_json::json!({
            "status": "ok",
            "model": config.pipeline.model_id,
            "layers": format!("{}-{}", config.pipeline.layer_start, config.pipeline.layer_end),
            "pipeline_head": config.is_pipeline_head(),
            "pipeline_tail": config.is_pipeline_tail(),
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
) -> ChatCompletionResponse {
    let text = String::from_utf8_lossy(&output.data).to_string();

    ChatCompletionResponse {
        id: format!("chatcmpl-{request_id}"),
        object: "chat.completion".to_string(),
        created: blueprint_sdk::std::time::SystemTime::now()
            .duration_since(blueprint_sdk::std::time::UNIX_EPOCH)
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
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    }
}
