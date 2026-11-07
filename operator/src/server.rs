//! Billing-aware OpenAI adapter to the distributed serving engine's HTTP head.
use crate::{
    config::OperatorConfig,
    pipeline::{usage, PipelineManager},
};
use axum::{
    extract::{DefaultBodyLimit, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;
use tangle_inference_core::server::{
    acquire_permit, authenticate_endpoint, authorize_request, backend_unavailable, error_response,
    metrics_handler, payment_required, resolve_payment_proof, settle_request,
};
use tangle_inference_core::{AppState, CostModel, CostParams, PerTokenCostModel, RequestGuard};
use tokio::task::JoinHandle;
use tower_http::{cors::CorsLayer, timeout::TimeoutLayer, trace::TraceLayer};

pub struct DistributedBackend {
    pub config: Arc<OperatorConfig>,
    pub pipeline: Arc<PipelineManager>,
    pub cost_model: PerTokenCostModel,
}
impl DistributedBackend {
    pub fn new(config: Arc<OperatorConfig>, pipeline: Arc<PipelineManager>) -> Self {
        let cost_model = PerTokenCostModel {
            price_per_input_token: config.pipeline.price_per_input_token,
            price_per_output_token: config.pipeline.price_per_output_token,
        };
        Self {
            config,
            pipeline,
            cost_model,
        }
    }
    pub fn calculate_cost(&self, prompt_tokens: u32, completion_tokens: u32) -> u64 {
        self.cost_model.calculate_cost(&CostParams {
            prompt_tokens,
            completion_tokens,
            ..Default::default()
        })
    }
}

pub async fn start_head(
    state: AppState,
    mut shutdown: tokio::sync::watch::Receiver<bool>,
) -> anyhow::Result<JoinHandle<()>> {
    anyhow::ensure!(
        state.backend::<DistributedBackend>().is_some(),
        "invalid distributed backend"
    );
    let bind = format!("{}:{}", state.server_config.host, state.server_config.port);
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/pipeline/activations", post(unsupported_activations))
        .route("/v1/pipeline/forward", post(unsupported_activations))
        .route("/v1/pipeline/status", get(pipeline_status))
        .route("/health", get(health_check))
        .route("/metrics", get(metrics_handler))
        .layer(DefaultBodyLimit::max(
            state.server_config.max_request_body_bytes,
        ))
        .layer(TimeoutLayer::new(Duration::from_secs(
            state.server_config.stream_timeout_secs,
        )))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            authenticate_endpoint,
        ))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(bind).await?;
    Ok(tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = shutdown.wait_for(|&stopped| stopped).await;
            })
            .await
        {
            tracing::error!(%error, "HTTP server stopped");
        }
    }))
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut request): Json<Value>,
) -> Response {
    let backend = state
        .backend::<DistributedBackend>()
        .expect("checked at startup");
    if !backend.config.is_pipeline_head() {
        return error_response(
            StatusCode::FORBIDDEN,
            "send inference to the engine head".into(),
            "pipeline_error",
            "not_head",
        );
    }
    if request["stream"].as_bool().unwrap_or(false) || !request["messages"].is_array() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "messages are required; this adapter serves non-streaming completions".into(),
            "invalid_request",
            "unsupported_request",
        );
    }
    let _permit = match acquire_permit(&state) {
        Ok(permit) => permit,
        Err(response) => return response,
    };
    let take = |request: &mut Value, key: &str| {
        let value = request.as_object_mut()?.remove(key)?;
        (!value.is_null()).then_some(value)
    };
    let (spend_auth, payment) = (
        take(&mut request, "spend_auth"),
        take(&mut request, "payment"),
    );
    let (Ok(spend_auth), Ok(payment)) = (
        spend_auth.map(serde_json::from_value).transpose(),
        payment.map(serde_json::from_value).transpose(),
    ) else {
        return error_response(
            StatusCode::BAD_REQUEST,
            "invalid payment proof".into(),
            "invalid_request",
            "invalid_payment",
        );
    };
    // The engine receives this body as-is, so the estimate must bound everything it can generate.
    let prompt_tokens =
        (request["messages"].to_string().len() / 4 + 1).min(u32::MAX as usize) as u32;
    let max_tokens = request["max_completion_tokens"]
        .as_u64()
        .or(request["max_tokens"].as_u64())
        .unwrap_or(512);
    let choices = request["n"].as_u64().unwrap_or(1).max(1);
    let completion_tokens = max_tokens.saturating_mul(choices).min(u32::MAX as u64) as u32;
    let estimated = backend.calculate_cost(prompt_tokens, completion_tokens);
    let healthy = backend.pipeline.healthy();
    let authorized = match resolve_payment_proof(&headers, spend_auth, payment) {
        Some(proof) => match authorize_request(&state, proof, estimated, healthy).await {
            Ok(authorized) => Some(authorized),
            Err(response) => return response,
        },
        None if state.billing_config.billing_required => {
            return payment_required(
                &state.billing_config,
                &state.tangle_config,
                state.operator_address,
                estimated,
            )
        }
        None => match healthy.await {
            true => None,
            false => return backend_unavailable(),
        },
    };
    let mut guard = RequestGuard::new(&backend.config.pipeline.model_id);
    let response = match backend.pipeline.complete(request).await {
        Ok(response) => response,
        Err(error) => {
            return error_response(
                StatusCode::BAD_GATEWAY,
                error.to_string(),
                "engine_error",
                "completion_failed",
            )
        }
    };
    let (prompt_tokens, completion_tokens) = usage(&response).expect("engine response validated");
    guard.set_tokens(prompt_tokens, completion_tokens);
    guard.set_success();
    if let Some(auth) = authorized {
        let cost = backend.calculate_cost(prompt_tokens, completion_tokens);
        settle_request(&state, &auth, Some(cost)).await;
    }
    Json(response).into_response()
}

async fn unsupported_activations() -> Response {
    error_response(
        StatusCode::GONE,
        "vLLM owns distributed execution; use its engine head".into(),
        "unsupported_endpoint",
        "activation_api_removed",
    )
}
async fn pipeline_status(State(state): State<AppState>) -> Json<Value> {
    Json(json!(
        state
            .backend::<DistributedBackend>()
            .expect("checked at startup")
            .pipeline
            .status()
            .await
    ))
}
async fn health_check(State(state): State<AppState>) -> Response {
    let backend = state
        .backend::<DistributedBackend>()
        .expect("checked at startup");
    if !backend.pipeline.healthy().await {
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "distributed engine unavailable".into(),
            "health_error",
            "engine_unavailable",
        );
    }
    Json(json!({"status":"ok","model":backend.config.pipeline.model_id,
        "layers":format!("{}-{}",backend.config.pipeline.layer_start,backend.config.pipeline.layer_end),
        "pipeline_head":backend.config.is_pipeline_head(),"pipeline_tail":backend.config.is_pipeline_tail()})).into_response()
}
