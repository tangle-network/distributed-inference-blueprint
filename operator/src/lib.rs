pub mod config;
pub mod pipeline;
pub use tangle_inference_core::qos;
pub mod server;

// Re-export shared infrastructure for downstream callers / tests.
pub use tangle_inference_core::billing;
pub use tangle_inference_core::metrics;
pub use tangle_inference_core::server::{
    error_response, extract_x402_spend_auth, payment_required, settle_billing, validate_spend_auth,
};
pub use tangle_inference_core::{
    detect_gpus, parse_nvidia_smi_output, AppState, AppStateBuilder, BillingClient, CostModel,
    CostParams, GpuInfo, NonceStore, PerTokenCostModel, RequestGuard, SpendAuthPayload,
};

use blueprint_sdk::std::sync::Arc;

use alloy_sol_types::sol;
use blueprint_sdk::macros::debug_job;
use blueprint_sdk::router::Router;
use blueprint_sdk::runner::error::RunnerError;
use blueprint_sdk::runner::BackgroundService;
use blueprint_sdk::tangle::extract::{TangleArg, TangleResult};
use blueprint_sdk::tangle::layers::TangleLayer;
use blueprint_sdk::Job;
use tokio::sync::oneshot;

use crate::config::OperatorConfig;
use crate::pipeline::PipelineManager;
use crate::server::DistributedBackend;

// --- ABI types for on-chain job encoding ---

sol! {
    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    struct InferenceRequest {
        string prompt;
        uint32 maxTokens;
        uint64 temperature;
    }

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    struct InferenceResult {
        string text;
        uint32 promptTokens;
        uint32 completionTokens;
    }

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    struct JoinPipelineRequest {
        uint64 pipelineId;
        uint32 layerStart;
        uint32 layerEnd;
    }

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    struct JoinPipelineResult {
        bool success;
        string endpoint;
    }

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    struct LeavePipelineRequest {
        uint64 pipelineId;
    }

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    struct LeavePipelineResult {
        bool success;
    }
}

// --- Job IDs ---

pub const INFERENCE_JOB: u8 = 0;
pub const JOIN_PIPELINE_JOB: u8 = 1;
pub const LEAVE_PIPELINE_JOB: u8 = 2;

// --- Shared state for the on-chain job handler ---

static PIPELINE_STATE: std::sync::OnceLock<PipelineState> = std::sync::OnceLock::new();

struct PipelineState {
    pipeline: Arc<PipelineManager>,
    config: Arc<OperatorConfig>,
}

#[allow(clippy::result_large_err)]
fn register_pipeline_state(
    config: Arc<OperatorConfig>,
    pipeline: Arc<PipelineManager>,
) -> Result<(), RunnerError> {
    PIPELINE_STATE
        .set(PipelineState { pipeline, config })
        .map_err(|_| RunnerError::Other("pipeline already registered".into()))
}

// --- Router ---

pub fn router() -> Router {
    Router::new()
        .route(
            INFERENCE_JOB,
            run_inference
                .layer(TangleLayer)
                .layer(blueprint_sdk::tee::TeeLayer::new()),
        )
        .route(
            JOIN_PIPELINE_JOB,
            join_pipeline
                .layer(TangleLayer)
                .layer(blueprint_sdk::tee::TeeLayer::new()),
        )
        .route(
            LEAVE_PIPELINE_JOB,
            leave_pipeline
                .layer(TangleLayer)
                .layer(blueprint_sdk::tee::TeeLayer::new()),
        )
}

// --- Job handlers ---

#[debug_job]
pub async fn run_inference(
    TangleArg(request): TangleArg<InferenceRequest>,
) -> Result<TangleResult<InferenceResult>, RunnerError> {
    let state = PIPELINE_STATE
        .get()
        .ok_or_else(|| RunnerError::Other("pipeline state not registered".into()))?;

    if !state.config.is_pipeline_head() {
        return Err(RunnerError::Other(
            "inference jobs must be submitted to the pipeline head operator".into(),
        ));
    }

    let response = state
        .pipeline
        .complete(serde_json::json!({
            "model": state.config.pipeline.model_id,
            "messages": [{"role":"user", "content":request.prompt}],
            "max_tokens":request.maxTokens, "temperature":request.temperature as f64 / 1000.0,
            "stream":false,
        }))
        .await
        .map_err(|error| RunnerError::Other(error.to_string().into()))?;
    let (prompt_tokens, completion_tokens) =
        pipeline::usage(&response).map_err(|error| RunnerError::Other(error.to_string().into()))?;
    let text = response["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| RunnerError::Other("engine returned no text".into()))?
        .to_string();
    Ok(TangleResult(InferenceResult {
        text,
        promptTokens: prompt_tokens,
        completionTokens: completion_tokens,
    }))
}

#[debug_job]
pub async fn join_pipeline(
    TangleArg(request): TangleArg<JoinPipelineRequest>,
) -> Result<TangleResult<JoinPipelineResult>, RunnerError> {
    let _ = request;
    Err(RunnerError::Other(
        "join through DistributedInferenceBSM.joinPipeline with the operator signer".into(),
    ))
}

#[debug_job]
pub async fn leave_pipeline(
    TangleArg(request): TangleArg<LeavePipelineRequest>,
) -> Result<TangleResult<LeavePipelineResult>, RunnerError> {
    let _ = request;
    Err(RunnerError::Other(
        "leave through DistributedInferenceBSM.leavePipeline with the operator signer".into(),
    ))
}

// --- Background service: HTTP server + pipeline coordinator ---

/// Runs the HTTP server and manages the pipeline lifecycle. On the head
/// operator this includes the billing-aware `AppState`; on intermediate and
/// tail operators a lightweight `PeerState` is used instead.
#[derive(Clone)]
pub struct DistributedInferenceServer {
    pub config: Arc<OperatorConfig>,
}

impl BackgroundService for DistributedInferenceServer {
    async fn start(&self) -> Result<oneshot::Receiver<Result<(), RunnerError>>, RunnerError> {
        let (tx, rx) = oneshot::channel();
        let config = self.config.clone();

        tokio::spawn(async move {
            let pipeline = Arc::new(PipelineManager::new(config.clone()));
            if let Err(error) = register_pipeline_state(config.clone(), pipeline.clone()) {
                let _ = tx.send(Err(error));
                return;
            }
            let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
            let start_result =
                build_head_state_and_start(config.clone(), pipeline, shutdown_rx).await;
            match start_result {
                Ok(handle) => {
                    tokio::select! {
                        result = handle => {
                            let _ = tx.send(result.map_err(|error| RunnerError::Other(error.to_string().into())));
                        }
                        _ = tokio::signal::ctrl_c() => {
                            let _ = shutdown_tx.send(true);
                            let _ = tx.send(Ok(()));
                        }
                    }
                }
                Err(error) => {
                    let _ = tx.send(Err(RunnerError::Other(error.to_string().into())));
                }
            }
        });

        Ok(rx)
    }
}

/// Build the head operator's billing-aware `AppState` and start the HTTP server.
async fn build_head_state_and_start(
    config: Arc<OperatorConfig>,
    pipeline: Arc<PipelineManager>,
    shutdown_rx: tokio::sync::watch::Receiver<bool>,
) -> anyhow::Result<tokio::task::JoinHandle<()>> {
    let billing_client = Arc::new(BillingClient::new(&config.tangle, &config.billing)?);
    let operator_address = billing_client.operator_address();
    let nonce_store = Arc::new(NonceStore::load(config.billing.nonce_store_path.clone()));
    let backend = DistributedBackend::new(config.clone(), pipeline);

    let state = AppStateBuilder::new()
        .billing(billing_client)
        .nonce_store(nonce_store)
        .server_config(Arc::new(config.server.clone()))
        .billing_config(Arc::new(config.billing.clone()))
        .tangle_config(Arc::new(config.tangle.clone()))
        .operator_address(operator_address)
        .backend(backend)
        .build()?;

    server::start_head(state, shutdown_rx).await
}
