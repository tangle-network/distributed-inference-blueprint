pub mod config;
pub mod network;
pub mod pipeline;
pub mod qos;
pub mod server;
pub mod shard;

// Re-export shared infrastructure for downstream callers / tests.
pub use tangle_inference_core::{
    detect_gpus, parse_nvidia_smi_output, AppState, AppStateBuilder, BillingClient, CostModel,
    CostParams, GpuInfo, NonceStore, PerTokenCostModel, RequestGuard, SpendAuthPayload,
};
pub use tangle_inference_core::server::{
    error_response, extract_x402_spend_auth, payment_required, settle_billing, validate_spend_auth,
};
pub use tangle_inference_core::metrics;
pub use tangle_inference_core::billing;

use blueprint_sdk::std::sync::Arc;
use blueprint_sdk::std::time::Duration;

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
use crate::network::PipelineNetwork;
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
    let _ = PIPELINE_STATE.set(PipelineState { pipeline, config });
    Ok(())
}

/// Shared state for direct testing (raw HTTP, no PipelineManager).
static DIRECT_ENDPOINT: std::sync::OnceLock<DirectEndpoint> = std::sync::OnceLock::new();

struct DirectEndpoint {
    url: String,
    client: reqwest::Client,
    layer_start: u32,
    layer_end: u32,
}

/// Initialize a raw HTTP endpoint for direct testing (wiremock).
pub fn init_direct_for_testing(base_url: &str, layer_start: u32, layer_end: u32) {
    let _ = DIRECT_ENDPOINT.set(DirectEndpoint {
        url: format!("{base_url}/process_layers"),
        client: reqwest::Client::new(),
        layer_start,
        layer_end,
    });
}

/// Direct activation processing — bypasses PipelineManager with a raw HTTP POST.
pub async fn process_activations_direct(
    input: &[f32],
    layer_start: u32,
    layer_end: u32,
) -> Result<Vec<f32>, RunnerError> {
    let endpoint = DIRECT_ENDPOINT
        .get()
        .ok_or_else(|| RunnerError::Other("direct endpoint not registered".into()))?;

    // layer_start/layer_end captured at registration time; caller passes
    // the same values for compatibility with earlier test harnesses.
    let _ = (endpoint.layer_start, endpoint.layer_end);

    let body = serde_json::json!({
        "activations": input,
        "layer_start": layer_start,
        "layer_end": layer_end,
    });

    let resp = endpoint
        .client
        .post(&endpoint.url)
        .json(&body)
        .send()
        .await
        .map_err(|e| RunnerError::Other(format!("activation processing failed: {e}").into()))?;

    let result: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| RunnerError::Other(format!("activation response parse failed: {e}").into()))?;

    let output = result["activations"]
        .as_array()
        .ok_or_else(|| RunnerError::Other("missing activations in response".into()))?
        .iter()
        .filter_map(|v| v.as_f64().map(|f| f as f32))
        .collect::<Vec<f32>>();

    Ok(output)
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

    let temperature = request.temperature as f32 / 1000.0;

    let activation = pipeline::ActivationPayload {
        request_id: uuid::Uuid::new_v4().to_string(),
        shape: vec![1, request.prompt.len() as u64, 1],
        data: request.prompt.as_bytes().to_vec(),
        metadata: pipeline::RequestMetadata {
            prompt: request.prompt.clone(),
            max_tokens: request.maxTokens,
            temperature,
            stream: false,
        },
    };

    let local_output = state
        .pipeline
        .process_activations(activation)
        .await
        .map_err(|e| RunnerError::Other(format!("layer processing failed: {e}").into()))?;

    let final_output = if state.config.is_pipeline_tail() {
        local_output
    } else {
        state
            .pipeline
            .forward_downstream(local_output.clone())
            .await
            .map_err(|e| RunnerError::Other(format!("pipeline forward failed: {e}").into()))?;
        local_output
    };

    let text = String::from_utf8_lossy(&final_output.data).to_string();

    Ok(TangleResult(InferenceResult {
        text,
        promptTokens: 0,
        completionTokens: 0,
    }))
}

#[debug_job]
pub async fn join_pipeline(
    TangleArg(request): TangleArg<JoinPipelineRequest>,
) -> Result<TangleResult<JoinPipelineResult>, RunnerError> {
    let state = PIPELINE_STATE
        .get()
        .ok_or_else(|| RunnerError::Other("pipeline state not registered".into()))?;

    tracing::info!(
        pipeline_id = request.pipelineId,
        layer_start = request.layerStart,
        layer_end = request.layerEnd,
        "join pipeline request"
    );

    let endpoint = format!(
        "http://{}:{}",
        state.config.server.host, state.config.server.port
    );

    Ok(TangleResult(JoinPipelineResult {
        success: true,
        endpoint,
    }))
}

#[debug_job]
pub async fn leave_pipeline(
    TangleArg(request): TangleArg<LeavePipelineRequest>,
) -> Result<TangleResult<LeavePipelineResult>, RunnerError> {
    tracing::info!(pipeline_id = request.pipelineId, "leave pipeline request");
    Ok(TangleResult(LeavePipelineResult { success: true }))
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
            // 1. Build the peer network + pipeline manager
            let network = Arc::new(PipelineNetwork::new(config.clone()));
            let pipeline = Arc::new(PipelineManager::new(config.clone(), network.clone()));

            // 2. Register shared state for on-chain job handlers
            if let Err(e) = register_pipeline_state(config.clone(), pipeline.clone()) {
                let _ = tx.send(Err(e));
                return;
            }

            // 3. Shutdown channel
            let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

            // 4. Start the HTTP server — head gets full billing AppState,
            //    intermediate/tail use the lightweight PeerState.
            let start_result: anyhow::Result<tokio::task::JoinHandle<()>> =
                if config.is_pipeline_head() {
                    build_head_state_and_start(
                        config.clone(),
                        pipeline.clone(),
                        network.clone(),
                        shutdown_rx.clone(),
                    )
                    .await
                } else {
                    let state = server::PeerState {
                        config: config.clone(),
                        pipeline: pipeline.clone(),
                        network: network.clone(),
                    };
                    server::start_peer(state, shutdown_rx.clone()).await
                };

            match start_result {
                Ok(_handle) => {
                    tracing::info!(
                        model = %config.pipeline.model_id,
                        layers = format!(
                            "{}-{}",
                            config.pipeline.layer_start, config.pipeline.layer_end
                        ),
                        head = config.is_pipeline_head(),
                        tail = config.is_pipeline_tail(),
                        "distributed inference server started"
                    );
                    let _ = tx.send(Ok(()));
                }
                Err(e) => {
                    tracing::error!(error = %e, "failed to start HTTP server");
                    let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                    return;
                }
            }

            // 5. Connectivity watchdog
            loop {
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(15)) => {}
                    _ = tokio::signal::ctrl_c() => {
                        tracing::info!("shutdown signal received");
                        let _ = shutdown_tx.send(true);
                        return;
                    }
                }

                let downstream_ok = network.check_downstream_health().await;
                let upstream_ok = network.check_upstream_health().await;

                if !config.is_pipeline_tail() && !downstream_ok {
                    tracing::warn!("downstream peer not reachable");
                }
                if !config.is_pipeline_head() && !upstream_ok {
                    tracing::warn!("upstream peer not reachable");
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
    network: Arc<PipelineNetwork>,
    shutdown_rx: tokio::sync::watch::Receiver<bool>,
) -> anyhow::Result<tokio::task::JoinHandle<()>> {
    let billing_client = Arc::new(BillingClient::new(&config.tangle, &config.billing)?);
    let operator_address = billing_client.operator_address();
    let nonce_store = Arc::new(NonceStore::load(config.billing.nonce_store_path.clone()));
    let backend = DistributedBackend::new(config.clone(), pipeline, network);

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
