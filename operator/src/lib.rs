pub mod config;
pub mod health;
pub mod network;
pub mod pipeline;
pub mod qos;
pub mod server;
pub mod shard;

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

// --- Shared state ---

static PIPELINE_STATE: std::sync::OnceLock<PipelineState> = std::sync::OnceLock::new();

struct PipelineState {
    pipeline: Arc<PipelineManager>,
    config: Arc<OperatorConfig>,
    client: reqwest::Client,
}

fn register_pipeline_state(
    config: Arc<OperatorConfig>,
    pipeline: Arc<PipelineManager>,
) -> Result<(), RunnerError> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(300))
        .build()
        .map_err(|e| RunnerError::Other(format!("failed to build HTTP client: {e}").into()))?;
    let _ = PIPELINE_STATE.set(PipelineState {
        pipeline,
        config,
        client,
    });
    Ok(())
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
    let state = PIPELINE_STATE.get().ok_or_else(|| {
        RunnerError::Other("pipeline state not registered".into())
    })?;

    let temperature = request.temperature as f32 / 1000.0;

    // For on-chain jobs, the head operator receives the request and
    // orchestrates the full pipeline pass.
    if !state.config.is_pipeline_head() {
        return Err(RunnerError::Other(
            "inference jobs must be submitted to the pipeline head operator".into(),
        ));
    }

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

    // Process through local layers
    let local_output = state
        .pipeline
        .process_activations(activation)
        .await
        .map_err(|e| RunnerError::Other(format!("layer processing failed: {e}").into()))?;

    // If multi-operator pipeline, forward through remaining stages
    let final_output = if state.config.is_pipeline_tail() {
        local_output
    } else {
        state
            .pipeline
            .forward_downstream(local_output.clone())
            .await
            .map_err(|e| RunnerError::Other(format!("pipeline forward failed: {e}").into()))?;
        // For on-chain jobs we use the synchronous forward path
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
    let state = PIPELINE_STATE.get().ok_or_else(|| {
        RunnerError::Other("pipeline state not registered".into())
    })?;

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
    tracing::info!(
        pipeline_id = request.pipelineId,
        "leave pipeline request"
    );

    Ok(TangleResult(LeavePipelineResult { success: true }))
}

// --- Background service ---

/// Runs the HTTP server and manages the pipeline lifecycle.
#[derive(Clone)]
pub struct DistributedInferenceServer {
    pub config: Arc<OperatorConfig>,
}

impl BackgroundService for DistributedInferenceServer {
    async fn start(&self) -> Result<oneshot::Receiver<Result<(), RunnerError>>, RunnerError> {
        let (tx, rx) = oneshot::channel();
        let config = self.config.clone();

        tokio::spawn(async move {
            // 1. Build the pipeline network
            let network = Arc::new(PipelineNetwork::new(config.clone()));

            // 2. Build the pipeline manager
            let pipeline = Arc::new(PipelineManager::new(config.clone(), network.clone()));

            // 3. Register shared state for on-chain job handlers
            if let Err(e) = register_pipeline_state(config.clone(), pipeline.clone()) {
                tracing::error!(error = %e, "failed to register pipeline state");
                let _ = tx.send(Err(e));
                return;
            }

            // 4. Build semaphore
            let max_concurrent = config.server.max_concurrent_requests;
            let semaphore = Arc::new(if max_concurrent == 0 {
                tokio::sync::Semaphore::new(tokio::sync::Semaphore::MAX_PERMITS)
            } else {
                tokio::sync::Semaphore::new(max_concurrent)
            });

            // 5. Shutdown channel
            let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

            // 6. Start HTTP server
            let state = server::AppState {
                config: config.clone(),
                pipeline: pipeline.clone(),
                network: network.clone(),
                semaphore,
            };

            match server::start(state, shutdown_rx).await {
                Ok(_handle) => {
                    tracing::info!(
                        model = %config.pipeline.model_id,
                        layers = format!("{}-{}", config.pipeline.layer_start, config.pipeline.layer_end),
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

            // 7. Connectivity watchdog
            loop {
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(15)) => {}
                    _ = tokio::signal::ctrl_c() => {
                        tracing::info!("shutdown signal received");
                        let _ = shutdown_tx.send(true);
                        return;
                    }
                }

                // Check peer connectivity
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
