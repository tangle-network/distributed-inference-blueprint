//! Model execution is owned by vLLM's distributed executor, including Ray workers.
//! The operator calls its OpenAI HTTP head; it never transports hidden tensors.
use crate::config::OperatorConfig;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStatus {
    pub model_id: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub total_layers: u32,
    pub is_head: bool,
    pub is_tail: bool,
    pub upstream_connected: bool,
    pub downstream_connected: bool,
    pub requests_processed: u64,
    pub pipeline_latency_ms: u64,
}

pub struct PipelineManager {
    config: Arc<OperatorConfig>,
    client: reqwest::Client,
    requests_processed: AtomicU64,
    latency_ms: AtomicU64,
}

impl PipelineManager {
    pub fn new(config: Arc<OperatorConfig>) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
            requests_processed: AtomicU64::new(0),
            latency_ms: AtomicU64::new(0),
        }
    }

    pub async fn complete(&self, mut request: Value) -> anyhow::Result<Value> {
        anyhow::ensure!(
            !request["stream"].as_bool().unwrap_or(false),
            "streaming is not supported by this adapter"
        );
        request["model"] = json!(self.config.pipeline.model_id);
        let start = std::time::Instant::now();
        let response = self
            .client
            .post(format!(
                "{}/v1/chat/completions",
                self.config.pipeline.vllm_endpoint.trim_end_matches('/')
            ))
            .timeout(std::time::Duration::from_secs(
                self.config.server.stream_timeout_secs,
            ))
            .json(&request)
            .send()
            .await?
            .error_for_status()?
            .json::<Value>()
            .await?;
        usage(&response)?;
        anyhow::ensure!(
            response["choices"]
                .as_array()
                .is_some_and(|v| !v.is_empty()),
            "engine returned no choices"
        );
        self.requests_processed.fetch_add(1, Ordering::Relaxed);
        self.latency_ms
            .store(start.elapsed().as_millis() as u64, Ordering::Relaxed);
        Ok(response)
    }

    pub async fn healthy(&self) -> bool {
        self.client
            .get(format!(
                "{}/health",
                self.config.pipeline.vllm_endpoint.trim_end_matches('/')
            ))
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await
            .is_ok_and(|response| response.status().is_success())
    }

    pub async fn status(&self) -> PipelineStatus {
        let ready = self.healthy().await;
        PipelineStatus {
            model_id: self.config.pipeline.model_id.clone(),
            layer_start: self.config.pipeline.layer_start,
            layer_end: self.config.pipeline.layer_end,
            total_layers: self.config.pipeline.total_layers,
            is_head: self.config.is_pipeline_head(),
            is_tail: self.config.is_pipeline_tail(),
            upstream_connected: ready,
            downstream_connected: ready,
            requests_processed: self.requests_processed.load(Ordering::Relaxed),
            pipeline_latency_ms: self.latency_ms.load(Ordering::Relaxed),
        }
    }
}

pub fn usage(response: &Value) -> anyhow::Result<(u32, u32)> {
    let count = |field: &str| -> anyhow::Result<u32> {
        Ok(u32::try_from(
            response["usage"][field]
                .as_u64()
                .ok_or_else(|| anyhow::anyhow!("engine omitted metered {field}"))?,
        )?)
    };
    Ok((count("prompt_tokens")?, count("completion_tokens")?))
}
