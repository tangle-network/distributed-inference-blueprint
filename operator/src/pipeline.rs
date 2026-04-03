//! Pipeline parallelism coordinator.
//!
//! Manages this operator's position in a multi-stage inference pipeline.
//! Each operator loads a contiguous range of model layers and forwards
//! intermediate activations to the next stage.
//!
//! Topology example (Llama 405B, 4 operators):
//!   Operator A (layers 0-25) -> B (26-50) -> C (51-75) -> D (76-100)

use blueprint_std::sync::Arc;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::config::OperatorConfig;
use crate::network::PipelineNetwork;

/// Describes the pipeline position and connectivity for one operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStageConfig {
    pub layer_start: u32,
    pub layer_end: u32,
    pub total_layers: u32,
    pub upstream_peer: Option<String>,
    pub downstream_peer: Option<String>,
}

/// Status of this operator's pipeline stage.
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

/// Intermediate activation tensor passed between pipeline stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationPayload {
    /// Unique request ID — tracks a single inference through the pipeline.
    pub request_id: String,

    /// Shape of the activation tensor: [batch, seq_len, hidden_dim]
    pub shape: Vec<u64>,

    /// Raw f16 bytes. Serialized as raw bytes for minimal overhead.
    /// Length must equal product(shape) * 2 (f16 = 2 bytes per element).
    #[serde(with = "serde_bytes_base64")]
    pub data: Vec<u8>,

    /// Metadata forwarded through the pipeline (prompt tokens, temperature, etc.)
    pub metadata: RequestMetadata,
}

/// Request metadata propagated through the pipeline alongside activations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetadata {
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub stream: bool,
}

/// Manages the operator's position in the pipeline and coordinates
/// activation forwarding between stages.
pub struct PipelineManager {
    config: Arc<OperatorConfig>,
    network: Arc<PipelineNetwork>,
    requests_processed: Arc<RwLock<u64>>,
    avg_latency_ms: Arc<RwLock<u64>>,
}

impl PipelineManager {
    pub fn new(config: Arc<OperatorConfig>, network: Arc<PipelineNetwork>) -> Self {
        Self {
            config,
            network,
            requests_processed: Arc::new(RwLock::new(0)),
            avg_latency_ms: Arc::new(RwLock::new(0)),
        }
    }

    /// Process activations through this operator's local layers.
    ///
    /// For the head operator: tokenize the prompt, run layers 0..layer_end,
    /// forward to downstream. For middle/tail: receive activations from
    /// upstream, run through local layers, forward or return.
    pub async fn process_activations(
        &self,
        input: ActivationPayload,
    ) -> anyhow::Result<ActivationPayload> {
        let start = blueprint_std::time::Instant::now();

        // Send activations to local vLLM shard for processing
        let client = reqwest::Client::new();
        let vllm_body = serde_json::json!({
            "activations": serde_json::to_value(&input)?,
            "layer_start": self.config.pipeline.layer_start,
            "layer_end": self.config.pipeline.layer_end,
        });

        let resp = client
            .post(format!("{}/process_layers", self.config.pipeline.vllm_endpoint))
            .json(&vllm_body)
            .timeout(blueprint_std::time::Duration::from_millis(
                self.config.network.activation_timeout_ms,
            ))
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("vLLM layer processing failed ({}): {}", status, body);
        }

        let output: ActivationPayload = resp.json().await?;

        // Update metrics
        let elapsed = start.elapsed().as_millis() as u64;
        {
            let mut count = self.requests_processed.write().await;
            *count += 1;
            let mut avg = self.avg_latency_ms.write().await;
            // Exponential moving average
            *avg = (*avg * 9 + elapsed) / 10;
        }

        Ok(output)
    }

    /// Handle activations received from the upstream peer.
    /// Processes through local layers, then either forwards downstream
    /// or returns the final output (if this is the tail).
    pub async fn handle_upstream_message(
        &self,
        activations: ActivationPayload,
    ) -> anyhow::Result<Option<ActivationPayload>> {
        let output = self.process_activations(activations).await?;

        if self.config.is_pipeline_tail() {
            // We're the last stage — return the final output
            Ok(Some(output))
        } else {
            // Forward to next stage
            self.forward_downstream(output).await?;
            Ok(None)
        }
    }

    /// Forward activations to the next operator in the pipeline.
    pub async fn forward_downstream(
        &self,
        activations: ActivationPayload,
    ) -> anyhow::Result<()> {
        self.network.send_activations(activations).await
    }

    /// Get current pipeline status for the /v1/pipeline/status endpoint.
    pub async fn status(&self) -> PipelineStatus {
        let requests_processed = *self.requests_processed.read().await;
        let pipeline_latency_ms = *self.avg_latency_ms.read().await;

        PipelineStatus {
            model_id: self.config.pipeline.model_id.clone(),
            layer_start: self.config.pipeline.layer_start,
            layer_end: self.config.pipeline.layer_end,
            total_layers: self.config.pipeline.total_layers,
            is_head: self.config.is_pipeline_head(),
            is_tail: self.config.is_pipeline_tail(),
            upstream_connected: self.network.is_upstream_connected().await,
            downstream_connected: self.network.is_downstream_connected().await,
            requests_processed,
            pipeline_latency_ms,
        }
    }
}

/// Base64 serde adapter for raw activation bytes.
/// Activation tensors are f16 and can be large; base64 is compact for JSON transport.
mod serde_bytes_base64 {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(data: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::Error;
        let encoded = hex::encode(data);
        serializer.serialize_str(&encoded)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
        let s = String::deserialize(deserializer)?;
        hex::decode(&s).map_err(serde::de::Error::custom)
    }
}

/// Calculate layer assignment for operator `i` out of `n` operators
/// serving a model with `total_layers` layers.
///
/// Returns (layer_start, layer_end) where layer_start is inclusive
/// and layer_end is exclusive.
pub fn calculate_layer_range(operator_index: u32, total_operators: u32, total_layers: u32) -> (u32, u32) {
    let start = operator_index * total_layers / total_operators;
    let end = (operator_index + 1) * total_layers / total_operators;
    (start, end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_range_even_split() {
        // 4 operators, 100 layers
        assert_eq!(calculate_layer_range(0, 4, 100), (0, 25));
        assert_eq!(calculate_layer_range(1, 4, 100), (25, 50));
        assert_eq!(calculate_layer_range(2, 4, 100), (50, 75));
        assert_eq!(calculate_layer_range(3, 4, 100), (75, 100));
    }

    #[test]
    fn test_layer_range_uneven_split() {
        // 3 operators, 80 layers — layers don't divide evenly
        assert_eq!(calculate_layer_range(0, 3, 80), (0, 26));
        assert_eq!(calculate_layer_range(1, 3, 80), (26, 53));
        assert_eq!(calculate_layer_range(2, 3, 80), (53, 80));
    }

    #[test]
    fn test_layer_range_two_operators() {
        assert_eq!(calculate_layer_range(0, 2, 126), (0, 63));
        assert_eq!(calculate_layer_range(1, 2, 126), (63, 126));
    }

    #[test]
    fn test_layer_range_single_operator() {
        assert_eq!(calculate_layer_range(0, 1, 100), (0, 100));
    }
}
