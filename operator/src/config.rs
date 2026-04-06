//! Distributed-inference operator configuration.
//!
//! Shared infrastructure config (`TangleConfig`, `ServerConfig`, `BillingConfig`,
//! `GpuConfig`) lives in `tangle-inference-core` and is re-exported here. The
//! only distributed-specific sections are `DistributedConfig` (pipeline layer
//! assignment + per-token pricing) and `NetworkConfig` (peer endpoints).

use serde::{Deserialize, Serialize};

pub use tangle_inference_core::{BillingConfig, GpuConfig, ServerConfig, TangleConfig};

use crate::qos::QoSConfig;

/// Top-level operator configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorConfig {
    /// Tangle network configuration (shared).
    pub tangle: TangleConfig,

    /// HTTP server configuration (shared).
    pub server: ServerConfig,

    /// Billing / ShieldedCredits configuration (shared).
    pub billing: BillingConfig,

    /// GPU configuration (shared).
    pub gpu: GpuConfig,

    /// Distributed-specific pipeline + pricing configuration.
    pub pipeline: DistributedConfig,

    /// Peer-to-peer networking configuration (distributed-specific).
    pub network: NetworkConfig,

    /// QoS heartbeat configuration (optional — disabled by default).
    #[serde(default)]
    pub qos: Option<QoSConfig>,
}

/// Distributed-inference-specific config: pipeline position, model info,
/// and per-token pricing. This is the only truly distributed-specific
/// section — everything else is shared.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// HuggingFace model ID (e.g. "meta-llama/Llama-3.1-405B-Instruct").
    pub model_id: String,

    /// First layer this operator serves (inclusive).
    pub layer_start: u32,

    /// Last layer this operator serves (exclusive).
    pub layer_end: u32,

    /// Total layers in the full model.
    pub total_layers: u32,

    /// Local vLLM endpoint serving this layer shard.
    pub vllm_endpoint: String,

    /// Maximum context length.
    #[serde(default = "default_max_model_len")]
    pub max_model_len: u32,

    /// HuggingFace token for gated models.
    pub hf_token: Option<String>,

    /// vLLM startup timeout in seconds.
    #[serde(default = "default_startup_timeout")]
    pub startup_timeout_secs: u64,

    /// Price per input token in base token units. Only meaningful on the
    /// head operator (which collects payment for the entire pipeline).
    /// Intermediate/tail operators receive their share via on-chain
    /// settlement, not from this field.
    #[serde(default)]
    pub price_per_input_token: u64,

    /// Price per output token in base token units (head operator only).
    #[serde(default)]
    pub price_per_output_token: u64,
}

/// Peer-to-peer networking config for activation forwarding between stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Upstream peer endpoint (sends activations to us). `None` on the head.
    pub upstream_peer: Option<String>,

    /// Downstream peer endpoint (we forward activations to it). `None` on the tail.
    pub downstream_peer: Option<String>,

    /// Timeout for activation send/receive in milliseconds.
    #[serde(default = "default_activation_timeout_ms")]
    pub activation_timeout_ms: u64,

    /// Maximum activation payload size in bytes (default 256 MiB).
    #[serde(default = "default_max_activation_bytes")]
    pub max_activation_bytes: usize,
}

fn default_max_model_len() -> u32 {
    8192
}

fn default_startup_timeout() -> u64 {
    600
}

fn default_activation_timeout_ms() -> u64 {
    30_000
}

fn default_max_activation_bytes() -> usize {
    256 * 1024 * 1024
}

impl OperatorConfig {
    /// Load config from file and env vars. Env prefix: `DIST_INF_`.
    pub fn load(path: Option<&str>) -> anyhow::Result<Self> {
        let mut builder = config::Config::builder();

        if let Some(path) = path {
            builder = builder.add_source(config::File::with_name(path));
        }

        builder = builder.add_source(
            config::Environment::with_prefix("DIST_INF")
                .separator("__")
                .try_parsing(true),
        );

        let cfg = builder.build()?.try_deserialize::<Self>()?;
        Ok(cfg)
    }

    /// Whether this operator is the first pipeline stage (accepts user requests).
    pub fn is_pipeline_head(&self) -> bool {
        self.pipeline.layer_start == 0
    }

    /// Whether this operator is the last pipeline stage (produces final output).
    pub fn is_pipeline_tail(&self) -> bool {
        self.pipeline.layer_end == self.pipeline.total_layers
    }

    /// Fraction of the model this operator serves (0.0..1.0).
    pub fn layer_fraction(&self) -> f64 {
        let layers_served = self.pipeline.layer_end - self.pipeline.layer_start;
        layers_served as f64 / self.pipeline.total_layers as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example_config_json() -> &'static str {
        r#"{
            "tangle": {
                "rpc_url": "http://localhost:8545",
                "chain_id": 31337,
                "operator_key": "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
                "shielded_credits": "0x0000000000000000000000000000000000000002",
                "blueprint_id": 1,
                "service_id": null
            },
            "pipeline": {
                "model_id": "meta-llama/Llama-3.1-405B-Instruct",
                "layer_start": 0,
                "layer_end": 32,
                "total_layers": 126,
                "vllm_endpoint": "http://127.0.0.1:8001",
                "price_per_input_token": 1,
                "price_per_output_token": 2
            },
            "network": {
                "upstream_peer": null,
                "downstream_peer": "http://op-b:8080"
            },
            "server": { "host": "0.0.0.0", "port": 8080 },
            "billing": { "max_spend_per_request": 1000000, "min_credit_balance": 1000 },
            "gpu": { "expected_gpu_count": 8, "min_vram_mib": 80000 }
        }"#
    }

    #[test]
    fn test_deserialize_full_config() {
        let cfg: OperatorConfig = serde_json::from_str(example_config_json()).unwrap();
        assert_eq!(cfg.pipeline.model_id, "meta-llama/Llama-3.1-405B-Instruct");
        assert_eq!(cfg.pipeline.layer_start, 0);
        assert_eq!(cfg.pipeline.layer_end, 32);
        assert_eq!(cfg.pipeline.total_layers, 126);
        assert!(cfg.is_pipeline_head());
        assert!(!cfg.is_pipeline_tail());
    }

    #[test]
    fn test_head_tail_detection() {
        let cfg: OperatorConfig = serde_json::from_str(example_config_json()).unwrap();
        assert!(cfg.is_pipeline_head());
        assert!(!cfg.is_pipeline_tail());
        assert!((cfg.layer_fraction() - (32.0 / 126.0)).abs() < 1e-9);
    }
}
