use serde::{Deserialize, Serialize};
use blueprint_sdk::std::fmt;
use blueprint_sdk::std::path::PathBuf;

/// Top-level operator configuration.
#[derive(Clone, Serialize, Deserialize)]
pub struct OperatorConfig {
    pub tangle: TangleConfig,
    pub pipeline: PipelineConfig,
    pub network: NetworkConfig,
    pub server: ServerConfig,
    pub billing: BillingConfig,
    pub gpu: GpuConfig,
    #[serde(default)]
    pub qos: Option<QoSConfig>,
}

impl fmt::Debug for OperatorConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OperatorConfig")
            .field("tangle", &self.tangle)
            .field("pipeline", &self.pipeline)
            .field("network", &self.network)
            .field("server", &self.server)
            .field("billing", &self.billing)
            .field("gpu", &self.gpu)
            .finish()
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TangleConfig {
    pub rpc_url: String,
    pub chain_id: u64,
    pub operator_key: String,
    pub tangle_core: String,
    pub shielded_credits: String,
    pub blueprint_id: u64,
    pub service_id: Option<u64>,
}

impl fmt::Debug for TangleConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TangleConfig")
            .field("rpc_url", &self.rpc_url)
            .field("chain_id", &self.chain_id)
            .field("operator_key", &"[REDACTED]")
            .field("tangle_core", &self.tangle_core)
            .field("shielded_credits", &self.shielded_credits)
            .field("blueprint_id", &self.blueprint_id)
            .field("service_id", &self.service_id)
            .finish()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// HuggingFace model ID (e.g. "meta-llama/Llama-3.1-405B-Instruct")
    pub model_id: String,

    /// First layer this operator serves (inclusive)
    pub layer_start: u32,

    /// Last layer this operator serves (exclusive)
    pub layer_end: u32,

    /// Total layers in the full model
    pub total_layers: u32,

    /// Local vLLM endpoint serving the assigned layer shard
    pub vllm_endpoint: String,

    /// Maximum context length
    #[serde(default = "default_max_model_len")]
    pub max_model_len: u32,

    /// HuggingFace token for gated models
    pub hf_token: Option<String>,

    /// vLLM startup timeout in seconds
    #[serde(default = "default_startup_timeout")]
    pub startup_timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Address this operator listens on for peer-to-peer activation forwarding
    #[serde(default = "default_listen_addr")]
    pub listen_addr: String,

    /// Port for activation forwarding
    #[serde(default = "default_p2p_port")]
    pub listen_port: u16,

    /// Upstream peer endpoint (operator that sends activations to us).
    /// None if this operator is the pipeline head (layer_start == 0).
    pub upstream_peer: Option<String>,

    /// Downstream peer endpoint (operator we forward activations to).
    /// None if this operator is the pipeline tail (layer_end == total_layers).
    pub downstream_peer: Option<String>,

    /// Timeout for activation send/receive in milliseconds
    #[serde(default = "default_activation_timeout_ms")]
    pub activation_timeout_ms: u64,

    /// Maximum activation payload size in bytes (default 256 MiB)
    #[serde(default = "default_max_activation_bytes")]
    pub max_activation_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,

    #[serde(default = "default_port")]
    pub port: u16,

    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_requests: usize,

    #[serde(default = "default_max_request_body_bytes")]
    pub max_request_body_bytes: usize,

    #[serde(default = "default_stream_timeout_secs")]
    pub stream_timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingConfig {
    /// Price per token in tsUSD base units. Proportional to layers served:
    /// if operator serves 25/100 layers, price = base_price * 25/100.
    pub price_per_token: u64,

    /// Whether billing is required on the head operator
    #[serde(default = "default_billing_required")]
    pub billing_required: bool,

    pub max_spend_per_request: u64,
    pub min_credit_balance: u64,

    #[serde(default)]
    pub min_charge_amount: u64,

    #[serde(default = "default_clock_skew_tolerance")]
    pub clock_skew_tolerance_secs: u64,

    #[serde(default)]
    pub payment_token_address: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    pub gpu_count: u32,
    pub total_vram_mib: u32,

    #[serde(default)]
    pub gpu_model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QoSConfig {
    #[serde(default)]
    pub heartbeat_interval_secs: u64,

    #[serde(default)]
    pub status_registry_address: Option<String>,
}

impl Default for QoSConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval_secs: 0,
            status_registry_address: None,
        }
    }
}

fn default_max_model_len() -> u32 {
    8192
}
fn default_startup_timeout() -> u64 {
    600
}
fn default_listen_addr() -> String {
    "0.0.0.0".to_string()
}
fn default_p2p_port() -> u16 {
    9090
}
fn default_activation_timeout_ms() -> u64 {
    30_000
}
fn default_max_activation_bytes() -> usize {
    256 * 1024 * 1024
}
fn default_host() -> String {
    "0.0.0.0".to_string()
}
fn default_port() -> u16 {
    8080
}
fn default_max_concurrent() -> usize {
    32
}
fn default_max_request_body_bytes() -> usize {
    16 * 1024 * 1024
}
fn default_stream_timeout_secs() -> u64 {
    600
}
fn default_billing_required() -> bool {
    true
}
fn default_clock_skew_tolerance() -> u64 {
    30
}

impl OperatorConfig {
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

    /// Whether this operator is the first stage in the pipeline (accepts user requests).
    pub fn is_pipeline_head(&self) -> bool {
        self.pipeline.layer_start == 0
    }

    /// Whether this operator is the last stage (produces final output).
    pub fn is_pipeline_tail(&self) -> bool {
        self.pipeline.layer_end == self.pipeline.total_layers
    }

    /// Fraction of the model this operator serves (0.0..1.0).
    pub fn layer_fraction(&self) -> f64 {
        let layers_served = self.pipeline.layer_end - self.pipeline.layer_start;
        layers_served as f64 / self.pipeline.total_layers as f64
    }
}
