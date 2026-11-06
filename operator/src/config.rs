use serde::{Deserialize, Serialize};

pub use tangle_inference_core::{BillingConfig, GpuConfig, ServerConfig, TangleConfig};

use crate::qos::QoSConfig;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorConfig {
    pub tangle: TangleConfig,
    pub server: ServerConfig,
    pub billing: BillingConfig,
    pub gpu: GpuConfig,
    pub pipeline: DistributedConfig,
    #[serde(default)]
    pub qos: Option<QoSConfig>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    pub model_id: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub total_layers: u32,
    pub vllm_endpoint: String,
    #[serde(default)]
    pub price_per_input_token: u64,
    #[serde(default)]
    pub price_per_output_token: u64,
}
impl OperatorConfig {
    pub fn load(path: Option<&str>) -> anyhow::Result<Self> {
        let mut builder = config::Config::builder();

        if let Some(path) = path {
            builder = builder.add_source(config::File::with_name(path));
        }

        builder = builder.add_source(
            config::Environment::with_prefix("DIST_INF")
                .prefix_separator("_")
                .separator("__")
                .try_parsing(true),
        );

        let cfg = builder.build()?.try_deserialize::<Self>()?;
        anyhow::ensure!(
            cfg.pipeline.layer_start < cfg.pipeline.layer_end
                && cfg.pipeline.layer_end <= cfg.pipeline.total_layers,
            "invalid layer range"
        );
        anyhow::ensure!(
            cfg.gpu.expected_gpu_count > 0
                && cfg.gpu.min_vram_mib > 0
                && cfg
                    .gpu
                    .gpu_model
                    .as_ref()
                    .is_some_and(|class| !class.is_empty()),
            "GPU class, count and VRAM per GPU are required"
        );
        Ok(cfg)
    }
    pub fn is_pipeline_head(&self) -> bool {
        self.pipeline.layer_start == 0
    }
    pub fn is_pipeline_tail(&self) -> bool {
        self.pipeline.layer_end == self.pipeline.total_layers
    }
    pub fn layer_fraction(&self) -> f64 {
        let layers_served = self.pipeline.layer_end - self.pipeline.layer_start;
        layers_served as f64 / self.pipeline.total_layers as f64
    }
}
