use alloy_sol_types::SolValue;
use distributed_inference::{config::OperatorConfig, DistributedInferenceServer};

fn registration_payload(config: &OperatorConfig) -> Vec<u8> {
    let endpoint = format!("http://{}:{}", config.server.host, config.server.port);

    (
        config.pipeline.model_id.clone(),
        config.pipeline.layer_start,
        config.pipeline.layer_end,
        config.pipeline.total_layers,
        config.gpu.expected_gpu_count,
        config.gpu.min_vram_mib,
        config
            .gpu
            .gpu_model
            .clone()
            .expect("GPU catalog class required"),
        endpoint,
    )
        .abi_encode()
}

tangle_inference_core::run_operator!(
    OperatorConfig::load(None),
    registration_payload,
    distributed_inference::router(),
    |config| DistributedInferenceServer { config },
    qos
);
