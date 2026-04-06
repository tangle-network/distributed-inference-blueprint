use blueprint_sdk::std::sync::Arc;

use alloy_sol_types::SolValue;
use blueprint_sdk::contexts::tangle::TangleClientContext;
use blueprint_sdk::runner::config::BlueprintEnvironment;
use blueprint_sdk::runner::tangle::config::TangleConfig;
use blueprint_sdk::runner::BlueprintRunner;
use blueprint_sdk::tangle::{TangleConsumer, TangleProducer};

use distributed_inference::config::OperatorConfig;
use distributed_inference::{detect_gpus, DistributedInferenceServer};

fn setup_log() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::from_default_env();
    fmt().with_env_filter(filter).init();
}

/// Build ABI-encoded registration payload for DistributedInferenceBSM.onRegister.
/// Includes layer range so the BSM can validate coverage and assign pipeline position.
fn registration_payload(config: &OperatorConfig) -> Vec<u8> {
    let endpoint = format!("http://{}:{}", config.server.host, config.server.port);

    (
        config.pipeline.model_id.clone(),
        config.pipeline.layer_start,
        config.pipeline.layer_end,
        config.pipeline.total_layers,
        config.gpu.expected_gpu_count,
        config.gpu.min_vram_mib,
        endpoint,
    )
        .abi_encode()
}

#[tokio::main]
#[allow(clippy::result_large_err)]
async fn main() -> Result<(), blueprint_sdk::Error> {
    setup_log();

    let config = OperatorConfig::load(None)
        .map_err(|e| blueprint_sdk::Error::Other(format!("config load failed: {e}")))?;
    let config = Arc::new(config);

    tracing::info!(
        model = %config.pipeline.model_id,
        layers = format!("{}-{}/{}", config.pipeline.layer_start, config.pipeline.layer_end, config.pipeline.total_layers),
        head = config.is_pipeline_head(),
        tail = config.is_pipeline_tail(),
        fraction = format!("{:.1}%", config.layer_fraction() * 100.0),
        "distributed inference operator starting"
    );

    let env = BlueprintEnvironment::load()?;

    // Registration mode
    if env.registration_mode() {
        let payload = registration_payload(&config);
        let output_path = env.registration_output_path();
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?;
        }
        std::fs::write(&output_path, &payload)
            .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?;
        tracing::info!(
            path = %output_path.display(),
            model = %config.pipeline.model_id,
            layer_start = config.pipeline.layer_start,
            layer_end = config.pipeline.layer_end,
            "registration payload saved"
        );
        return Ok(());
    }

    // GPU detection (non-fatal)
    match detect_gpus().await {
        Ok(gpus) => {
            tracing::info!(count = gpus.len(), "detected GPUs");
            for gpu in &gpus {
                tracing::info!(name = %gpu.name, vram_mib = gpu.memory_total_mib, "GPU");
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, "GPU detection failed");
        }
    }

    // Tangle client
    let tangle_client = env
        .tangle_client()
        .await
        .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?;

    let service_id = env
        .protocol_settings
        .tangle()
        .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?
        .service_id
        .ok_or_else(|| blueprint_sdk::Error::Other("no service ID configured".to_string()))?;

    let tangle_producer = TangleProducer::new(tangle_client.clone(), service_id);
    let tangle_consumer = TangleConsumer::new(tangle_client.clone());

    // QoS heartbeat (deferred — needs pipeline manager, started inside BackgroundService)
    let qos_enabled = config
        .qos
        .as_ref()
        .map(|q| q.heartbeat_interval_secs > 0)
        .unwrap_or(false);
    if qos_enabled {
        tracing::info!("QoS heartbeat will start with background service");
    }

    let server = DistributedInferenceServer {
        config: config.clone(),
    };

    BlueprintRunner::builder(TangleConfig::default(), env)
        .router(distributed_inference::router())
        .producer(tangle_producer)
        .consumer(tangle_consumer)
        .background_service(server)
        .run()
        .await?;

    Ok(())
}
