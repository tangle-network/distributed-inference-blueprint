//! QoS heartbeat for distributed inference operators.
//!
//! Submits pipeline-specific metrics: layers served, tokens processed,
//! pipeline latency, and peer connectivity status.

use blueprint_sdk::std::sync::Arc;
use blueprint_sdk::std::time::Duration;

use alloy::{
    network::EthereumWallet,
    primitives::Address,
    providers::{Provider, ProviderBuilder},
    signers::local::PrivateKeySigner,
    sol,
};

use crate::config::OperatorConfig;
use crate::pipeline::PipelineManager;

sol! {
    #[sol(rpc)]
    interface IOperatorStatusRegistry {
        struct MetricPair {
            string key;
            uint64 value;
        }

        function submitHeartbeat(
            uint64 serviceId,
            uint64 blueprintId,
            uint64 blockNumber,
            MetricPair[] calldata metrics
        ) external;
    }
}

/// Start the QoS heartbeat loop.
pub async fn start_heartbeat(
    config: Arc<OperatorConfig>,
    pipeline: Arc<PipelineManager>,
) -> anyhow::Result<tokio::task::JoinHandle<()>> {
    let qos = config
        .qos
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("qos config missing"))?;

    let interval_secs = qos.heartbeat_interval_secs;
    if interval_secs == 0 {
        anyhow::bail!("heartbeat disabled (interval = 0)");
    }

    let registry_addr: Address = qos
        .status_registry_address
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("status_registry_address not configured"))?
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid status_registry_address: {e}"))?;

    let signer: PrivateKeySigner = config.tangle.operator_key.parse()?;
    let wallet = EthereumWallet::from(signer);
    let rpc_url: reqwest::Url = config.tangle.rpc_url.parse()?;
    let service_id = config
        .tangle
        .service_id
        .ok_or_else(|| anyhow::anyhow!("service_id required for QoS heartbeat"))?;
    let blueprint_id = config.tangle.blueprint_id;

    let layer_start = config.pipeline.layer_start;
    let layer_end = config.pipeline.layer_end;

    let handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));
        interval.tick().await; // skip first immediate tick

        loop {
            interval.tick().await;

            let status = pipeline.status().await;

            let metrics = vec![
                IOperatorStatusRegistry::MetricPair {
                    key: "layers_served".to_string(),
                    value: (layer_end - layer_start) as u64,
                },
                IOperatorStatusRegistry::MetricPair {
                    key: "tokens_processed".to_string(),
                    value: status.requests_processed, // approximation
                },
                IOperatorStatusRegistry::MetricPair {
                    key: "pipeline_latency_ms".to_string(),
                    value: status.pipeline_latency_ms,
                },
                IOperatorStatusRegistry::MetricPair {
                    key: "upstream_connected".to_string(),
                    value: status.upstream_connected as u64,
                },
                IOperatorStatusRegistry::MetricPair {
                    key: "downstream_connected".to_string(),
                    value: status.downstream_connected as u64,
                },
            ];

            match send_heartbeat(
                &wallet,
                &rpc_url,
                registry_addr,
                service_id,
                blueprint_id,
                metrics,
            )
            .await
            {
                Ok(()) => {
                    tracing::debug!(service_id, blueprint_id, "heartbeat submitted");
                }
                Err(e) => {
                    tracing::warn!(error = %e, "heartbeat submission failed");
                }
            }
        }
    });

    Ok(handle)
}

async fn send_heartbeat(
    wallet: &EthereumWallet,
    rpc_url: &reqwest::Url,
    registry_addr: Address,
    service_id: u64,
    blueprint_id: u64,
    metrics: Vec<IOperatorStatusRegistry::MetricPair>,
) -> anyhow::Result<()> {
    let provider = ProviderBuilder::new()
        .wallet(wallet.clone())
        .connect_http(rpc_url.clone());

    let block_number = provider.get_block_number().await?;

    let registry = IOperatorStatusRegistry::new(registry_addr, &provider);
    let call = registry.submitHeartbeat(service_id, blueprint_id, block_number, metrics);

    let tx_hash = call.send().await?.watch().await?;
    tracing::trace!(?tx_hash, "heartbeat tx confirmed");

    Ok(())
}
