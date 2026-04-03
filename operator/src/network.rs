//! Inter-operator networking for pipeline parallelism.
//!
//! Uses point-to-point request-response (not gossip) to forward
//! intermediate activations between pipeline stages. Activations
//! are serialized as raw f16 bytes for minimal overhead.

use blueprint_sdk::std::sync::Arc;
use blueprint_sdk::std::time::Duration;
use tokio::sync::RwLock;

use crate::config::OperatorConfig;
use crate::pipeline::ActivationPayload;

/// Manages connections between pipeline stages.
pub struct PipelineNetwork {
    config: Arc<OperatorConfig>,
    client: reqwest::Client,
    upstream_connected: Arc<RwLock<bool>>,
    downstream_connected: Arc<RwLock<bool>>,
}

impl PipelineNetwork {
    pub fn new(config: Arc<OperatorConfig>) -> Self {
        let timeout_ms = config.network.activation_timeout_ms;
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(timeout_ms))
            .pool_max_idle_per_host(4)
            .build()
            .expect("failed to build HTTP client for pipeline network");

        Self {
            config,
            client,
            upstream_connected: Arc::new(RwLock::new(false)),
            downstream_connected: Arc::new(RwLock::new(false)),
        }
    }

    /// Send intermediate activations to the next stage in the pipeline.
    ///
    /// Serializes the activation tensor as a JSON envelope with hex-encoded
    /// raw f16 bytes. For production, this should use a binary protocol
    /// (e.g., gRPC with protobuf) to avoid base64/hex overhead.
    pub async fn send_activations(
        &self,
        payload: ActivationPayload,
    ) -> anyhow::Result<()> {
        let downstream = self
            .config
            .network
            .downstream_peer
            .as_ref()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no downstream peer configured — this operator is the pipeline tail"
                )
            })?;

        let url = format!("{}/v1/pipeline/activations", downstream);

        let resp = self
            .client
            .post(&url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| {
                tracing::error!(
                    error = %e,
                    downstream = downstream.as_str(),
                    "failed to send activations to downstream peer"
                );
                // Mark downstream as disconnected
                let connected = self.downstream_connected.clone();
                tokio::spawn(async move {
                    *connected.write().await = false;
                });
                anyhow::anyhow!("downstream activation send failed: {e}")
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!(
                "downstream peer rejected activations ({}): {}",
                status,
                body
            );
        }

        // Mark downstream as connected on success
        *self.downstream_connected.write().await = true;

        Ok(())
    }

    /// Send activations to the downstream peer and wait for the final result.
    ///
    /// Used by intermediate operators that need to collect the tail's output
    /// and return it to the caller (the head operator).
    pub async fn send_activations_and_wait(
        &self,
        payload: ActivationPayload,
    ) -> anyhow::Result<ActivationPayload> {
        let downstream = self
            .config
            .network
            .downstream_peer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no downstream peer configured"))?;

        let url = format!("{}/v1/pipeline/forward", downstream);

        let resp = self
            .client
            .post(&url)
            .json(&payload)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("downstream forward failed ({}): {}", status, body);
        }

        *self.downstream_connected.write().await = true;

        let result: ActivationPayload = resp.json().await?;
        Ok(result)
    }

    /// Receive activations from the upstream peer.
    ///
    /// This is called by the HTTP server handler when it receives a POST
    /// to /v1/pipeline/activations. The actual receiving is done via axum,
    /// so this method is a no-op marker — the real receive path is in server.rs.
    pub async fn mark_upstream_connected(&self) {
        *self.upstream_connected.write().await = true;
    }

    pub async fn mark_upstream_disconnected(&self) {
        *self.upstream_connected.write().await = false;
    }

    pub async fn is_upstream_connected(&self) -> bool {
        *self.upstream_connected.read().await
    }

    pub async fn is_downstream_connected(&self) -> bool {
        *self.downstream_connected.read().await
    }

    /// Probe the downstream peer's health endpoint.
    pub async fn check_downstream_health(&self) -> bool {
        let Some(ref downstream) = self.config.network.downstream_peer else {
            return true; // tail has no downstream
        };

        let url = format!("{}/health", downstream);
        match self.client.get(&url).send().await {
            Ok(resp) => {
                let connected = resp.status().is_success();
                *self.downstream_connected.write().await = connected;
                connected
            }
            Err(_) => {
                *self.downstream_connected.write().await = false;
                false
            }
        }
    }

    /// Probe the upstream peer's health endpoint.
    pub async fn check_upstream_health(&self) -> bool {
        let Some(ref upstream) = self.config.network.upstream_peer else {
            return true; // head has no upstream
        };

        let url = format!("{}/health", upstream);
        match self.client.get(&url).send().await {
            Ok(resp) => {
                let connected = resp.status().is_success();
                *self.upstream_connected.write().await = connected;
                connected
            }
            Err(_) => {
                *self.upstream_connected.write().await = false;
                false
            }
        }
    }
}
