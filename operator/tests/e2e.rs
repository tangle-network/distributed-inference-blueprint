use std::sync::Arc;

use wiremock::{
    matchers::{method, path},
    Mock, MockServer, ResponseTemplate,
};

use distributed_inference::config::{
    BillingConfig, DistributedConfig, GpuConfig, OperatorConfig, ServerConfig, TangleConfig,
};
use distributed_inference::pipeline::PipelineManager;
use distributed_inference::server::DistributedBackend;
use distributed_inference::{AppStateBuilder, BillingClient, NonceStore};

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// Build a config for a single-operator pipeline (head + tail, all layers).
fn test_config_single_operator(vllm_port: u16) -> OperatorConfig {
    OperatorConfig {
        tangle: TangleConfig {
            rpc_url: "http://localhost:8545".into(),
            chain_id: 31337,
            operator_key: "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80".into(),
            shielded_credits: "0x0000000000000000000000000000000000000000".into(),
            blueprint_id: 1,
            service_id: Some(1),
        },
        pipeline: DistributedConfig {
            model_id: "test-model".into(),
            layer_start: 0,
            layer_end: 32,
            total_layers: 32,
            vllm_endpoint: format!("http://127.0.0.1:{vllm_port}"),
            price_per_input_token: 1,
            price_per_output_token: 1,
        },
        server: ServerConfig {
            host: "127.0.0.1".into(),
            port: 0,
            max_concurrent_requests: 32,
            max_request_body_bytes: 16 * 1024 * 1024,
            stream_timeout_secs: 30,
            idle_chunk_timeout_secs: 30,
            max_line_buf_bytes: 1024 * 1024,
            max_per_account_requests: 0,
        },
        billing: BillingConfig {
            payment_rails: tangle_inference_core::PaymentRails::SHIELDED,
            billing_required: false,
            max_spend_per_request: 1000000,
            min_credit_balance: 100,
            min_charge_amount: 0,
            claim_max_retries: 3,
            clock_skew_tolerance_secs: 30,
            max_gas_price_gwei: 0,
            nonce_store_path: None,
            direct_replay_store_path: None,
            payment_token_address: None,
        },
        gpu: GpuConfig {
            expected_gpu_count: 0,
            min_vram_mib: 0,
            gpu_model: None,
            monitor_interval_secs: 30,
        },
        qos: None,
    }
}

async fn start_test_server(
    config: OperatorConfig,
) -> (
    u16,
    tokio::sync::watch::Sender<bool>,
    tokio::task::JoinHandle<()>,
) {
    let server_port = free_port();
    let mut config = config;
    config.server.port = server_port;
    let config = Arc::new(config);

    let pipeline = Arc::new(PipelineManager::new(config.clone()));
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

    let billing = Arc::new(
        BillingClient::new(&config.tangle, &config.billing)
            .expect("failed to build test billing client"),
    );
    let operator_address = billing.operator_address();
    let nonce_store = Arc::new(NonceStore::load(None));
    let backend = DistributedBackend::new(config.clone(), pipeline);
    let state = AppStateBuilder::new()
        .billing(billing)
        .nonce_store(nonce_store)
        .server_config(Arc::new(config.server.clone()))
        .billing_config(Arc::new(config.billing.clone()))
        .tangle_config(Arc::new(config.tangle.clone()))
        .operator_address(operator_address)
        .backend(backend)
        .build()
        .expect("build AppState");

    let handle = distributed_inference::server::start_head(state, shutdown_rx)
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    (server_port, shutdown_tx, handle)
}

// -- Tests --

#[tokio::test]
async fn test_health_check_single_operator() {
    let mock_vllm = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&mock_vllm)
        .await;
    let config = test_config_single_operator(mock_vllm.address().port());

    let (port, _tx, _h) = start_test_server(config).await;

    let resp = reqwest::get(format!("http://127.0.0.1:{port}/health"))
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");
    assert_eq!(body["model"], "test-model");
    assert_eq!(body["layers"], "0-32");
    assert_eq!(body["pipeline_head"], true);
    assert_eq!(body["pipeline_tail"], true);
}

#[tokio::test]
async fn test_pipeline_status() {
    let mock_vllm = MockServer::start().await;
    let config = test_config_single_operator(mock_vllm.address().port());

    let (port, _tx, _h) = start_test_server(config).await;

    let resp = reqwest::get(format!("http://127.0.0.1:{port}/v1/pipeline/status"))
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["model_id"], "test-model");
    assert_eq!(body["layer_start"], 0);
    assert_eq!(body["layer_end"], 32);
    assert_eq!(body["total_layers"], 32);
    assert_eq!(body["is_head"], true);
    assert_eq!(body["is_tail"], true);
}

#[tokio::test]
async fn test_chat_completions_billing_required() {
    let mock_vllm = MockServer::start().await;
    let mut config = test_config_single_operator(mock_vllm.address().port());
    config.billing.billing_required = true;

    let (port, _tx, _h) = start_test_server(config).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
        .json(&serde_json::json!({
            "messages": [{"role": "user", "content": "Hello"}],
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 402);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["error"], "payment_required");
    assert!(body["description"].as_str().unwrap().contains("SpendAuth"));
}

#[tokio::test]
async fn d9_real_engine_protocol_and_usage() {
    let engine = MockServer::start().await;
    Mock::given(method("POST")).and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id":"chatcmpl-engine", "object":"chat.completion", "created":1,
            "model":"test-model", "choices":[{"index":0,"message":{"role":"assistant","content":"engine output"},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":13,"completion_tokens":7,"total_tokens":20}
        }))).expect(1).mount(&engine).await;
    let (port, _tx, _h) =
        start_test_server(test_config_single_operator(engine.address().port())).await;
    let response = reqwest::Client::new()
        .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
        .json(&serde_json::json!({"messages":[{"role":"user","content":"hello"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let response: serde_json::Value = response.json().await.unwrap();
    assert_eq!(
        response["choices"][0]["message"]["content"],
        "engine output"
    );
    assert_eq!(response["usage"]["prompt_tokens"], 13);
    assert_eq!(response["usage"]["completion_tokens"], 7);
}
