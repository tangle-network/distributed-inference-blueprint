use std::sync::Arc;

use tokio::sync::Semaphore;
use wiremock::{
    matchers::{method, path},
    Mock, MockServer, ResponseTemplate,
};

use distributed_inference::config::{
    BillingConfig, GpuConfig, NetworkConfig, OperatorConfig, PipelineConfig, ServerConfig,
    TangleConfig,
};
use distributed_inference::network::PipelineNetwork;
use distributed_inference::pipeline::{self, PipelineManager};

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
            operator_key: "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
                .into(),
            tangle_core: "0x0000000000000000000000000000000000000000".into(),
            shielded_credits: "0x0000000000000000000000000000000000000000".into(),
            blueprint_id: 1,
            service_id: Some(1),
        },
        pipeline: PipelineConfig {
            model_id: "test-model".into(),
            layer_start: 0,
            layer_end: 32,
            total_layers: 32,
            vllm_endpoint: format!("http://127.0.0.1:{vllm_port}"),
            max_model_len: 4096,
            hf_token: None,
            startup_timeout_secs: 5,
        },
        network: NetworkConfig {
            listen_addr: "127.0.0.1".into(),
            listen_port: 0,
            upstream_peer: None,
            downstream_peer: None,
            activation_timeout_ms: 5000,
            max_activation_bytes: 256 * 1024 * 1024,
        },
        server: ServerConfig {
            host: "127.0.0.1".into(),
            port: 0,
            max_concurrent_requests: 32,
            max_request_body_bytes: 16 * 1024 * 1024,
            stream_timeout_secs: 30,
        },
        billing: BillingConfig {
            price_per_token: 1,
            billing_required: false,
            max_spend_per_request: 1000000,
            min_credit_balance: 100,
            min_charge_amount: 0,
            clock_skew_tolerance_secs: 30,
            payment_token_address: None,
        },
        gpu: GpuConfig {
            gpu_count: 0,
            total_vram_mib: 0,
            gpu_model: None,
        },
        qos: None,
    }
}

async fn start_test_server(
    config: OperatorConfig,
) -> (u16, tokio::sync::watch::Sender<bool>, tokio::task::JoinHandle<()>) {
    let server_port = free_port();
    let mut config = config;
    config.server.port = server_port;
    let config = Arc::new(config);

    let network = Arc::new(PipelineNetwork::new(config.clone()));
    let pipeline = Arc::new(PipelineManager::new(config.clone(), network.clone()));
    let semaphore = Arc::new(Semaphore::new(32));
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

    let state = distributed_inference::server::AppState {
        config,
        pipeline,
        network,
        semaphore,
    };

    let handle = distributed_inference::server::start(state, shutdown_rx)
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    (server_port, shutdown_tx, handle)
}

// -- Tests --

#[tokio::test]
async fn test_health_check_single_operator() {
    // Single operator = head + tail, no downstream needed
    let mock_vllm = MockServer::start().await;
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
async fn test_chat_completions_single_operator() {
    let mock_vllm = MockServer::start().await;

    // Mock the vLLM shard's /process_layers endpoint
    Mock::given(method("POST"))
        .and(path("/process_layers"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "request_id": "test-req",
            "shape": [1, 5, 1],
            "data": hex::encode(b"Hello world!"),
            "metadata": {
                "prompt": "test",
                "max_tokens": 512,
                "temperature": 0.7,
                "stream": false
            }
        })))
        .mount(&mock_vllm)
        .await;

    let config = test_config_single_operator(mock_vllm.address().port());
    let (port, _tx, _h) = start_test_server(config).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
        .json(&serde_json::json!({
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 512,
            "temperature": 0.7,
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "chat.completion");
    assert!(body["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert!(body["choices"][0]["message"]["content"].as_str().is_some());
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
    assert!(body["error"]["message"].as_str().unwrap().contains("SpendAuth"));
}

#[tokio::test]
async fn test_receive_activations() {
    let mock_vllm = MockServer::start().await;

    // Mock the vLLM shard
    Mock::given(method("POST"))
        .and(path("/process_layers"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "request_id": "act-req-1",
            "shape": [1, 3, 1],
            "data": hex::encode(b"processed"),
            "metadata": {
                "prompt": "test",
                "max_tokens": 100,
                "temperature": 0.5,
                "stream": false
            }
        })))
        .mount(&mock_vllm)
        .await;

    let config = test_config_single_operator(mock_vllm.address().port());
    let (port, _tx, _h) = start_test_server(config).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{port}/v1/pipeline/activations"))
        .json(&serde_json::json!({
            "request_id": "act-req-1",
            "shape": [1, 5, 1],
            "data": hex::encode(b"input activations"),
            "metadata": {
                "prompt": "test",
                "max_tokens": 100,
                "temperature": 0.5,
                "stream": false
            }
        }))
        .send()
        .await
        .unwrap();

    // Single operator is tail, so it returns the final output as JSON
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["request_id"], "act-req-1");
    assert!(body["data"].as_str().is_some());
}

// -- Unit tests for calculate_layer_range --

#[test]
fn test_calculate_layer_range_even() {
    let (start, end) = pipeline::calculate_layer_range(0, 4, 100);
    assert_eq!((start, end), (0, 25));

    let (start, end) = pipeline::calculate_layer_range(3, 4, 100);
    assert_eq!((start, end), (75, 100));
}

#[test]
fn test_calculate_layer_range_uneven() {
    let (start, end) = pipeline::calculate_layer_range(0, 3, 80);
    assert_eq!((start, end), (0, 26));

    let (start, end) = pipeline::calculate_layer_range(2, 3, 80);
    assert_eq!((start, end), (53, 80));
}

#[test]
fn test_calculate_layer_range_single() {
    let (start, end) = pipeline::calculate_layer_range(0, 1, 100);
    assert_eq!((start, end), (0, 100));
}

#[test]
fn test_config_is_head_tail() {
    let mock_port = 9999;
    let config = test_config_single_operator(mock_port);
    assert!(config.is_pipeline_head());
    assert!(config.is_pipeline_tail());
    assert!((config.layer_fraction() - 1.0).abs() < f64::EPSILON);
}
