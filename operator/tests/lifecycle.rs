//! Full lifecycle test -- activation processing through real handler + wiremock backend.

use anyhow::{Result, ensure};
use wiremock::{MockServer, Mock, ResponseTemplate, matchers::{method, path}};

#[tokio::test]
async fn test_process_activations_direct_with_wiremock() -> Result<()> {
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/process_layers"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "activations": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
            "layer_start": 0,
            "layer_end": 25
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    distributed_inference::init_direct_for_testing(&mock_server.uri(), 0, 25);

    let input = vec![0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

    let result = distributed_inference::process_activations_direct(&input, 0, 25).await;

    match result {
        Ok(output) => {
            ensure!(output.len() == 8, "expected 8 activations, got {}", output.len());
            ensure!(
                (output[0] - 0.5).abs() < 1e-6,
                "expected first activation ~0.5, got {}",
                output[0]
            );
            ensure!(
                (output[7] - 1.2).abs() < 1e-6,
                "expected last activation ~1.2, got {}",
                output[7]
            );
        }
        Err(e) => panic!("Activation processing failed: {e}"),
    }

    mock_server.verify().await;

    Ok(())
}
