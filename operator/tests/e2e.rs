use std::path::PathBuf;

use distributed_inference::pipeline::{
    ActivationPayload, PipelineStageConfig, PipelineStatus, RequestMetadata,
    calculate_layer_range,
};
use distributed_inference::shard::{ShardConfig, assign_layers};

// --- Pipeline Config ---

#[test]
fn pipeline_stage_config_serialization() {
    let config = PipelineStageConfig {
        layer_start: 0,
        layer_end: 25,
        total_layers: 100,
        upstream_peer: None,
        downstream_peer: Some("http://operator-b:8080".to_string()),
    };

    let json = serde_json::to_string(&config).unwrap();
    let deserialized: PipelineStageConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.layer_start, 0);
    assert_eq!(deserialized.layer_end, 25);
    assert_eq!(deserialized.total_layers, 100);
    assert!(deserialized.upstream_peer.is_none());
    assert_eq!(
        deserialized.downstream_peer.as_deref(),
        Some("http://operator-b:8080")
    );
}

#[test]
fn pipeline_stage_config_head() {
    let config = PipelineStageConfig {
        layer_start: 0,
        layer_end: 32,
        total_layers: 126,
        upstream_peer: None,
        downstream_peer: Some("http://next:8080".to_string()),
    };

    assert!(config.upstream_peer.is_none()); // head has no upstream
    assert!(config.downstream_peer.is_some());
}

#[test]
fn pipeline_stage_config_tail() {
    let config = PipelineStageConfig {
        layer_start: 95,
        layer_end: 126,
        total_layers: 126,
        upstream_peer: Some("http://prev:8080".to_string()),
        downstream_peer: None,
    };

    assert!(config.upstream_peer.is_some());
    assert!(config.downstream_peer.is_none()); // tail has no downstream
}

// --- Layer Range Calculation ---

#[test]
fn calculate_layer_range_even_split() {
    assert_eq!(calculate_layer_range(0, 4, 100), (0, 25));
    assert_eq!(calculate_layer_range(1, 4, 100), (25, 50));
    assert_eq!(calculate_layer_range(2, 4, 100), (50, 75));
    assert_eq!(calculate_layer_range(3, 4, 100), (75, 100));
}

#[test]
fn calculate_layer_range_uneven() {
    // 3 operators, 80 layers
    assert_eq!(calculate_layer_range(0, 3, 80), (0, 26));
    assert_eq!(calculate_layer_range(1, 3, 80), (26, 53));
    assert_eq!(calculate_layer_range(2, 3, 80), (53, 80));
}

#[test]
fn calculate_layer_range_single_operator() {
    assert_eq!(calculate_layer_range(0, 1, 126), (0, 126));
}

#[test]
fn layer_ranges_no_gaps_no_overlaps() {
    for total_operators in 1..=8 {
        for total_layers in [32, 56, 80, 100, 126] {
            let mut prev_end = 0u32;
            for i in 0..total_operators {
                let (start, end) = calculate_layer_range(i, total_operators, total_layers);
                assert_eq!(start, prev_end, "gap at operator {i} for {total_operators} ops, {total_layers} layers");
                assert!(end > start, "empty range at operator {i}");
                prev_end = end;
            }
            assert_eq!(prev_end, total_layers, "not all layers covered for {total_operators} ops, {total_layers} layers");
        }
    }
}

// --- Shard Config ---

#[test]
fn shard_config_fraction() {
    let cfg = ShardConfig {
        model_id: "meta-llama/Llama-3.1-405B-Instruct".to_string(),
        layer_start: 0,
        layer_end: 32,
        total_layers: 126,
        download_dir: PathBuf::from("/tmp/shards"),
    };

    assert_eq!(cfg.layer_count(), 32);
    let fraction = cfg.fraction();
    assert!((fraction - 32.0 / 126.0).abs() < f64::EPSILON);
}

#[test]
fn shard_config_estimated_vram() {
    let cfg = ShardConfig {
        model_id: "test".to_string(),
        layer_start: 0,
        layer_end: 25,
        total_layers: 100,
        download_dir: PathBuf::from("/tmp"),
    };

    // 25% of 80 GiB model (81920 MiB) = 20480 MiB + 2048 overhead = 22528
    let vram = cfg.estimated_vram_mib(81920);
    assert_eq!(vram, 20480 + 2048);
}

#[test]
fn assign_layers_matches_calculate_layer_range() {
    // Both functions should produce identical results
    for n in 1..=6 {
        for total in [32, 80, 100, 126] {
            for i in 0..n {
                assert_eq!(
                    assign_layers(i, n, total),
                    calculate_layer_range(i, n, total),
                    "mismatch at i={i}, n={n}, total={total}"
                );
            }
        }
    }
}

// --- Activation Payload ---

#[test]
fn activation_payload_serialization() {
    let payload = ActivationPayload {
        request_id: "req-abc-123".to_string(),
        shape: vec![1, 128, 4096],
        data: vec![0x00, 0x3C, 0x00, 0x40], // 2 f16 values
        metadata: RequestMetadata {
            prompt: "Hello, world!".to_string(),
            max_tokens: 256,
            temperature: 0.7,
            stream: false,
        },
    };

    let json = serde_json::to_string(&payload).unwrap();
    let deserialized: ActivationPayload = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.request_id, "req-abc-123");
    assert_eq!(deserialized.shape, vec![1, 128, 4096]);
    assert_eq!(deserialized.data, vec![0x00, 0x3C, 0x00, 0x40]);
    assert_eq!(deserialized.metadata.prompt, "Hello, world!");
    assert_eq!(deserialized.metadata.max_tokens, 256);
    assert!((deserialized.metadata.temperature - 0.7).abs() < f32::EPSILON);
    assert!(!deserialized.metadata.stream);
}

#[test]
fn pipeline_status_serialization() {
    let status = PipelineStatus {
        model_id: "meta-llama/Llama-3.1-405B".to_string(),
        layer_start: 0,
        layer_end: 32,
        total_layers: 126,
        is_head: true,
        is_tail: false,
        upstream_connected: false,
        downstream_connected: true,
        requests_processed: 42,
        pipeline_latency_ms: 150,
    };

    let json = serde_json::to_string(&status).unwrap();
    let deserialized: PipelineStatus = serde_json::from_str(&json).unwrap();

    assert!(deserialized.is_head);
    assert!(!deserialized.is_tail);
    assert_eq!(deserialized.requests_processed, 42);
    assert_eq!(deserialized.pipeline_latency_ms, 150);
}

// --- Request Metadata ---

#[test]
fn request_metadata_serialization() {
    let meta = RequestMetadata {
        prompt: "Explain quantum computing".to_string(),
        max_tokens: 512,
        temperature: 0.0,
        stream: true,
    };

    let json = serde_json::to_string(&meta).unwrap();
    let deserialized: RequestMetadata = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.prompt, "Explain quantum computing");
    assert_eq!(deserialized.max_tokens, 512);
    assert!((deserialized.temperature).abs() < f32::EPSILON);
    assert!(deserialized.stream);
}
