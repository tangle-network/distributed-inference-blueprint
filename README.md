<p align="center">
  <img src="https://raw.githubusercontent.com/tangle-network/tangle/main/assets/Tangle%20Banner.png" alt="Tangle Network Banner" width="100%">
</p>

# Distributed Inference Blueprint

Pipeline-parallel inference for models too large for a single GPU. Split 400B+ parameter models across multiple Tangle operators, where each operator serves a contiguous range of transformer layers.

## How It Works

```
User Request
    |
    v
Operator A (layers 0-31)    ──activations──>
Operator B (layers 32-63)   ──activations──>
Operator C (layers 64-95)   ──activations──>
Operator D (layers 96-126)  ──response──>  User
```

Each operator:
1. Registers on-chain with its GPU capabilities and assigned layer range
2. Downloads only the model weights for its layers (not the full model)
3. Runs a local vLLM instance serving its layer shard
4. Receives intermediate activations from the upstream operator
5. Processes through its layers and forwards to the downstream operator
6. The tail operator produces the final output and returns it through the pipeline

Payment is split proportionally: an operator serving 25/100 layers gets 25% of the per-token price.

## Components

| Component | Path | Description |
|---|---|---|
| Operator binary | `operator/src/main.rs` | Startup, registration, BlueprintRunner |
| Router + jobs | `operator/src/lib.rs` | INFERENCE_JOB, JOIN_PIPELINE_JOB, LEAVE_PIPELINE_JOB |
| Pipeline manager | `operator/src/pipeline.rs` | Activation processing, forwarding, topology |
| Model sharding | `operator/src/shard.rs` | Selective layer download from HuggingFace |
| Peer networking | `operator/src/network.rs` | Point-to-point activation transfer between stages |
| HTTP server | `operator/src/server.rs` | OpenAI-compatible API (head), activation endpoints |
| Config | `operator/src/config.rs` | Pipeline, network, billing, GPU configuration |
| QoS heartbeat | `operator/src/qos.rs` | layers_served, pipeline_latency_ms, peer connectivity |
| Health | `operator/src/health.rs` | GPU detection via nvidia-smi |
| BSM contract | `contracts/src/DistributedInferenceBSM.sol` | Pipeline groups, layer validation, payment splitting |

## Supported Models

| Model | Parameters | Layers | Min Operators | Min VRAM per Operator |
|---|---|---|---|---|
| Llama 3.1 405B | 405B | 126 | 4 | ~210 GiB (4x H100 80GB) |
| Mixtral 8x22B | 176B | 56 | 2 | ~180 GiB (2x H100 80GB) |
| Llama 3.1 70B | 70B | 80 | 2 | ~70 GiB (2x A100 80GB) |
| DeepSeek-V2 | 236B | 60 | 4 | ~120 GiB |

## TEE Support

All pipeline stages run inside TEE enclaves (TeeLayer). Intermediate activations are signed by each operator, providing verifiable computation guarantees across the pipeline.

## Quick Start

### 1. Deploy the BSM contract

```bash
# Deploy DistributedInferenceBSM to Tangle
forge create contracts/src/DistributedInferenceBSM.sol:DistributedInferenceBSM --rpc-url $TANGLE_RPC
```

### 2. Create a pipeline

```bash
# Create a pipeline for Llama 405B (126 layers, 4 operators)
cast send $BSM_ADDRESS "createPipeline(string,uint32,uint256)" \
  "meta-llama/Llama-3.1-405B-Instruct" 126 1000 --rpc-url $TANGLE_RPC
```

### 3. Configure and start operators

Each operator needs a config specifying its layer range:

```toml
# Operator A config (head)
[pipeline]
model_id = "meta-llama/Llama-3.1-405B-Instruct"
layer_start = 0
layer_end = 32
total_layers = 126
vllm_endpoint = "http://127.0.0.1:8000"

[network]
listen_addr = "0.0.0.0"
listen_port = 9090
downstream_peer = "http://operator-b:8080"
# upstream_peer not set (this is the head)
```

```bash
# Build and run
cargo build --release
DIST_INF__PIPELINE__MODEL_ID="meta-llama/Llama-3.1-405B-Instruct" \
DIST_INF__PIPELINE__LAYER_START=0 \
DIST_INF__PIPELINE__LAYER_END=32 \
./target/release/distributed-inference-operator
```

### 4. Send inference requests

```bash
# Requests go to the head operator (layer_start == 0)
curl http://operator-a:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-405B-Instruct",
    "messages": [{"role": "user", "content": "Explain pipeline parallelism."}],
    "max_tokens": 256
  }'

# Check pipeline status on any operator
curl http://operator-b:8080/v1/pipeline/status
```

## Related Repos

- [vllm-inference-blueprint](https://github.com/tangle-network/vllm-inference-blueprint) -- Single-operator vLLM inference
- [blueprint-sdk](https://github.com/tangle-network/blueprint) -- Tangle Blueprint SDK
- [tangle-router](https://github.com/tangle-network/tangle-router) -- AI model routing and pricing
