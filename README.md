# Distributed inference blueprint

Serve a model through vLLM's distributed engine and assign participating operators model layers on Tangle. vLLM owns weights, tokenization, GPU execution and transport between Ray workers. The operator sends complete OpenAI requests to that engine's HTTP head and returns its output and metered usage.

Configure the engine with the core serving recipe and vLLM's supported distributed executor. Set `[pipeline].vllm_endpoint` to its HTTP head. Set `model_id`, `layer_start`, `layer_end`, and `total_layers` to this operator's allocation; GPU capability is a catalog class, count and VRAM per device. Core owns startup, registration payload transport, QoS, billing and the recipe catalog. GPU detection/catalog integration follows the SDK release; configured GPU facts must come from the actual allocation until then.

The head serves non-streaming `POST /v1/chat/completions`; other positions refuse customer inference. Responses preserve the engine's choices and usage. Missing usage is an upstream error. `/health` checks the engine and returns 503 when unavailable. `/v1/pipeline/status` reports its health and completed requests. The removed activation endpoints return 410: vLLM has no public `/process_layers` endpoint.

The blueprint owner creates each on-chain pipeline with a model and a GPU requirement. `joinPipeline` checks the registered model, layer range, GPU class, VRAM per GPU, and the proportional GPU count. Operators sign their own joins and leaves. Core and the Tangle protocol own payments. Unregistering an operator invalidates coverage until valid membership is restored. Join/leave chain jobs return an error directing callers to those signer-authorized BSM methods.

Set the chain, server, billing, GPU and pipeline sections through `DIST_INF_` environment variables or `OperatorConfig::load`. Run `cargo run --bin distributed-inference-operator`. Customer HTTP requests use core x402 SpendAuth when billing is enabled.

Deploy the manager with [Deploy.s.sol](contracts/script/Deploy.s.sol), `TANGLE_CORE` and a Foundry keystore, set its address in [definition.json](deploy/definition.json), then register with the [cargo-tangle deployment guide](https://github.com/tangle-network/blueprint#deploying-to-testnetmainnet) using the same wallet in the SDK keystore. The CLI owns registration encoding and metadata hashing; `cargo tangle blueprint deploy tangle --help` lists its options. [Release](.github/workflows/release.yml) publishes native cargo-dist archives and `dist-manifest.json`.

Run `cargo test`, `cargo clippy --all-targets -- -D warnings`, and `cd contracts && forge test` through `beelink-gate`. Integration tests use the actual HTTP adapter with only the engine process replaced at its HTTP boundary, plus deployed Solidity contracts with foreign-signer attempts.
