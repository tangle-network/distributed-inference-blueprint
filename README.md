# Distributed inference blueprint

Serve a model through vLLM's distributed engine and assign participating operators model layers on Tangle. vLLM owns weights, tokenization, GPU execution and transport between Ray workers. The operator sends complete OpenAI requests to that engine's HTTP head and returns its output and metered usage.

Configure the engine with the core serving recipe and vLLM's supported distributed executor. Set `[pipeline].vllm_endpoint` to its HTTP head. Set `model_id`, `layer_start`, `layer_end`, and `total_layers` to this operator's allocation; GPU capability is a catalog class, count and VRAM per device. Core owns startup, registration payload transport, QoS, billing and the recipe catalog. GPU detection/catalog integration follows the SDK release; configured GPU facts must come from the actual allocation until then.

The head serves non-streaming `POST /v1/chat/completions`; other positions refuse customer inference. Responses preserve the engine's choices and usage. Missing usage is an upstream error. `/health` checks the engine and returns 503 when unavailable. `/v1/pipeline/status` reports its health and completed requests. The removed activation endpoints return 410: vLLM has no public `/process_layers` endpoint.

The blueprint owner creates each on-chain pipeline with a model and a GPU requirement. `joinPipeline` checks the registered model, layer range, GPU class, VRAM per GPU, and the proportional GPU count. Operators sign their own joins and leaves. Core and the Tangle protocol own payments. Unregistering an operator invalidates coverage until valid membership is restored. Join/leave chain jobs return an error directing callers to those signer-authorized BSM methods.

Set the chain, server, billing, GPU and pipeline sections through `DIST_INF_` environment variables or `OperatorConfig::load`. Run `cargo run --bin distributed-inference-operator`. Customer HTTP requests use core x402 SpendAuth when billing is enabled.

Register through `deploy/register-blueprint.sh` with `RPC_URL`, `TANGLE_CORE` and the signer supplied through the vault. The wrapper uses core to read the latest release's prepared definition; `contracts/script/Deploy.s.sol` only deploys the model-specific manager. `BLUEPRINT_DEFINITION` selects an explicit prepared definition; `BLUEPRINT_BINARY` permits a real local ELF only on loopback Anvil chain 31337. The maintained release workflow uses core's packager to publish the SDK manifest, archive names and actual binary digest before publishing the prepared definition.

Run `cargo test`, `cargo clippy --all-targets -- -D warnings`, and `cd contracts && forge test` through `beelink-gate`. Integration tests use the actual HTTP adapter with only the engine process replaced at its HTTP boundary, plus deployed Solidity contracts with foreign-signer attempts.
