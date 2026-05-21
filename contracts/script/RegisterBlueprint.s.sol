// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import { Script, console2 } from "forge-std/Script.sol";
import { Types } from "tnt-core/libraries/Types.sol";
import { DistributedInferenceBSM } from "../src/DistributedInferenceBSM.sol";

/// @notice Minimal interface for Tangle blueprint registration.
interface ITangle {
    function createBlueprint(Types.BlueprintDefinition calldata def) external returns (uint64);
}

/// @title RegisterBlueprint
/// @notice Deploys DistributedInferenceBSM and registers the
///         distributed-inference blueprint on Tangle in a single broadcast.
/// @dev    Run via: `forge script contracts/script/RegisterBlueprint.s.sol
///         --rpc-url $RPC_URL --broadcast --slow`
///
///         The BSM has no constructor args and no initializer — it is a
///         plain (non-upgradeable) coordinator deployed directly. Pricing and
///         payment splitting are handled inside the BSM proportionally to
///         layer ranges; the blueprint itself is configured for event-driven
///         pricing so each inference request settles independently.
contract RegisterBlueprint is Script {
    // ─────────────────────────────────────────────────────────────────────────
    // Defaults — overridable via env vars for non-anvil chains.
    // ─────────────────────────────────────────────────────────────────────────

    // Anvil well-known deployer key (default when no PRIVATE_KEY env is set).
    uint256 constant DEFAULT_DEPLOYER_KEY =
        0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80;

    // Tangle protocol address on a LocalTestnet anvil snapshot. For real
    // chains (Base Sepolia, mainnet) pass TANGLE_CORE via env.
    address constant DEFAULT_TANGLE = 0xCf7Ed3AccA5a467e9e704C703E8D87F634fB0Fc9;

    // USDC on Base Sepolia. Documented here for parity with sibling
    // inference blueprints; the DistributedInferenceBSM does not itself hold
    // a payment-token address (per-pipeline pricing is denominated in base
    // units of whatever token the service request supplies), so this is
    // emitted purely as deployment metadata for downstream tooling.
    address constant DEFAULT_PAYMENT_TOKEN = 0x036CbD53842c5426634e7929541eC2318f3dCF7e;

    function run() external {
        uint256 deployerKey = vm.envOr("PRIVATE_KEY", DEFAULT_DEPLOYER_KEY);
        address tangleAddr = vm.envOr("TANGLE_CORE", DEFAULT_TANGLE);
        address paymentToken = vm.envOr("PAYMENT_TOKEN", DEFAULT_PAYMENT_TOKEN);

        ITangle tangle = ITangle(tangleAddr);

        vm.startBroadcast(deployerKey);

        // ── Deploy DistributedInferenceBSM ──────────────────────────────────
        // The BSM pre-binds `tangleCore` in its constructor so the subsequent
        // `Tangle.createBlueprint` → `onBlueprintCreated` hook is a no-op
        // rebind to the same address (see {DistributedInferenceBSM.onBlueprintCreated}).
        DistributedInferenceBSM bsm = new DistributedInferenceBSM(tangleAddr);

        // ── Register on Tangle ──────────────────────────────────────────────
        uint64 blueprintId = tangle.createBlueprint(_buildDefinition(address(bsm)));

        vm.stopBroadcast();

        // ── Output for bash wrapper parsing ─────────────────────────────────
        console2.log("DEPLOY_DISTRIBUTED_BSM=%s", vm.toString(address(bsm)));
        console2.log("DEPLOY_DISTRIBUTED_BLUEPRINT_ID=%s", vm.toString(blueprintId));
        console2.log("DEPLOY_DISTRIBUTED_PAYMENT_TOKEN=%s", vm.toString(paymentToken));
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Blueprint Definition builder
    // ═════════════════════════════════════════════════════════════════════════

    function _buildDefinition(address manager) internal pure returns (Types.BlueprintDefinition memory def) {
        def.metadataUri = "https://github.com/tangle-network/distributed-inference-blueprint";
        // metadataHash is a digest of the canonical metadata JSON. Until that
        // payload is pinned via IPFS, derive it from the metadataUri so the
        // value is deterministic + traceable.
        def.metadataHash = keccak256(bytes(def.metadataUri));
        def.manager = manager;
        def.masterManagerRevision = 0;
        def.hasConfig = true;

        // Dynamic membership: operators join/leave specific pipelines after
        // service activation. Event-driven pricing: payments settle per
        // inference request, with the BSM splitting them proportionally to
        // each operator's layer span.
        def.config = Types.BlueprintConfig({
            membership: Types.MembershipModel.Dynamic,
            pricing: Types.PricingModel.EventDriven,
            minOperators: 2, // pipeline parallelism requires >=2 stages
            maxOperators: 0, // unbounded — large models may span many stages
            subscriptionRate: 0,
            subscriptionInterval: 0,
            eventRate: 0 // operators negotiate price per call via the BSM
        });

        def.metadata = Types.BlueprintMetadata({
            name: "Distributed Inference Blueprint",
            description: "Pipeline-parallel inference for very large models split across multiple Tangle operators",
            author: "Tangle Network",
            category: "AI/Inference",
            codeRepository: "https://github.com/tangle-network/distributed-inference-blueprint",
            logo: "",
            website: "https://tangle.tools",
            license: "MIT",
            profilingData: ""
        });

        def.jobs = _buildJobs();

        // Operator registration payload — see DistributedInferenceBSM.onRegister:
        //   abi.decode(_, (string,uint32,uint32,uint32,uint32,uint32,string))
        //     modelId, layerStart, layerEnd, totalLayers, gpuCount, totalVramMib, endpoint
        // On-chain schema is kept empty to match the pattern used by sibling
        // inference blueprints; the canonical shape is enforced by the BSM
        // and by the Rust operator's `JoinPipelineRequest`.
        def.registrationSchema = "";
        def.requestSchema = "";

        def.sources = new Types.BlueprintSource[](1);
        Types.BlueprintBinary[] memory bins = new Types.BlueprintBinary[](1);
        bins[0] = Types.BlueprintBinary({
            arch: Types.BlueprintArchitecture.Amd64,
            os: Types.BlueprintOperatingSystem.Linux,
            name: "distributed-inference-blueprint",
            sha256: bytes32(uint256(0xdeadbeef))
        });
        def.sources[0] = Types.BlueprintSource({
            kind: Types.BlueprintSourceKind.Native,
            container: Types.ImageRegistrySource("", "", ""),
            wasm: Types.WasmSource(Types.WasmRuntime.Unknown, Types.BlueprintFetcherKind.None, "", ""),
            native: Types.NativeSource(
                Types.BlueprintFetcherKind.None,
                "file:///target/release/distributed-inference-blueprint",
                "./target/release/distributed-inference-blueprint"
            ),
            testing: Types.TestingSource(
                "distributed-inference-blueprint-bin", "distributed-inference-blueprint", "."
            ),
            binaries: bins
        });

        def.supportedMemberships = new Types.MembershipModel[](1);
        def.supportedMemberships[0] = Types.MembershipModel.Dynamic;
    }

    function _buildJobs() internal pure returns (Types.JobDefinition[] memory jobs) {
        // Job IDs mirror `operator/src/lib.rs`:
        //   0 = INFERENCE_JOB
        //   1 = JOIN_PIPELINE_JOB
        //   2 = LEAVE_PIPELINE_JOB
        // The Rust operator enforces these shapes; on-chain schemas are kept
        // empty to match the pattern used by sibling inference blueprints
        // where params/result types live with the running operator, not the
        // Blueprint registry.
        jobs = new Types.JobDefinition[](3);

        jobs[0] = Types.JobDefinition({
            name: "inference",
            description: "Run pipeline-parallel inference; routed via the head stage and forwarded down the operator pipeline",
            metadataUri: "",
            paramsSchema: "",
            resultSchema: ""
        });

        jobs[1] = Types.JobDefinition({
            name: "join_pipeline",
            description: "Join an existing pipeline group for a model with an assigned contiguous layer range",
            metadataUri: "",
            paramsSchema: "",
            resultSchema: ""
        });

        jobs[2] = Types.JobDefinition({
            name: "leave_pipeline",
            description: "Leave a pipeline group between inference requests; deactivates the pipeline until coverage is restored",
            metadataUri: "",
            paramsSchema: "",
            resultSchema: ""
        });
    }
}
