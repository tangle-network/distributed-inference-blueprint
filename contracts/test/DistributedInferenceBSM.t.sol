// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import { Test } from "forge-std/Test.sol";
import { DistributedInferenceBSM } from "../src/DistributedInferenceBSM.sol";

contract DistributedInferenceBSMTest is Test {
    DistributedInferenceBSM public bsm;

    address public operator1 = address(0x1111);
    address public operator2 = address(0x2222);
    address public operator3 = address(0x3333);
    address public operator4 = address(0x4444);
    address public unregistered = address(0x9999);

    function setUp() public {
        bsm = new DistributedInferenceBSM();

        // Register 4 operators
        _registerOperator(operator1, 0, 25, "https://op1.example.com");
        _registerOperator(operator2, 25, 50, "https://op2.example.com");
        _registerOperator(operator3, 50, 75, "https://op3.example.com");
        _registerOperator(operator4, 75, 100, "https://op4.example.com");
    }

    // --- Pipeline Creation ---

    function test_createPipeline() public {
        uint64 pipelineId = bsm.createPipeline("meta-llama/Llama-3.1-405B", 100, 1000);
        assertEq(pipelineId, 0);
        assertEq(bsm.pipelineCount(), 1);

        (string memory modelId, uint32 totalLayers, bool active,) = bsm.pipelines(pipelineId);
        assertEq(keccak256(bytes(modelId)), keccak256(bytes("meta-llama/Llama-3.1-405B")));
        assertEq(totalLayers, 100);
        assertFalse(active); // not active until fully covered
    }

    function test_createMultiplePipelines() public {
        bsm.createPipeline("model-a", 80, 500);
        bsm.createPipeline("model-b", 126, 2000);
        assertEq(bsm.pipelineCount(), 2);
    }

    // --- Join Pipeline ---

    function test_joinPipeline_singleOperator() public {
        uint64 pid = bsm.createPipeline("model", 100, 1000);

        vm.prank(operator1);
        bsm.joinPipeline(pid, 0, 100);

        // Should be active (all layers covered by one operator)
        assertTrue(bsm.isFullyCovered(pid));
    }

    function test_joinPipeline_fullCoverage() public {
        uint64 pid = bsm.createPipeline("model", 100, 1000);

        vm.prank(operator1);
        bsm.joinPipeline(pid, 0, 25);

        vm.prank(operator2);
        bsm.joinPipeline(pid, 25, 50);

        vm.prank(operator3);
        bsm.joinPipeline(pid, 50, 75);

        assertFalse(bsm.isFullyCovered(pid)); // still missing 75-100

        vm.prank(operator4);
        bsm.joinPipeline(pid, 75, 100);

        assertTrue(bsm.isFullyCovered(pid));
    }

    function test_joinPipeline_layerRangeOverlap() public {
        uint64 pid = bsm.createPipeline("model", 100, 1000);

        vm.prank(operator1);
        bsm.joinPipeline(pid, 0, 50);

        // Overlapping range: 25-75 overlaps with 0-50
        vm.prank(operator2);
        vm.expectRevert(
            abi.encodeWithSelector(DistributedInferenceBSM.LayerRangeOverlap.selector, pid, 25, 75)
        );
        bsm.joinPipeline(pid, 25, 75);
    }

    function test_joinPipeline_invalidRange() public {
        uint64 pid = bsm.createPipeline("model", 100, 1000);

        // start >= end
        vm.prank(operator1);
        vm.expectRevert(
            abi.encodeWithSelector(DistributedInferenceBSM.LayerRangeInvalid.selector, 50, 50, 100)
        );
        bsm.joinPipeline(pid, 50, 50);
    }

    function test_joinPipeline_exceedsTotalLayers() public {
        uint64 pid = bsm.createPipeline("model", 100, 1000);

        vm.prank(operator1);
        vm.expectRevert(
            abi.encodeWithSelector(DistributedInferenceBSM.LayerRangeInvalid.selector, 0, 150, 100)
        );
        bsm.joinPipeline(pid, 0, 150);
    }

    function test_joinPipeline_unregisteredOperator() public {
        uint64 pid = bsm.createPipeline("model", 100, 1000);

        vm.prank(unregistered);
        vm.expectRevert(
            abi.encodeWithSelector(DistributedInferenceBSM.OperatorNotRegistered.selector, unregistered)
        );
        bsm.joinPipeline(pid, 0, 25);
    }

    function test_joinPipeline_alreadyInPipeline() public {
        uint64 pid = bsm.createPipeline("model", 100, 1000);

        vm.prank(operator1);
        bsm.joinPipeline(pid, 0, 25);

        vm.prank(operator1);
        vm.expectRevert(
            abi.encodeWithSelector(DistributedInferenceBSM.OperatorAlreadyInPipeline.selector, pid, operator1)
        );
        bsm.joinPipeline(pid, 25, 50);
    }

    // --- Leave Pipeline ---

    function test_leavePipeline() public {
        uint64 pid = _createFullPipeline();

        vm.prank(operator4);
        bsm.leavePipeline(pid);

        // Pipeline should no longer be fully covered
        assertFalse(bsm.isFullyCovered(pid));

        address[] memory ops = bsm.getPipelineOperators(pid);
        assertEq(ops.length, 3);
    }

    function test_leavePipeline_notInPipeline() public {
        uint64 pid = bsm.createPipeline("model", 100, 1000);

        vm.prank(operator1);
        vm.expectRevert(
            abi.encodeWithSelector(DistributedInferenceBSM.OperatorNotInPipeline.selector, pid, operator1)
        );
        bsm.leavePipeline(pid);
    }

    // --- Payment Split ---

    function test_getOperatorPricing() public {
        uint64 pid = _createFullPipeline();

        uint256 price1 = bsm.getOperatorPricing(pid, operator1);
        uint256 price2 = bsm.getOperatorPricing(pid, operator2);

        // Each operator covers 25/100 = 25% of layers, so gets 25% of base price
        assertEq(price1, 250); // 1000 * 25 / 100
        assertEq(price2, 250);
    }

    function test_recordUsage_distributesPayment() public {
        uint64 pid = _createFullPipeline();

        bsm.recordUsage(pid, 1000, 10_000);

        // Each operator should get 25% of 10_000 = 2_500
        assertEq(bsm.operatorRevenue(pid, operator1), 2_500);
        assertEq(bsm.operatorRevenue(pid, operator2), 2_500);
        assertEq(bsm.operatorRevenue(pid, operator3), 2_500);
        assertEq(bsm.operatorRevenue(pid, operator4), 2_500);
    }

    function test_recordUsage_unevenLayers() public {
        uint64 pid = bsm.createPipeline("model", 100, 10_000);

        // Operator1: 60 layers, operator2: 40 layers
        vm.prank(operator1);
        bsm.joinPipeline(pid, 0, 60);

        vm.prank(operator2);
        bsm.joinPipeline(pid, 60, 100);

        bsm.recordUsage(pid, 500, 10_000);

        assertEq(bsm.operatorRevenue(pid, operator1), 6_000); // 60% of 10_000
        assertEq(bsm.operatorRevenue(pid, operator2), 4_000); // 40% of 10_000
    }

    // --- Operator Registration ---

    function test_registerOperator() public view {
        (uint32 layerStart, uint32 layerEnd, uint32 totalVramMib, string memory endpoint, bool active) =
            bsm.operatorCaps(operator1);
        assertEq(layerStart, 0);
        assertEq(layerEnd, 25);
        assertTrue(active);
        assertTrue(totalVramMib > 0);
        assertTrue(bytes(endpoint).length > 0);
    }

    // --- View Functions ---

    function test_getPipelineOperators_ordered() public {
        uint64 pid = _createFullPipeline();
        address[] memory ops = bsm.getPipelineOperators(pid);

        assertEq(ops.length, 4);
        // Should be sorted by layer range (head -> tail)
        assertEq(ops[0], operator1);
        assertEq(ops[1], operator2);
        assertEq(ops[2], operator3);
        assertEq(ops[3], operator4);
    }

    // --- Helpers ---

    function _registerOperator(
        address op,
        uint32 layerStart,
        uint32 layerEnd,
        string memory endpoint
    ) internal {
        bytes memory regData = abi.encode(
            "meta-llama/Llama-3.1-405B",
            layerStart,
            layerEnd,
            uint32(100), // total layers
            uint32(2),   // gpu count
            uint32(80_000), // vram
            endpoint
        );
        bsm.onRegister(op, regData);
    }

    function _createFullPipeline() internal returns (uint64 pid) {
        pid = bsm.createPipeline("meta-llama/Llama-3.1-405B", 100, 1000);

        vm.prank(operator1);
        bsm.joinPipeline(pid, 0, 25);

        vm.prank(operator2);
        bsm.joinPipeline(pid, 25, 50);

        vm.prank(operator3);
        bsm.joinPipeline(pid, 50, 75);

        vm.prank(operator4);
        bsm.joinPipeline(pid, 75, 100);
    }
}
