// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;
import {Test} from "forge-std/Test.sol";
import {DistributedInferenceBSM} from "../src/DistributedInferenceBSM.sol";
contract DistributedInferenceBSMTest is Test {
    DistributedInferenceBSM bsm;
    address alice = address(0x1111);
    address bob = address(0x2222);
    address attacker = address(0x9999);
    function setUp() public {
        bsm = new DistributedInferenceBSM(address(this));
        bsm.onBlueprintCreated(1,address(this),address(this));
    }
    function register(address operator, uint32 count, uint32 vram) internal {
        bsm.onRegister(operator, abi.encode("model",uint32(0),uint32(100),uint32(100),count,vram,"nvidia-h100","https://operator.example.com"));
    }
    function create() internal returns(uint64) {
        return bsm.createPipeline("model",100,DistributedInferenceBSM.GpuRequirement("nvidia-h100",4,80000));
    }
    function test_D9_unauthorizedMoneyAndCreation() public {
        register(alice,4,80000);
        uint64 id=create();
        vm.prank(alice); bsm.joinPipeline(id,0,100);
        vm.prank(attacker);
        (bool legacyUsage,) = address(bsm).call(abi.encodeWithSignature("recordUsage(uint64,uint256,uint256)",id,1000,10000));
        assertFalse(legacyUsage, "removed unsigned revenue selector must revert");
        vm.prank(attacker); vm.expectRevert();
        bsm.createPipeline("attacker",100,DistributedInferenceBSM.GpuRequirement("nvidia-h100",1,1));
    }
    function test_D9_smallGpuCannotJoinEvenWithEnoughTotalVram() public {
        register(alice,8,40000);
        uint64 id=create();
        vm.prank(alice); vm.expectRevert(abi.encodeWithSelector(DistributedInferenceBSM.InsufficientVram.selector,alice,uint32(80000),uint32(40000)));
        bsm.joinPipeline(id,0,100);
    }
    function test_D9_classCountAndModelMustMatch() public {
        register(alice,1,80000);
        uint64 id=create();
        vm.prank(alice); vm.expectRevert(); bsm.joinPipeline(id,0,100);
        bsm.onRegister(bob,abi.encode("model",uint32(0),uint32(100),uint32(100),uint32(4),uint32(80000),"nvidia-a100","https://operator.example.com"));
        vm.prank(bob); vm.expectRevert(); bsm.joinPipeline(id,0,100);
    }
    function test_D9_unregisterStopsCoverageAndRejoinPreservesOrder() public {
        register(alice,4,80000); register(bob,4,80000);
        uint64 id=create();
        vm.prank(alice); bsm.joinPipeline(id,0,50);
        vm.prank(bob); bsm.joinPipeline(id,50,100);
        assertTrue(bsm.isFullyCovered(id));
        bsm.onUnregister(alice);
        assertFalse(bsm.isFullyCovered(id));
        vm.prank(alice); bsm.leavePipeline(id);
        register(alice,4,80000);
        vm.prank(alice); bsm.joinPipeline(id,0,50);
        assertTrue(bsm.isFullyCovered(id));
    }
    function test_D9_unregisteredOperatorIsEvictedAndActiveOneIsNot() public {
        register(alice,4,80000); register(bob,4,80000);
        uint64 id=create();
        vm.prank(alice); bsm.joinPipeline(id,0,50);
        vm.prank(bob); bsm.joinPipeline(id,50,100);
        vm.prank(attacker); vm.expectRevert(); bsm.evictInactive(id,bob);
        bsm.onUnregister(alice);
        vm.prank(attacker); bsm.evictInactive(id,alice);
        assertEq(bsm.getPipelineOperators(id).length,1);
        assertEq(bsm.allocatedGpus(alice),0);
        register(address(0x3333),4,80000);
        vm.prank(address(0x3333)); bsm.joinPipeline(id,0,50);
        assertTrue(bsm.isFullyCovered(id));
    }
    function test_D9_gpusAreNotCountedTwiceAcrossPipelines() public {
        register(alice,4,80000);
        uint64 first=create(); uint64 second=create();
        vm.prank(alice); bsm.joinPipeline(first,0,100);
        vm.prank(alice); vm.expectRevert(abi.encodeWithSelector(DistributedInferenceBSM.GpuRequirementNotMet.selector,alice));
        bsm.joinPipeline(second,0,100);
        vm.prank(alice); bsm.leavePipeline(first);
        vm.prank(alice); bsm.joinPipeline(second,0,100);
        assertEq(bsm.allocatedGpus(alice),4);
    }
    function test_D9_cannotRebindOrRegisterForeignCapability() public {
        vm.prank(attacker); vm.expectRevert(); bsm.onBlueprintCreated(2,attacker,attacker);
        vm.expectRevert(); bsm.onBlueprintCreated(2,attacker,address(this));
        vm.prank(attacker); vm.expectRevert();
        bsm.onRegister(alice,abi.encode("model",uint32(0),uint32(100),uint32(100),uint32(4),uint32(80000),"nvidia-h100","https://attacker.example.com"));
        vm.expectRevert(); new DistributedInferenceBSM(address(0));
    }
}
