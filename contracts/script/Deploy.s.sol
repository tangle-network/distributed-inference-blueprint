// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;
import {Script, console2} from "forge-std/Script.sol";
import {DistributedInferenceBSM} from "../src/DistributedInferenceBSM.sol";

/// Model-specific manager deployment; cargo-tangle registers the definition.
contract Deploy is Script {
    function run() external {
        address registry = vm.envAddress("TANGLE_CORE");
        require(registry.code.length != 0, "TANGLE_CORE has no code");
        vm.startBroadcast();
        DistributedInferenceBSM manager = new DistributedInferenceBSM(registry);
        vm.stopBroadcast();
        console2.log("DEPLOY_DISTRIBUTED_BSM=%s", vm.toString(address(manager)));
    }
}
