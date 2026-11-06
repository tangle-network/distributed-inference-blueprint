// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;
import {RegisterBlueprint} from "tangle-inference-core/script/RegisterBlueprint.s.sol";
import {DistributedInferenceBSM} from "../src/DistributedInferenceBSM.sol";
contract DeployInference is RegisterBlueprint {
    function _deployManager() internal override returns(address) {
        return address(new DistributedInferenceBSM(vm.envAddress("TANGLE_CORE")));
    }
}
