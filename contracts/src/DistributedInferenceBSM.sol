// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import { BlueprintServiceManagerBase } from "tnt-core/BlueprintServiceManagerBase.sol";
contract DistributedInferenceBSM is BlueprintServiceManagerBase {

    struct PipelineGroup {
        string modelId;
        address[] operators;     // ordered by layer range (head -> tail)
        uint32 totalLayers;
        bool active;
        uint256 createdAt;
    }

    struct OperatorCapabilities {
        uint32 layerStart;
        uint32 layerEnd;
        uint32 gpuCount;
        uint32 vramPerGpuMib;
        string gpuClass;
        bytes32 modelHash;
        uint32 totalLayers;
        string endpoint;
        bool active;
    }

    struct GpuRequirement {
        string gpuClass;
        uint32 count;
        uint32 vramPerGpuMib;
    }

    struct LayerRange {
        uint32 start;    // inclusive
        uint32 end;      // exclusive
        address operator;
    }
    uint64 public nextPipelineId;
    mapping(uint64 => PipelineGroup) public pipelines;
    mapping(uint64 => GpuRequirement) public gpuRequirements;
    mapping(uint64 => mapping(address => LayerRange)) public layerRanges;
    mapping(uint64 => LayerRange[]) internal pipelineRanges;
    mapping(address => OperatorCapabilities) public operatorCaps;
    mapping(address => uint256) private memberships;

    event PipelineCreated(uint64 indexed pipelineId, string modelId, uint32 totalLayers);
    event OperatorJoined(uint64 indexed pipelineId, address indexed operator, uint32 layerStart, uint32 layerEnd);
    event OperatorLeft(uint64 indexed pipelineId, address indexed operator);
    event PipelineActivated(uint64 indexed pipelineId);

    error PipelineNotFound(uint64 pipelineId);
    error PipelineNotActive(uint64 pipelineId);
    error OperatorNotRegistered(address operator);
    error OperatorAlreadyInPipeline(uint64 pipelineId, address operator);
    error LayerRangeOverlap(uint64 pipelineId, uint32 start, uint32 end);
    error LayerRangeInvalid(uint32 start, uint32 end, uint32 totalLayers);
    error InsufficientVram(address operator, uint32 required, uint32 available);
    error GpuRequirementNotMet(address operator);
    error OperatorNotInPipeline(uint64 pipelineId, address operator);
    constructor(address _tangleCore) {
        require(_tangleCore != address(0), "tangle required");
        tangleCore = _tangleCore;
    }
    function onBlueprintCreated(
        uint64 _blueprintId,
        address _owner,
        address _tangleCore
    )
        external
        override
    {
        require(msg.sender == tangleCore && _tangleCore == tangleCore, "not tangle");
        if (blueprintOwner != address(0)) revert AlreadyInitialized();
        require(_owner != address(0), "owner required");
        blueprintId = _blueprintId;
        blueprintOwner = _owner;
    }
    function onRegister(
        address operator,
        bytes calldata registrationInputs
    )
        external
        payable
        override
        onlyFromTangle
    {
        (string memory modelId, uint32 layerStart, uint32 layerEnd, uint32 totalLayers,
            uint32 gpuCount, uint32 vramPerGpuMib, string memory gpuClass, string memory endpoint)
            = abi.decode(registrationInputs, (string,uint32,uint32,uint32,uint32,uint32,string,string));
        require(bytes(modelId).length > 0 && gpuCount > 0 && vramPerGpuMib > 0 && bytes(gpuClass).length > 0, "invalid GPU capability");
        if (layerStart >= layerEnd || layerEnd > totalLayers) revert LayerRangeInvalid(layerStart, layerEnd, totalLayers);
        require(!operatorCaps[operator].active && memberships[operator] == 0, "leave existing pipelines before registering");
        operatorCaps[operator] = OperatorCapabilities(layerStart, layerEnd, gpuCount,
            vramPerGpuMib, gpuClass, keccak256(bytes(modelId)), totalLayers, endpoint, true);
    }

    function onUnregister(address operator) external override onlyFromTangle {
        operatorCaps[operator].active = false;
    }

    function createPipeline(
        string calldata modelId,
        uint32 totalLayers,
        GpuRequirement calldata gpu
    ) external onlyBlueprintOwner returns (uint64 pipelineId) {
        require(totalLayers > 0 && bytes(modelId).length > 0, "invalid model");
        require(gpu.count > 0 && gpu.vramPerGpuMib > 0 && bytes(gpu.gpuClass).length > 0, "GPU requirement required");
        pipelineId = nextPipelineId++;
        gpuRequirements[pipelineId] = gpu;

        PipelineGroup storage p = pipelines[pipelineId];
        p.modelId = modelId;
        p.totalLayers = totalLayers;
        p.active = false;
        p.createdAt = block.timestamp;


        emit PipelineCreated(pipelineId, modelId, totalLayers);
    }
    function joinPipeline(
        uint64 pipelineId,
        uint32 layerStart,
        uint32 layerEnd
    ) external {
        PipelineGroup storage p = pipelines[pipelineId];
        if (bytes(p.modelId).length == 0) revert PipelineNotFound(pipelineId);

        OperatorCapabilities storage caps = operatorCaps[msg.sender];
        if (!caps.active) revert OperatorNotRegistered(msg.sender);

        if (layerStart >= layerEnd || layerEnd > p.totalLayers) {
            revert LayerRangeInvalid(layerStart, layerEnd, p.totalLayers);
        }
        if (layerRanges[pipelineId][msg.sender].end > 0) {
            revert OperatorAlreadyInPipeline(pipelineId, msg.sender);
        }
        LayerRange[] storage ranges = pipelineRanges[pipelineId];
        for (uint256 i = 0; i < ranges.length; i++) {
            if (layerStart < ranges[i].end && layerEnd > ranges[i].start) {
                revert LayerRangeOverlap(pipelineId, layerStart, layerEnd);
            }
        }
        GpuRequirement storage gpu = gpuRequirements[pipelineId];
        uint256 requiredCount = (uint256(gpu.count) * (layerEnd - layerStart) + p.totalLayers - 1) / p.totalLayers;
        if (caps.modelHash != keccak256(bytes(p.modelId)) || caps.totalLayers != p.totalLayers
            || layerStart < caps.layerStart || layerEnd > caps.layerEnd
            || keccak256(bytes(caps.gpuClass)) != keccak256(bytes(gpu.gpuClass))
            || caps.gpuCount < requiredCount) revert GpuRequirementNotMet(msg.sender);
        if (caps.vramPerGpuMib < gpu.vramPerGpuMib)
            revert InsufficientVram(msg.sender, gpu.vramPerGpuMib, caps.vramPerGpuMib);

        LayerRange memory range = LayerRange({
            start: layerStart,
            end: layerEnd,
            operator: msg.sender
        });
        ranges.push(range);
        ++memberships[msg.sender];
        layerRanges[pipelineId][msg.sender] = range;
        _insertOperatorSorted(pipelineId, msg.sender, layerStart);

        emit OperatorJoined(pipelineId, msg.sender, layerStart, layerEnd);
        if (_allLayersCovered(pipelineId)) {
            p.active = true;
            emit PipelineActivated(pipelineId);
        }
    }
    function leavePipeline(uint64 pipelineId) external {
        PipelineGroup storage p = pipelines[pipelineId];
        if (bytes(p.modelId).length == 0) revert PipelineNotFound(pipelineId);

        LayerRange storage range = layerRanges[pipelineId][msg.sender];
        if (range.end == 0) revert OperatorNotInPipeline(pipelineId, msg.sender);
        LayerRange[] storage ranges = pipelineRanges[pipelineId];
        for (uint256 i = 0; i < ranges.length; i++) {
            if (ranges[i].operator == msg.sender) {
                ranges[i] = ranges[ranges.length - 1];
                ranges.pop();
                break;
            }
        }
        delete layerRanges[pipelineId][msg.sender];
        --memberships[msg.sender];
        _removeOperator(pipelineId, msg.sender);
        p.active = false;

        emit OperatorLeft(pipelineId, msg.sender);
    }
    function getPipelineOperators(uint64 pipelineId) external view returns (address[] memory) {
        return pipelines[pipelineId].operators;
    }
    function isFullyCovered(uint64 pipelineId) external view returns (bool) {
        return _allLayersCovered(pipelineId);
    }
    function pipelineCount() external view returns (uint64) {
        return nextPipelineId;
    }
    function _insertOperatorSorted(
        uint64 pipelineId,
        address operator,
        uint32 layerStart
    ) internal {
        PipelineGroup storage p = pipelines[pipelineId];
        uint256 insertIdx = p.operators.length;

        for (uint256 i = 0; i < p.operators.length; i++) {
            if (layerRanges[pipelineId][p.operators[i]].start > layerStart) {
                insertIdx = i;
                break;
            }
        }

        p.operators.push(operator); // extend array
        for (uint256 i = p.operators.length - 1; i > insertIdx; i--) {
            p.operators[i] = p.operators[i - 1];
        }
        p.operators[insertIdx] = operator;
    }
    function _removeOperator(uint64 pipelineId, address operator) internal {
        PipelineGroup storage p = pipelines[pipelineId];
        for (uint256 i = 0; i < p.operators.length; i++) {
            if (p.operators[i] == operator) {
                for (uint256 j = i; j + 1 < p.operators.length; ++j) p.operators[j] = p.operators[j + 1];
                p.operators.pop();
                break;
            }
        }
    }
    function _allLayersCovered(uint64 pipelineId) internal view returns (bool) {
        PipelineGroup storage p = pipelines[pipelineId];
        if (p.operators.length == 0) return false;
        uint32 covered = 0;
        for (uint256 i = 0; i < p.operators.length; i++) {
            LayerRange storage range = layerRanges[pipelineId][p.operators[i]];
            if (!operatorCaps[p.operators[i]].active || range.start != covered) return false;
            covered = range.end;
        }
        return covered == p.totalLayers;
    }
}
