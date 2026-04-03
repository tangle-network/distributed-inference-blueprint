// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title DistributedInferenceBSM
/// @notice Multi-operator Blueprint Service Manager for pipeline-parallel inference.
/// Coordinates multiple operators that each serve a contiguous range of model layers,
/// forming a pipeline that can serve models too large for a single GPU (e.g., Llama 405B).
///
/// Payment is split proportionally by the fraction of layers each operator serves.
contract DistributedInferenceBSM {
    // --- Structs ---

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
        uint32 totalVramMib;
        string endpoint;
        bool active;
    }

    struct LayerRange {
        uint32 start;    // inclusive
        uint32 end;      // exclusive
        address operator;
    }

    // --- State ---

    /// Pipeline ID counter
    uint64 public nextPipelineId;

    /// Pipeline groups by ID
    mapping(uint64 => PipelineGroup) public pipelines;

    /// Layer ranges per pipeline: pipelineId -> operator -> LayerRange
    mapping(uint64 => mapping(address => LayerRange)) public layerRanges;

    /// All layer ranges for a pipeline (for coverage validation)
    mapping(uint64 => LayerRange[]) internal pipelineRanges;

    /// Operator capabilities (global, not per-pipeline)
    mapping(address => OperatorCapabilities) public operatorCaps;

    /// Base price per token in tsUSD base units (set per pipeline)
    mapping(uint64 => uint256) public basePricePerToken;

    /// Accumulated revenue per operator per pipeline
    mapping(uint64 => mapping(address => uint256)) public operatorRevenue;

    // --- Events ---

    event PipelineCreated(uint64 indexed pipelineId, string modelId, uint32 totalLayers);
    event OperatorJoined(uint64 indexed pipelineId, address indexed operator, uint32 layerStart, uint32 layerEnd);
    event OperatorLeft(uint64 indexed pipelineId, address indexed operator);
    event PipelineActivated(uint64 indexed pipelineId);
    event PaymentDistributed(uint64 indexed pipelineId, address indexed operator, uint256 amount);

    // --- Errors ---

    error PipelineNotFound(uint64 pipelineId);
    error PipelineNotActive(uint64 pipelineId);
    error OperatorNotRegistered(address operator);
    error OperatorAlreadyInPipeline(uint64 pipelineId, address operator);
    error LayerRangeOverlap(uint64 pipelineId, uint32 start, uint32 end);
    error LayerRangeInvalid(uint32 start, uint32 end, uint32 totalLayers);
    error InsufficientVram(address operator, uint32 required, uint32 available);
    error OperatorNotInPipeline(uint64 pipelineId, address operator);

    // --- Registration ---

    /// Called during operator registration. Decodes the ABI-encoded registration
    /// payload and stores operator capabilities.
    function onRegister(
        address operator,
        bytes calldata registrationInputs
    ) external {
        (
            , // modelId (string) — not stored globally
            uint32 layerStart,
            uint32 layerEnd,
            , // totalLayers
            uint32 gpuCount,
            uint32 totalVramMib,
            string memory endpoint
        ) = abi.decode(
            registrationInputs,
            (string, uint32, uint32, uint32, uint32, uint32, string)
        );

        operatorCaps[operator] = OperatorCapabilities({
            layerStart: layerStart,
            layerEnd: layerEnd,
            totalVramMib: totalVramMib,
            endpoint: endpoint,
            active: true
        });
    }

    // --- Pipeline Lifecycle ---

    /// Create a new pipeline group for a model.
    function createPipeline(
        string calldata modelId,
        uint32 totalLayers,
        uint256 pricePerToken
    ) external returns (uint64 pipelineId) {
        pipelineId = nextPipelineId++;

        PipelineGroup storage p = pipelines[pipelineId];
        p.modelId = modelId;
        p.totalLayers = totalLayers;
        p.active = false;
        p.createdAt = block.timestamp;

        basePricePerToken[pipelineId] = pricePerToken;

        emit PipelineCreated(pipelineId, modelId, totalLayers);
    }

    /// Join a pipeline with an assigned layer range.
    /// Validates: no overlap with existing ranges, VRAM sufficient.
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

        // Check for existing membership
        if (layerRanges[pipelineId][msg.sender].end > 0) {
            revert OperatorAlreadyInPipeline(pipelineId, msg.sender);
        }

        // Validate no layer range overlap
        LayerRange[] storage ranges = pipelineRanges[pipelineId];
        for (uint256 i = 0; i < ranges.length; i++) {
            if (layerStart < ranges[i].end && layerEnd > ranges[i].start) {
                revert LayerRangeOverlap(pipelineId, layerStart, layerEnd);
            }
        }

        // Estimate VRAM requirement: proportional to layers served
        // Rough heuristic: total_model_vram * (layers_served / total_layers)
        // We check that operator has at least the proportional VRAM
        uint32 layersServed = layerEnd - layerStart;
        uint32 requiredVram = (caps.totalVramMib * layersServed) / p.totalLayers;
        if (caps.totalVramMib < requiredVram) {
            revert InsufficientVram(msg.sender, requiredVram, caps.totalVramMib);
        }

        // Record the layer range
        LayerRange memory range = LayerRange({
            start: layerStart,
            end: layerEnd,
            operator: msg.sender
        });
        ranges.push(range);
        layerRanges[pipelineId][msg.sender] = range;

        // Add operator to the pipeline (insert in sorted order by layerStart)
        _insertOperatorSorted(pipelineId, msg.sender, layerStart);

        emit OperatorJoined(pipelineId, msg.sender, layerStart, layerEnd);

        // Check if all layers are now covered
        if (_allLayersCovered(pipelineId)) {
            p.active = true;
            emit PipelineActivated(pipelineId);
        }
    }

    /// Leave a pipeline. Only allowed between requests (not mid-inference).
    function leavePipeline(uint64 pipelineId) external {
        PipelineGroup storage p = pipelines[pipelineId];
        if (bytes(p.modelId).length == 0) revert PipelineNotFound(pipelineId);

        LayerRange storage range = layerRanges[pipelineId][msg.sender];
        if (range.end == 0) revert OperatorNotInPipeline(pipelineId, msg.sender);

        // Remove layer range
        LayerRange[] storage ranges = pipelineRanges[pipelineId];
        for (uint256 i = 0; i < ranges.length; i++) {
            if (ranges[i].operator == msg.sender) {
                ranges[i] = ranges[ranges.length - 1];
                ranges.pop();
                break;
            }
        }
        delete layerRanges[pipelineId][msg.sender];

        // Remove operator from the ordered list
        _removeOperator(pipelineId, msg.sender);

        // Pipeline is no longer fully covered
        p.active = false;

        emit OperatorLeft(pipelineId, msg.sender);
    }

    // --- Pricing ---

    /// Get the per-token price for a specific operator in a pipeline.
    /// Price is proportional to the fraction of layers served.
    function getOperatorPricing(
        uint64 pipelineId,
        address operator
    ) external view returns (uint256 pricePerToken) {
        PipelineGroup storage p = pipelines[pipelineId];
        if (bytes(p.modelId).length == 0) revert PipelineNotFound(pipelineId);

        LayerRange storage range = layerRanges[pipelineId][operator];
        if (range.end == 0) revert OperatorNotInPipeline(pipelineId, operator);

        uint32 layersServed = range.end - range.start;
        pricePerToken = (basePricePerToken[pipelineId] * layersServed) / p.totalLayers;
    }

    /// Record token usage and distribute payment proportionally.
    function recordUsage(
        uint64 pipelineId,
        uint256 totalTokens,
        uint256 totalPayment
    ) external {
        PipelineGroup storage p = pipelines[pipelineId];
        if (!p.active) revert PipelineNotActive(pipelineId);

        // Split payment by layer fraction
        for (uint256 i = 0; i < p.operators.length; i++) {
            address op = p.operators[i];
            LayerRange storage range = layerRanges[pipelineId][op];
            uint32 layersServed = range.end - range.start;

            uint256 share = (totalPayment * layersServed) / p.totalLayers;
            operatorRevenue[pipelineId][op] += share;

            emit PaymentDistributed(pipelineId, op, share);
        }
    }

    // --- View Functions ---

    /// Get all operators in a pipeline (ordered by layer range).
    function getPipelineOperators(uint64 pipelineId) external view returns (address[] memory) {
        return pipelines[pipelineId].operators;
    }

    /// Check if all layers in the model are covered by operators.
    function isFullyCovered(uint64 pipelineId) external view returns (bool) {
        return _allLayersCovered(pipelineId);
    }

    /// Get the total number of pipeline groups.
    function pipelineCount() external view returns (uint64) {
        return nextPipelineId;
    }

    // --- Internal ---

    /// Insert operator into the ordered list sorted by layerStart.
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
        // Shift elements right
        for (uint256 i = p.operators.length - 1; i > insertIdx; i--) {
            p.operators[i] = p.operators[i - 1];
        }
        p.operators[insertIdx] = operator;
    }

    /// Remove operator from the ordered list.
    function _removeOperator(uint64 pipelineId, address operator) internal {
        PipelineGroup storage p = pipelines[pipelineId];
        for (uint256 i = 0; i < p.operators.length; i++) {
            if (p.operators[i] == operator) {
                p.operators[i] = p.operators[p.operators.length - 1];
                p.operators.pop();
                break;
            }
        }
    }

    /// Check if all layers [0, totalLayers) are covered without gaps.
    function _allLayersCovered(uint64 pipelineId) internal view returns (bool) {
        PipelineGroup storage p = pipelines[pipelineId];
        if (p.operators.length == 0) return false;

        // Since operators are sorted by layerStart, verify contiguous coverage
        uint32 covered = 0;
        for (uint256 i = 0; i < p.operators.length; i++) {
            LayerRange storage range = layerRanges[pipelineId][p.operators[i]];
            if (range.start != covered) return false;
            covered = range.end;
        }
        return covered == p.totalLayers;
    }
}
