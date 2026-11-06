#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CORE_DIR="$(cargo metadata --manifest-path "$ROOT/Cargo.toml" --format-version 1 | jq -er '.packages[] | select(.name == "tangle-inference-core") | .manifest_path | sub("/Cargo.toml$"; "")')"
cd "$ROOT/contracts"
forge script script/Deploy.s.sol:DeployInference --remappings "tangle-inference-core/=$CORE_DIR/contracts/" --rpc-url "${RPC_URL:?Set RPC_URL}" --broadcast --slow
