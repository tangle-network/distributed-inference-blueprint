#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CORE_DIR="$(cargo metadata --locked --manifest-path "$ROOT/Cargo.toml" --format-version 1 | jq -er '.packages[] | select(.name == "tangle-inference-core") | .manifest_path | sub("/Cargo.toml$"; "")')"
BLUEPRINT_DEFINITION="$(node "$CORE_DIR/scripts/register-blueprint.mjs" "$ROOT/deploy/definition.json" "$ROOT")"
export BLUEPRINT_DEFINITION
trap 'rm -f -- "$BLUEPRINT_DEFINITION"' EXIT
cd "$ROOT/contracts"
forge script script/Deploy.s.sol:DeployInference --remappings "tangle-inference-core/=$CORE_DIR/contracts/" --rpc-url "${RPC_URL:?Set RPC_URL}" --broadcast --slow
