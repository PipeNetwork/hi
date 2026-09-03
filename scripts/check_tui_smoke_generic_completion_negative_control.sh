#!/usr/bin/env bash
# Prove the interactive regressions catch removal of the production guards.
# The Cargo feature used here is intentionally unsafe and must stay off in all
# normal, packaged, and release builds.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

FEATURE=smoke-negative-control-disable-generic-completion-guards
ARTIFACT_ROOT=${HI_SMOKE_NEGATIVE_CONTROL_ARTIFACTS:-target/tui-smoke-negative-control}
SCENARIOS=(
  bench/tui-smoke/scenarios/repeated_generic_unfinished_plan/scenario.toml
  bench/tui-smoke/scenarios/mutation_then_generic_recap/scenario.toml
  bench/tui-smoke/scenarios/validation_then_generic_recap/scenario.toml
)

echo "building hi-smoke and guard-disabled hi"
cargo build -p hi-smoke --no-default-features
cargo build -p hi --no-default-features --features "$FEATURE"

for scenario in "${SCENARIOS[@]}"; do
  name=$(basename "$(dirname "$scenario")")
  artifacts="$ARTIFACT_ROOT/guards-disabled/$name"
  echo "negative control (must fail): $name"
  if target/debug/hi-smoke --hi-bin target/debug/hi run "$scenario" \
      --mode scripted \
      --tag generic-completion-guard-negative-control \
      --artifacts "$artifacts"; then
    echo "ERROR: $name passed with generic-completion guards disabled" >&2
    exit 1
  fi
done

echo "rebuilding normal hi with all negative-control features off"
cargo build -p hi --no-default-features

for scenario in "${SCENARIOS[@]}"; do
  name=$(basename "$(dirname "$scenario")")
  artifacts="$ARTIFACT_ROOT/guards-enabled/$name"
  echo "production control (must pass): $name"
  target/debug/hi-smoke --hi-bin target/debug/hi run "$scenario" \
    --mode scripted \
    --tag generic-completion-guard-negative-control \
    --artifacts "$artifacts"
done

echo "generic-completion negative control proved all three regressions"
