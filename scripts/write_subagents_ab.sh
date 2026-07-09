#!/usr/bin/env bash
# A/B the write-`delegate` subagent on a bench suite: run every task with the
# write subagent OFF, then ON, and print both hi-eval summaries so pass@1 / tokens
# / time can be compared. This is the experiment that decides whether `delegate`
# is worth turning on by default (see the PR / discussion). Artifacts from each
# side are labeled `write_subagents=off|on` for deeper analysis.
#
# Requires a pipenetwork key (PIPENETWORK_API_KEY / HI_API_KEY). Optional:
#   HI_MODEL (default: ipop/coder-balanced)
#   $1       tasks dir (default: bench/tasks; try bench/hard for bigger tasks)
#   $2       hi-eval configs (default: verify)
# Exit: 0 ran both sides · 2 skipped (no key).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

TASKS="${1:-bench/tasks}"
CONFIGS="${2:-verify}"
MODEL="${HI_MODEL:-ipop/coder-balanced}"
KEY="${PIPENETWORK_API_KEY:-${HI_API_KEY:-}}"
[[ -z "$KEY" ]] && { echo "SKIP: set PIPENETWORK_API_KEY to run the A/B" >&2; exit 2; }

# The child `hi` (and its delegate grandchild) must be the release binary with the
# delegate tier; build it once and pin it so hi-eval doesn't pick a stale debug one.
export HI_BIN="${HI_BIN:-$PWD/target/release/hi}"
[[ -x "$HI_BIN" ]] || { echo "building hi (release)…" >&2; cargo build --release -p hi >&2 || exit 1; }
export HI_API_KEY="$KEY" HI_MODEL="$MODEL"

STAMP=$(date +%Y%m%d-%H%M%S)
OFF_DIR="target/hi-eval/ab-$STAMP/off"
ON_DIR="target/hi-eval/ab-$STAMP/on"
run_eval() { cargo run -q -p hi-eval -- "$TASKS" --provider pipenetwork --configs "$CONFIGS" --artifacts "$1"; }

echo "== write_subagents A/B · tasks=$TASKS · configs=$CONFIGS · model=$MODEL ==" >&2

echo "--- side A: write_subagents OFF ---" >&2
run_eval "$OFF_DIR" 2>&1 | tee /tmp/ab-off.log

echo "--- side B: write_subagents ON ---" >&2
HI_WRITE_SUBAGENTS=1 run_eval "$ON_DIR" 2>&1 | tee /tmp/ab-on.log

echo "" >&2
echo "======================= A/B COMPARISON =======================" >&2
echo "[OFF] $OFF_DIR"
grep -aE "pass@1|pass@k|tok/task|tokens|why:|steer" /tmp/ab-off.log | tail -8
echo ""
echo "[ON ] $ON_DIR"
grep -aE "pass@1|pass@k|tok/task|tokens|why:|steer" /tmp/ab-on.log | tail -8
echo ""
echo "Decision: turn on by default only if ON is pass-rate neutral-or-better and the"
echo "token/time cost is acceptable. Per-run artifacts are labeled write_subagents=off|on."
