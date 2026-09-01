#!/usr/bin/env bash
# The bench/long A/B: genuinely multi-turn tasks, goal-driven vs plain — with
# EQUAL total step budgets (goal side: N turns × S steps; plain side: 1 turn ×
# N*S steps), so the comparison is about structure, not budget.
#
# Requires PIPENETWORK_API_KEY. Optional: HI_MODEL, HI_EVAL_TURNS (default 6),
# HI_EVAL_TURN_STEPS (default 25), $1 tasks dir.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
TASKS="${1:-bench/long}"
MODEL="${HI_MODEL:-ipop/coder-balanced}"
KEY="${PIPENETWORK_API_KEY:-${HI_API_KEY:-}}"
[[ -z "$KEY" ]] && { echo "SKIP: set PIPENETWORK_API_KEY" >&2; exit 2; }
export HI_BIN="${HI_BIN:-$PWD/target/release/hi}"
[[ -x "$HI_BIN" ]] || { cargo build --release -p hi >&2 || exit 1; }
export HI_API_KEY="$KEY" PIPENETWORK_API_KEY="$KEY" HI_MODEL="$MODEL"
export HI_EVAL_TURNS="${HI_EVAL_TURNS:-6}" HI_EVAL_TURN_STEPS="${HI_EVAL_TURN_STEPS:-25}"
STAMP=$(date +%Y%m%d-%H%M%S)
run_eval() { cargo run -q -p hi-eval -- "$TASKS" --profile=pipenetwork --configs=verify --artifacts="$1"; }
echo "== bench/long A/B · $HI_EVAL_TURNS x $HI_EVAL_TURN_STEPS steps · model=$MODEL ==" >&2
echo "--- side A: plain (1 turn × $((HI_EVAL_TURNS * HI_EVAL_TURN_STEPS)) steps) ---" >&2
run_eval "target/hi-eval/long-ab-$STAMP/off" 2>&1 | tee /tmp/long-ab-off.log
echo "--- side B: goal-driven ($HI_EVAL_TURNS turns × $HI_EVAL_TURN_STEPS steps) ---" >&2
HI_EVAL_GOAL=1 run_eval "target/hi-eval/long-ab-$STAMP/on" 2>&1 | tee /tmp/long-ab-on.log
echo "=== artifacts: target/hi-eval/long-ab-$STAMP/{off,on} ===" >&2
