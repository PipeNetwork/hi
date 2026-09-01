#!/usr/bin/env bash
# A/B goal mode on a bench suite: run every task plain, then with HI_EVAL_GOAL=1
# (each task becomes a planner-decomposed long-horizon goal), and print both
# hi-eval summaries. Decides whether the /goal contract helps on bounded tasks.
#
# Requires PIPENETWORK_API_KEY. Optional: HI_MODEL, $1 tasks dir, $2 configs.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
TASKS="${1:-bench/multi}"
CONFIGS="${2:-verify}"
MODEL="${HI_MODEL:-ipop/coder-balanced}"
KEY="${PIPENETWORK_API_KEY:-${HI_API_KEY:-}}"
[[ -z "$KEY" ]] && { echo "SKIP: set PIPENETWORK_API_KEY" >&2; exit 2; }
export HI_BIN="${HI_BIN:-$PWD/target/release/hi}"
[[ -x "$HI_BIN" ]] || { cargo build --release -p hi >&2 || exit 1; }
export HI_API_KEY="$KEY" PIPENETWORK_API_KEY="$KEY" HI_MODEL="$MODEL"
STAMP=$(date +%Y%m%d-%H%M%S)
run_eval() { cargo run -q -p hi-eval -- "$TASKS" --profile=pipenetwork --configs="$CONFIGS" --artifacts="$1"; }
echo "== goal-mode A/B · tasks=$TASKS · model=$MODEL ==" >&2
echo "--- side A: plain prompting ---" >&2
run_eval "target/hi-eval/goal-ab-$STAMP/off" 2>&1 | tee /tmp/goal-ab-off.log
echo "--- side B: goal mode (HI_EVAL_GOAL=1) ---" >&2
HI_EVAL_GOAL=1 run_eval "target/hi-eval/goal-ab-$STAMP/on" 2>&1 | tee /tmp/goal-ab-on.log
echo "" >&2
echo "=== artifacts: target/hi-eval/goal-ab-$STAMP/{off,on} (runs.jsonl labeled goal_mode) ===" >&2
