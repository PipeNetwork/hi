#!/bin/bash
# Run hi on Terminal-Bench 2.0 via Harbor.
#
# Usage:
#   scripts/terminal_bench.sh sample          # 10-task health-check sample
#   scripts/terminal_bench.sh full            # the full dataset (mind cost!)
#   scripts/terminal_bench.sh task <glob>     # tasks matching a name glob
#
# Requires: docker running, `uv tool install harbor`, a Linux build of hi at
# $HI_AGENT_BINARY (see docs/terminal-bench.md), and PIPENETWORK_API_KEY (or
# override HI_TB_MODEL, e.g. anthropic/claude-sonnet-5 + ANTHROPIC_API_KEY).
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-sample}"
MODEL="${HI_TB_MODEL:-pipenetwork/ipop/coder-balanced}"
DATASET="${HI_TB_DATASET:-terminal-bench@2.0}"
JOBS_DIR="${HI_TB_JOBS_DIR:-jobs/terminal-bench}"
CONCURRENT="${HI_TB_CONCURRENT:-4}"

: "${HI_AGENT_BINARY:?set HI_AGENT_BINARY to a Linux build of hi (docs/terminal-bench.md)}"
[ -f "$HI_AGENT_BINARY" ] || { echo "HI_AGENT_BINARY not found: $HI_AGENT_BINARY"; exit 2; }
docker ps >/dev/null || { echo "docker daemon not running"; exit 2; }

ARGS=(
  run
  --dataset "$DATASET"
  --agent integrations.terminal_bench.hi_agent:HiAgent
  --model "$MODEL"
  --n-concurrent "$CONCURRENT"
  --jobs-dir "$JOBS_DIR"
  --agent-include-logs 'hi-report.json'
  --agent-include-logs 'hi.txt'
)

case "$MODE" in
  sample)
    # A fixed, arbitrary-but-stable spread of tasks for plumbing checks.
    for t in "chem*" "git*" "hello*" "csv*" "build*" "fix*" "path*" "log*" "compress*" "server*"; do
      ARGS+=(--include-task-name "$t")
    done
    ;;
  full) ;;
  task)
    ARGS+=(--include-task-name "${2:?usage: terminal_bench.sh task <glob>}")
    ;;
  *)
    echo "unknown mode: $MODE (sample|full|task <glob>)"; exit 2 ;;
esac

exec harbor "${ARGS[@]}"
