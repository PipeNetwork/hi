#!/usr/bin/env bash
# Run Terminal-Bench 2.1 with the hi agent adapter.
#
#   ./run.sh -m anthropic/claude-opus-4-8 -l 5     # 5 tasks
#   ./run.sh -m anthropic/claude-opus-4-8 -i <task-name>
#   ./run.sh -m anthropic/claude-opus-4-8 -l 89    # full suite ($$$)
#
# Everything after ./run.sh is passed to `harbor run` verbatim. NOTE the flag
# semantics: `-l/--n-tasks` limits how many TASKS run; `-k/--n-attempts` is
# attempts PER TRIAL. Without an explicit task selector this script caps the
# run at 5 tasks so a missing flag can never silently mean the full suite.
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -f dist/hi-linux ]; then
  echo "dist/hi-linux missing — run ./build-linux.sh first" >&2
  exit 1
fi

if [ ! -d .venv ]; then
  uv venv .venv
fi
uv pip install -q -e . harbor --python .venv/bin/python

limit=()
case " $* " in
  *" -l "* | *" --n-tasks "* | *" -t "* | *" --task "* | *" -i "* | *" --include-task-name "*) ;;
  *)
    echo "no task selector given — defaulting to -l 5" >&2
    limit=(-l 5)
    ;;
esac

# `${limit[@]+…}`: macOS bash 3.2 treats an empty array as unbound under -u.
# --artifact: containers are deleted after each trial; without this the full
# hi transcript dies with them and only exception tails survive for analysis.
exec .venv/bin/harbor run \
  -d terminal-bench/terminal-bench-2-1 \
  --agent-import-path hi_terminal_bench:HiAgent \
  --artifact /installed-agent/hi-output.txt \
  ${limit[@]+"${limit[@]}"} \
  "$@"
