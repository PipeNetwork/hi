#!/usr/bin/env bash
# Live smoke test for the read-only `explore` subagent against the pipenetwork
# provider. Asks a codebase question that rewards delegating a bounded read-only
# investigation, then checks the subagent actually ran (default-on, no override)
# and a grounded answer came back.
#
# Requires a pipenetwork key: PIPENETWORK_API_KEY (or HI_API_KEY). Optional:
#   HI_MODEL   (default: ipop/coder-balanced)
#   HI_BIN     (default: ./target/release/hi)
#
# Exit: 0 pass · 1 fail · 2 skipped (no key).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

HI_BIN="${HI_BIN:-$PWD/target/release/hi}"
MODEL="${HI_MODEL:-ipop/coder-balanced}"
KEY="${PIPENETWORK_API_KEY:-${HI_API_KEY:-}}"

if [[ -z "$KEY" ]]; then
  echo "SKIP: set PIPENETWORK_API_KEY (or HI_API_KEY) to run the live explore smoke test" >&2
  exit 2
fi
if [[ ! -x "$HI_BIN" ]]; then
  echo "building hi (release)…" >&2
  cargo build --release -p hi >&2 || exit 1
fi

# The question spans several files, so a capable model should delegate it to the
# read-only explore subagent rather than reading everything into its own context.
PROMPT='Use the explore subagent to find where the read-only tool set is defined \
in this repo (the is_read_only function in crates/hi-tools) and report which tool \
names it includes. Delegate the investigation to a subagent, then summarize its findings.'

echo "== explore live smoke: model=$MODEL ==" >&2
OUT=$(HI_API_KEY="$KEY" "$HI_BIN" \
  --provider pipenetwork --model "$MODEL" \
  --no-save --temperature 0 --max-steps 40 \
  "$PROMPT" 2>&1)
status=$?
printf '%s\n' "$OUT"

if [[ $status -ne 0 ]]; then
  echo "FAIL: hi exited $status" >&2
  exit 1
fi

# Evidence: the explore tool was invoked (its call / prefixed child activity is
# echoed) AND the answer names the function under investigation.
ran_explore=$(grep -ciE 'explore' <<<"$OUT")
found_answer=$(grep -ciE 'is_read_only|read[-_ ]only tool' <<<"$OUT")
echo "-- explore mentions: $ran_explore · answer signal: $found_answer --" >&2

if (( ran_explore > 0 && found_answer > 0 )); then
  echo "PASS: explore subagent ran and returned a grounded answer" >&2
  exit 0
fi
echo "FAIL: no evidence the explore subagent ran and answered" >&2
exit 1
