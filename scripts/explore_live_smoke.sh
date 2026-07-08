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

# A broad, cross-file question the model should hand to the explore subagent.
# Phrased WITHOUT "read-only / review / investigate / audit" wording on purpose:
# those trip hi's review-intent classifier, which puts the *parent* turn into
# read-only mode and filters out `explore` (it isn't classified read-only).
PROMPT='Use the explore tool to summarize how the built-in tool set is defined and \
registered in this repo (which files and structures), then give me the summary.'

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

# Real evidence of delegation: the subagent's read-only tools echo with an
# `explore:` prefix (e.g. `explore:grep`) — that only appears if a child agent
# actually ran, not if the model merely wrote "explore" in prose. Also require a
# grounded answer that names the tool machinery.
ran_explore=$(grep -acE 'explore:(read|list|grep|glob)' <<<"$OUT")
found_answer=$(grep -ciE 'ToolSpec|TOOL_SPECS|tool set|registered' <<<"$OUT")
echo "-- subagent tool calls: $ran_explore · answer signal: $found_answer --" >&2

if (( ran_explore > 0 && found_answer > 0 )); then
  echo "PASS: explore subagent actually ran ($ran_explore read-only calls) and answered" >&2
  exit 0
fi
echo "FAIL: no evidence a child explore subagent ran (looked for explore:<tool> calls)" >&2
exit 1
