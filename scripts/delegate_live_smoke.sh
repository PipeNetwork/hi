#!/usr/bin/env bash
# Live smoke test for the write-capable `delegate` subagent against pipenetwork.
# Creates a throwaway git repo with a bug + a verify command, compels the model to
# delegate the fix, and checks the verified change was applied back to the tree.
#
# Requires a pipenetwork key: PIPENETWORK_API_KEY (or HI_API_KEY). Optional:
#   HI_MODEL (default: ipop/coder-balanced), HI_BIN (default: ./target/release/hi).
# Exit: 0 pass · 1 fail · 2 skipped (no key).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
HI_BIN="${HI_BIN:-$PWD/target/release/hi}"
MODEL="${HI_MODEL:-ipop/coder-balanced}"
KEY="${PIPENETWORK_API_KEY:-${HI_API_KEY:-}}"
[[ -z "$KEY" ]] && { echo "SKIP: set PIPENETWORK_API_KEY to run the delegate smoke test" >&2; exit 2; }
[[ -x "$HI_BIN" ]] || { echo "building hi (release)…" >&2; cargo build --release -p hi >&2 || exit 1; }

WD="$(mktemp -d)/repo"; mkdir -p "$WD"
printf 'def count_vowels(s):\n    return sum(1 for c in s if c in "aei")\n' > "$WD/solution.py"
printf "import solution as s\nassert s.count_vowels('hello')==2 and s.count_vowels('AEIOU')==5 and s.count_vowels('rhythm')==0\nprint('ok')\n" > "$WD/check.py"
( cd "$WD" && git init -q && git add -A && git -c user.email=t@t -c user.name=t commit -qm init )

echo "== delegate live smoke: model=$MODEL ==" >&2
OUT=$(cd "$WD" && HI_WRITE_SUBAGENTS=1 HI_API_KEY="$KEY" "$HI_BIN" \
  --provider pipenetwork --model "$MODEL" --no-save --temperature 0 --max-steps 40 \
  'Use the delegate tool for this task (delegate it rather than doing the edits inline). Call delegate with task="fix count_vowels in solution.py so it counts a,e,i,o,u case-insensitively" and verify="python3 check.py".' 2>&1)
printf '%s\n' "$OUT"

delegated=$(grep -acE 'delegate subagent|delegate applied' <<<"$OUT")
echo "-- delegate markers: $delegated --" >&2
if (( delegated > 0 )) && ( cd "$WD" && python3 check.py >/dev/null 2>&1 ); then
  echo "PASS: delegate ran and the verified fix was applied back" >&2
  rm -rf "$WD"; exit 0
fi
echo "FAIL: delegate didn't run or the verified fix wasn't applied ($WD)" >&2
exit 1
