#!/usr/bin/env bash
# Replay the harness judge over committed tapes (no provider). Optionally
# compare a live artifacts/harness/summary.json to eval-baseline/harness.json.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

echo "hi-eval judge --suite bench/harness"
cargo run -q -p hi-eval -- judge --suite bench/harness

SUMMARY=${1:-}
BASELINE=${2:-eval-baseline/harness.json}
if [[ -z "${SUMMARY}" ]]; then
  echo "replay only (pass a summary.json to compare live rates)"
  exit 0
fi
if [[ -d "${SUMMARY}" ]]; then
  SUMMARY="${SUMMARY}/summary.json"
fi

python3 - "$SUMMARY" "$BASELINE" <<'PY'
import json
import pathlib
import sys

current = json.loads(pathlib.Path(sys.argv[1]).read_text())
baseline = json.loads(pathlib.Path(sys.argv[2]).read_text())
errors = []

def rate(doc, key):
    value = doc.get(key)
    return None if value is None else float(value)

for key in ("process_pass_rate", "budget_pass_rate"):
    now = rate(current, key)
    was = rate(baseline, key)
    if now is None or was is None:
        continue
    if was - now > 0.05:
        errors.append(f"{key} regressed {was:.2%} -> {now:.2%}")

for key in ("write_overwrite_violations", "image_elision_misses"):
    now = int(current.get(key) or 0)
    if now > 0:
        errors.append(f"{key}={now} (must be 0)")

if errors:
    raise SystemExit("harness regression:\n- " + "\n- ".join(errors))
print("harness live summary within baseline")
PY
