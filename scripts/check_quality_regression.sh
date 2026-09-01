#!/usr/bin/env bash
# Compare a live artifacts/quality/summary.json to eval-baseline/quality.json.
# Tape replay stays in check_harness_regression.sh (Core CI, no provider).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

SUMMARY=${1:-}
BASELINE=${2:-eval-baseline/quality.json}
if [[ -z "${SUMMARY}" ]]; then
  echo "usage: $0 artifacts/quality [eval-baseline/quality.json]" >&2
  exit 2
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
    if now is None:
        errors.append(f"{key} missing from live summary")
        continue
    if was is None:
        errors.append(f"{key} missing from baseline")
        continue
    # Same 5pp drop as harness. On a 7-row suite one cell is ~14pp, so a
    # single process fail against the locked 5/7 is a regression.
    if was - now > 0.05:
        errors.append(f"{key} regressed {was:.2%} -> {now:.2%}")

for key in ("write_overwrite_violations", "image_elision_misses"):
    now = int(current.get(key) or 0)
    if now > 0:
        errors.append(f"{key}={now} (must be 0)")

if errors:
    raise SystemExit("quality regression:\n- " + "\n- ".join(errors))
print("quality live summary within baseline")
PY
