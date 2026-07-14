#!/usr/bin/env bash
# Enforce the release evaluation guardrails against a checked-in baseline.
set -euo pipefail

CURRENT=${1:-artifacts/eval}
BASELINE=${2:-eval-baseline/core-0.2.json}

if [ -d "$CURRENT" ]; then
    CURRENT="$CURRENT/summary.json"
fi

python3 - "$CURRENT" "$BASELINE" <<'PY'
import json
import pathlib
import sys

current_path = pathlib.Path(sys.argv[1])
baseline_path = pathlib.Path(sys.argv[2])
if not current_path.is_file():
    raise SystemExit(f"missing evaluation summary: {current_path}")
if not baseline_path.is_file():
    raise SystemExit(f"missing evaluation baseline: {baseline_path}")

current = json.loads(current_path.read_text())
baseline = json.loads(baseline_path.read_text())
errors = []

if current.get("schema_version") != 2:
    raise SystemExit("current evaluation summary must use schema_version 2")
if baseline.get("schema_version") != 2:
    raise SystemExit("evaluation baseline must use schema_version 2")
required = {
    "candidate_pass_rate",
    "solve_at_n",
    "pass_at_k",
    "false_verified_count",
    "infrastructure_error_rate",
    "solve_rate",
    "cost_per_solved",
}
missing = sorted(required - current.keys())
if missing:
    raise SystemExit("evaluation summary is missing required field(s): " + ", ".join(missing))

for field in ("candidate_pass_rate", "solve_at_n", "infrastructure_error_rate", "solve_rate"):
    value = float(current[field])
    if not 0.0 <= value <= 1.0:
        raise SystemExit(f"evaluation summary field {field} must be between 0 and 1")
if current["pass_at_k"] is not None and not 0.0 <= float(current["pass_at_k"]) <= 1.0:
    raise SystemExit("evaluation summary field pass_at_k must be null or between 0 and 1")
if current["cost_per_solved"] is not None and float(current["cost_per_solved"]) < 0:
    raise SystemExit("evaluation summary field cost_per_solved must be null or non-negative")

false_verified = int(current["false_verified_count"])
if false_verified < 0:
    raise SystemExit("evaluation summary field false_verified_count must be non-negative")
if false_verified:
    errors.append(f"{false_verified} candidate(s) were reported verified but failed the final oracle")

infra = float(current["infrastructure_error_rate"])
if infra > 0.02:
    errors.append(f"infrastructure error rate {infra:.2%} exceeds 2%")

solve = current.get("solve_rate")
baseline_solve = baseline.get("solve_rate")
if solve is not None and baseline_solve is not None:
    regression = float(baseline_solve) - float(solve)
    if regression > 0.05:
        errors.append(
            f"solve rate regressed {regression:.2%} "
            f"({float(baseline_solve):.2%} -> {float(solve):.2%})"
        )

cost = current.get("cost_per_solved")
baseline_cost = baseline.get("cost_per_solved")
if cost is not None and baseline_cost not in (None, 0):
    increase = float(cost) / float(baseline_cost) - 1.0
    if increase > 0.20:
        errors.append(
            f"cost per solved task increased {increase:.2%} "
            f"({float(baseline_cost):.4f} -> {float(cost):.4f})"
        )

print(
    "evaluation guardrails: "
    f"false_verified={false_verified} infra={infra:.2%} "
    f"solve={solve!r} cost_per_solved={cost!r}"
)
if baseline_solve is None:
    print("note: solve-rate baseline is not populated; capture the first provider-backed 0.2 run")
if baseline_cost is None:
    print("note: cost baseline is not populated; cost regression is not yet enforceable")
if errors:
    raise SystemExit("evaluation regression:\n- " + "\n- ".join(errors))
PY
