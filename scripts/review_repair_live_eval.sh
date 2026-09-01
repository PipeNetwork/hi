#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run a small live A/B for read-only review-repair behavior.

Required:
  BASELINE_DIR=/path/to/baseline-worktree
  CANDIDATE_DIR=/path/to/candidate-worktree   (defaults to current directory)
  MODELS="small/model,strong/model"
  HI_BASE_URL=...
  HI_API_KEY=...

Optional:
  TRIALS=3
  OUT_DIR=/tmp/hi-review-repair-live/<timestamp>
  HI_PROVIDER=openai

Example:
  BASELINE_DIR=../hi-before-review-repair \
  CANDIDATE_DIR=. \
  MODELS="qwen/qwen3-coder-small,anthropic/claude-sonnet-4" \
  HI_BASE_URL="$HI_BASE_URL" HI_API_KEY="$HI_API_KEY" \
  bash scripts/review_repair_live_eval.sh

The script builds both worktrees, runs six read-only review prompts in isolated
tiny crates, stores stdout/stderr plus --report JSON, then prints a telemetry
summary grouped by branch/model/scenario. Compare repair exhaustion separately
from step-cap hits; step-cap hits are regressions.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

: "${BASELINE_DIR:?set BASELINE_DIR to a worktree checked out before review-repair hardening}"
CANDIDATE_DIR="${CANDIDATE_DIR:-$(pwd)}"
: "${MODELS:?set MODELS to a comma-separated small/strong model pair}"
: "${HI_BASE_URL:?set HI_BASE_URL for the OpenAI-compatible endpoint}"
: "${HI_API_KEY:?set HI_API_KEY for the endpoint}"

TRIALS="${TRIALS:-3}"
HI_PROVIDER="${HI_PROVIDER:-openai}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-${TMPDIR:-/tmp}/hi-review-repair-live/$STAMP}"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
SCENARIOS=(
  no_evidence
  listing_only
  search_without_read
  inspected_disclaimer
  generic_inventory
  valid_bounded
)

sanitize() {
  printf '%s' "$1" | tr -cs 'A-Za-z0-9_.-' '-'
}

prompt_for() {
  case "$1" in
    no_evidence)
      printf '%s\n' "Review this tiny crate status. Do not modify files. Return concise findings only."
      ;;
    listing_only)
      printf '%s\n' "Review the repository layout and current status. Do not modify files. Return concise findings only."
      ;;
    search_without_read)
      printf '%s\n' "Review for unsafe unwraps. Search for target_marker or unwrap before answering. Do not modify files. Return concise findings only."
      ;;
    inspected_disclaimer)
      printf '%s\n' "Review src/lib.rs for security issues. Do not modify files. Return findings and limits from inspected evidence."
      ;;
    generic_inventory)
      printf '%s\n' "Review the codebase status. Do not modify files. Avoid a generic inventory; return concrete findings and limits."
      ;;
    valid_bounded)
      printf '%s\n' "Read src/lib.rs and give one concise bounded status finding with limits. Do not modify files."
      ;;
    *)
      echo "unknown scenario: $1" >&2
      exit 2
      ;;
  esac
}

make_fixture() {
  local dir="$1"
  mkdir -p "$dir/src" "$dir/home"
  cat >"$dir/Cargo.toml" <<'EOF'
[package]
name = "review-repair-live-fixture"
version = "0.1.0"
edition = "2024"

[lib]
path = "src/lib.rs"
EOF
  cat >"$dir/src/lib.rs" <<'EOF'
pub fn target_marker(input: Option<&str>) -> usize {
    let value = input.unwrap_or("fallback");
    value.len()
}

pub fn stable_status() -> &'static str {
    "ready"
}
EOF
}

build_hi() {
  local label="$1"
  local worktree="$2"
  local target="$OUT_DIR/build/$label"
  echo "building hi in $worktree" >&2
  cargo build -q -p hi --manifest-path "$worktree/Cargo.toml" --target-dir "$target"
  printf '%s\n' "$target/debug/hi"
}

run_one() {
  local branch="$1"
  local hi_bin="$2"
  local model="$3"
  local trial="$4"
  local scenario="$5"
  local model_dir
  model_dir="$(sanitize "$model")"
  local run_dir="$OUT_DIR/$branch/$model_dir/trial-$trial/$scenario"
  local fixture="$run_dir/work"
  local report="$run_dir/report.json"
  local stdout="$run_dir/stdout.txt"
  local stderr="$run_dir/stderr.txt"
  local status_file="$run_dir/status.txt"
  mkdir -p "$run_dir"
  make_fixture "$fixture"

  local prompt
  prompt="$(prompt_for "$scenario")"
  printf '%s\n' "$prompt" >"$run_dir/prompt.txt"

  set +e
  (
    cd "$fixture"
    HOME="$fixture/home" \
      "$hi_bin" \
      --provider "$HI_PROVIDER" \
      --model "$model" \
      --base-url "$HI_BASE_URL" \
      --api-key "$HI_API_KEY" \
      --no-save \
      --no-memory \
      --no-auto-compact \
      --no-finalize \
      --report "$report" \
      "$prompt"
  ) >"$stdout" 2>"$stderr"
  local status=$?
  set -e
  printf '%s\n' "$status" >"$status_file"
  printf '%-10s %-32s trial=%s scenario=%-22s status=%s\n' \
    "$branch" "$model" "$trial" "$scenario" "$status"
}

BASELINE_HI="$(build_hi baseline "$BASELINE_DIR")"
CANDIDATE_HI="$(build_hi candidate "$CANDIDATE_DIR")"

for trial in $(seq 1 "$TRIALS"); do
  for model in "${MODEL_LIST[@]}"; do
    model="$(printf '%s' "$model" | xargs)"
    [[ -n "$model" ]] || continue
    for scenario in "${SCENARIOS[@]}"; do
      run_one baseline "$BASELINE_HI" "$model" "$trial" "$scenario"
      run_one candidate "$CANDIDATE_HI" "$model" "$trial" "$scenario"
    done
  done
done

python3 - "$OUT_DIR" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
groups = defaultdict(list)

for report in root.rglob("report.json"):
    rel = report.relative_to(root).parts
    if len(rel) < 5:
        continue
    branch, model, trial, scenario = rel[:4]
    run_dir = report.parent
    status = int((run_dir / "status.txt").read_text().strip() or "1")
    stdout = (run_dir / "stdout.txt").read_text(errors="ignore")
    stderr = (run_dir / "stderr.txt").read_text(errors="ignore")
    visible = (stdout + "\n" + stderr).lower()
    leak = "insufficient evidence" in visible or "quality_rejected" in visible
    try:
        data = json.loads(report.read_text())
    except Exception:
        data = {}
    telemetry = data.get("telemetry", {})
    step_cap = bool(telemetry.get("stopped_by_step_cap") or telemetry.get("hit_step_cap"))
    repair_exhausted = bool(telemetry.get("review_repair_stopped_by_exhaustion"))
    stalled = bool(telemetry.get("stalled_unfinished") or telemetry.get("stalled_repeating"))
    accepted = status == 0 and not stalled and not step_cap and not leak
    useful_incomplete = status == 0 and repair_exhausted and not step_cap and not leak
    recovered = accepted or useful_incomplete
    groups[(branch, model, scenario)].append({
        "recovered": recovered,
        "accepted": accepted,
        "useful_incomplete": useful_incomplete,
        "nudges": int(telemetry.get("quality_repair_nudges") or 0),
        "step_cap": step_cap,
        "repair_exhausted": repair_exhausted,
        "leak": leak,
        "reason": telemetry.get("review_repair_exhaustion_reason") or "",
    })

print("\nsummary")
print("branch\tmodel\tscenario\tn\trecovery\taccepted\tincomplete\tavg_nudges_recovered\tstep_cap\trepair_exhausted\tleaks\treasons")
for key in sorted(groups):
    rows = groups[key]
    n = len(rows)
    recovered = sum(r["recovered"] for r in rows)
    accepted = sum(r["accepted"] for r in rows)
    incomplete = sum(r["useful_incomplete"] for r in rows)
    recovered_nudges = [r["nudges"] for r in rows if r["recovered"]]
    avg_nudges = sum(recovered_nudges) / len(recovered_nudges) if recovered_nudges else 0.0
    step_cap = sum(r["step_cap"] for r in rows)
    repair_exhausted = sum(r["repair_exhausted"] for r in rows)
    leaks = sum(r["leak"] for r in rows)
    reasons = ",".join(sorted({r["reason"] for r in rows if r["reason"]})) or "-"
    print(
        f"{key[0]}\t{key[1]}\t{key[2]}\t{n}\t"
        f"{recovered}/{n}\t{accepted}/{n}\t{incomplete}/{n}\t"
        f"{avg_nudges:.2f}\t{step_cap}\t{repair_exhausted}\t{leaks}\t{reasons}"
    )

print(f"\nartifacts: {root}")
PY
