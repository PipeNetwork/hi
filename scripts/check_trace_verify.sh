#!/usr/bin/env bash
# CI guard for the trace hash chain: `hi trace verify` must accept a valid
# sample trace and reject a tampered one. Catches regressions in
# hi_trace::validate_trace (event-hash recompute, chain linkage, manifest
# consistency) that unit tests alone might miss when the CLI wiring drifts.
set -euo pipefail

STATE_HOME="$(mktemp -d)"
trap 'rm -rf "$STATE_HOME"' EXIT
export XDG_STATE_HOME="$STATE_HOME"

echo "== creating sample trace =="
TRACE_ID="$(cargo run --quiet -p hi-trace --example make_sample_trace -- "$STATE_HOME")"
echo "sample trace: $TRACE_ID"

echo "== verify: a valid trace must pass =="
OUT="$(cargo run --quiet -p hi -- trace verify "$TRACE_ID")"
echo "$OUT"
if ! grep -q "integrity:   ok" <<<"$OUT"; then
    echo "FAIL: valid trace did not verify" >&2
    exit 1
fi

echo "== verify: a tampered journal must fail =="
JOURNAL="$STATE_HOME/hi/rsi/$TRACE_ID/events.jsonl"
# Flip one hex digit in the first event's hash, preserving byte length so the
# event-hash recompute gate (not the byte-count gate) is what must fire.
python3 - "$JOURNAL" <<'PY'
import json, sys
path = sys.argv[1]
lines = open(path).read().splitlines()
event = json.loads(lines[0])
h = event["event_hash"]
event["event_hash"] = ("1" if h[0] == "0" else "0") + h[1:]
lines[0] = json.dumps(event)
open(path, "w").write("\n".join(lines) + "\n")
PY

if cargo run --quiet -p hi -- trace verify "$TRACE_ID" >/dev/null 2>&1; then
    echo "FAIL: tampered trace verified ok" >&2
    exit 1
fi
echo "PASS: tampered trace rejected"
echo "trace verify guard: OK"
