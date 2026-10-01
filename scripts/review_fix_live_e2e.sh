#!/usr/bin/env bash
# Live end-to-end: drive real hi against pipenetwork.ai on a buggy std-only
# IRC chat server (Welcome never reaches the socket; KICK does not drop the
# target). hi must read the failing tests, fix the server, and pass
# `cargo test --offline` through the verify loop. A second verify-only turn
# ("Run cargo test to verify the welcome-message change…") must complete
# rather than stall as no_progress after a green suite.
#
# Credentials: the saved default profile in the local hi config (its sealed
# key in the private store), or PIPENETWORK_API_KEY / HI_API_KEY if exported.
# HI_PROFILE or HI_PROVIDER selects a different route.
#
# This is deliberately not a one-token canary. Budget is large enough for a
# multi-round review/fix/re-test loop:
#   --max-tokens 16384 · --max-steps 80 · --max-verify-repairs 8
#   ~20-minute wall clock (review/fix plus a verify-only follow-up)
#
# Optional:
#   HI_PROFILE (default: the saved default profile)
#   HI_PROVIDER (default: unset; forces a provider and drops the profile key)
#   HI_MODEL   (default: that profile's model)
#   HI_BIN     (default: freshly built ./target/debug/hi)
#   HI_LIVE_E2E_KEEP=1  keep the workdir on failure
#
# Exit: 0 pass · 1 fail · 2 skipped (no key).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

HI_BIN="${HI_BIN:-$PWD/target/debug/hi}"
FIXTURE="$PWD/bench/tui-smoke/scenarios/live_review_fix_chat/fixture"

# shellcheck source=live_e2e_auth.sh
source "$SCRIPT_DIR/live_e2e_auth.sh"

if ! has_saved_hi_credential; then
  echo "SKIP: no saved hi credential (configure a profile with hi auth / hi login, or set PIPENETWORK_API_KEY / HI_API_KEY)" >&2
  exit 2
fi
if [[ ! -d "$FIXTURE" ]]; then
  echo "FAIL: fixture missing at $FIXTURE" >&2
  exit 1
fi
if [[ ! -x "$HI_BIN" ]]; then
  echo "building hi (debug)…" >&2
  cargo build -p hi >&2 || exit 1
fi

WD="$(mktemp -d "${TMPDIR:-/tmp}/hi-live-e2e.XXXXXX")"
cleanup() {
  if [[ -n "${HI_LIVE_E2E_KEEP:-}" && "${status:-1}" -ne 0 ]]; then
    echo "keeping workdir $WD" >&2
    return
  fi
  rm -rf "$WD"
}
status=1
trap cleanup EXIT

mkdir -p "$WD/repo" "$WD/xdg-state"
cp -R "$FIXTURE/." "$WD/repo/"
# Offline verify in the child needs a lockfile; generate it here (no deps).
( cd "$WD/repo" && cargo generate-lockfile --offline >/dev/null 2>&1 || cargo generate-lockfile >&2 )
( cd "$WD/repo" && git init -q && git add -A && git -c user.email=t@t -c user.name=t commit -qm init )

echo "== preflight: fixture tests must fail before hi runs ==" >&2
if ( cd "$WD/repo" && cargo test --offline --test integration >/dev/null 2>&1 ); then
  echo "FAIL: buggy fixture already passed cargo test — e2e has nothing to fix" >&2
  exit 1
fi

PROMPT='Review this IRC-style chat server for major correctness bugs and fix them.

Read tests/integration.rs and src/main.rs first. The integration tests fail today.

Do not use bash or any other shell. Use read, grep, and edit/write only. cargo test --offline runs automatically after your edits.

Keep the existing tests; do not weaken or delete them. Make the tests pass by fixing the server.'

REPORT="$WD/report.json"
HI_ARGS=(
  --allow-unverified --no-save --no-memory --no-finalize
  --temperature 0 --max-tokens 16384 --max-steps 80
  --max-verify-repairs 8 --verify "cargo test --offline"
  --report "$REPORT"
)
apply_live_route
if [[ -n "${HI_MODEL:-}" ]]; then
  HI_ARGS+=(--model "$HI_MODEL")
fi
echo "== review-fix live e2e: credential=$(live_credential_label) route=$(live_route_label) model=${HI_MODEL:-profile} workdir=$WD/repo ==" >&2

set +e
( cd "$WD/repo" && \
  XDG_STATE_HOME="$WD/xdg-state" \
  HI_DISABLE_UPDATE_CHECK=1 \
  HI_DISABLE_FEEDBACK=1 \
  "$HI_BIN" "${HI_ARGS[@]}" \
  "$PROMPT" < /dev/null )
hi_status=$?
set -e

echo "-- hi exit $hi_status --" >&2
if [[ -f "$REPORT" ]]; then
  python3 - "$REPORT" <<'PY' >&2
import json, sys
p = sys.argv[1]
try:
    r = json.load(open(p))
except Exception as e:
    print(f"report unreadable: {e}")
    sys.exit(0)
usage = (r.get("usage") or {}).get("session") or {}
turn = (r.get("usage") or {}).get("turn") or {}
ver = r.get("verification") or {}
print(
    "tokens session={}/{} turn={}/{} verify={} stages={}".format(
        usage.get("input_tokens"),
        usage.get("output_tokens"),
        turn.get("input_tokens"),
        turn.get("output_tokens"),
        ver.get("status"),
        ver.get("rounds"),
    )
)
PY
fi

echo "== postflight: cargo test --offline must pass, tests must still exist ==" >&2
if ! grep -q 'read_until("Welcome")' "$WD/repo/tests/integration.rs"; then
  echo "FAIL: integration test no longer checks Welcome" >&2
  exit 1
fi
if ! grep -q 'after kick' "$WD/repo/tests/integration.rs"; then
  echo "FAIL: integration test no longer checks kick delivery" >&2
  exit 1
fi
if ! ( cd "$WD/repo" && cargo test --offline --test integration >&2 ); then
  echo "FAIL: cargo test --offline still fails after hi ($WD/repo)" >&2
  exit 1
fi
if [[ "$hi_status" -ne 0 ]]; then
  echo "FAIL: tests pass but hi exited $hi_status" >&2
  exit 1
fi
if [[ -f "$REPORT" ]]; then
  if ! python3 - "$REPORT" <<'PY' >&2
import json, sys
p = sys.argv[1]
try:
    r = json.load(open(p))
except Exception as e:
    print(f"report unreadable: {e}")
    sys.exit(1)
asst = str(r.get("assistant_response") or "").strip()
turn_end = str(r.get("turn_end") or (r.get("outcome") or {}).get("turn_end") or "")
outcome = r.get("outcome") or {}
status = str(outcome.get("status") or "")
error = r.get("error") or {}
kind = str(error.get("kind") or "")
tools = r.get("tools") or []
omitted = []
for tool in tools:
    out = str(tool.get("output") or "")
    if out.endswith(" · omitted") and out.startswith("read "):
        path = out[len("read "):].split(" · ", 1)[0]
        if path and not path.startswith("·") and path not in omitted:
            omitted.append(path)
if status == "failed" or kind in ("empty_stop", "tool_storm"):
    print("FAIL: review-and-fix ended as an error stop status={} kind={}".format(status, kind))
    sys.exit(1)
if not asst:
    print("FAIL: empty assistant_response after review-and-fix (silent inspect-stop)")
    print("turn_end={} tools={} omitted={}".format(turn_end, len(tools), omitted))
    sys.exit(1)
if "stopped repeating the same inspect" in turn_end and not asst:
    print("FAIL: inspect-repeat closed the turn with no verdict")
    sys.exit(1)
if len(omitted) >= 8:
    print("FAIL: cheap-shrink hid unique files: {}".format(omitted))
    sys.exit(1)
print("assistant_chars={} turn_end={} omitted_unique={}".format(len(asst), turn_end, len(omitted)))
PY
  then
    echo "FAIL: review-and-fix report failed silent-stop / unique-stub checks ($REPORT)" >&2
    exit 1
  fi
fi

# Second turn: a verify-only follow-up whose wording contains the noun
# "change". A false-positive mutation classifier used to spend recovery on
# "no file changes" after a green cargo test and settle as no_progress.
VERIFY_PROMPT="Run cargo test to verify the welcome-message change didn't break anything."
VERIFY_REPORT="$WD/report-verify.json"
HI_ARGS=(
  --allow-unverified --no-save --no-memory --no-finalize
  --temperature 0 --max-tokens 8192 --max-steps 24
  --max-verify-repairs 2 --verify "cargo test --offline"
  --report "$VERIFY_REPORT"
)
apply_live_route
if [[ -n "${HI_MODEL:-}" ]]; then
  HI_ARGS+=(--model "$HI_MODEL")
fi
echo "== follow-up verify-only turn: $VERIFY_PROMPT ==" >&2
set +e
( cd "$WD/repo" && \
  XDG_STATE_HOME="$WD/xdg-state" \
  HI_DISABLE_UPDATE_CHECK=1 \
  HI_DISABLE_FEEDBACK=1 \
  "$HI_BIN" "${HI_ARGS[@]}" \
  "$VERIFY_PROMPT" < /dev/null )
verify_status=$?
set -e
echo "-- verify-only hi exit $verify_status --" >&2
if [[ ! -f "$VERIFY_REPORT" ]]; then
  echo "FAIL: verify-only follow-up wrote no report" >&2
  exit 1
fi
if ! python3 - "$VERIFY_REPORT" <<'PY' >&2
import json, sys
p = sys.argv[1]
try:
    r = json.load(open(p))
except Exception as e:
    print(f"verify report unreadable: {e}")
    sys.exit(1)
outcome = r.get("outcome") or {}
status = str(outcome.get("status") or "")
stop = str(outcome.get("stop_reason") or "")
print("verify-only outcome status={} stop_reason={}".format(status, stop))
if status == "failed" or stop in ("no_progress", "stalled"):
    sys.exit(1)
PY
then
  echo "FAIL: verify-only follow-up settled as failed/no_progress ($VERIFY_REPORT)" >&2
  exit 1
fi
if [[ "$verify_status" -ne 0 ]]; then
  echo "FAIL: verify-only follow-up exited $verify_status after a green suite" >&2
  exit 1
fi

echo "PASS: pipenetwork live e2e fixed the chat server, cargo test --offline is green, and a verify-only follow-up did not stall" >&2
status=0
exit 0
