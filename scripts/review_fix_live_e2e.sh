#!/usr/bin/env bash
# Live end-to-end: drive real hi against pipenetwork.ai on a buggy std-only
# IRC chat server (Welcome never reaches the socket; KICK does not drop the
# target). hi must read the failing tests, fix the server, and pass
# `cargo test --offline` through the verify loop. A second verify-only turn
# ("Run cargo test to verify the welcome-message change…") must complete
# rather than stall as no_progress after a green suite.
#
# Credentials (first match wins):
#   1. PIPENETWORK_API_KEY / HI_API_KEY in the environment
#   2. The hi client's saved pipenetwork profile in ~/.config/hi
#      (auth-store:// api_key_ref, or the pipenetwork pairing key)
#
# This is deliberately not a one-token canary. Budget is large enough for a
# multi-round review/fix/re-test loop:
#   --max-tokens 16384 · --max-steps 80 · --max-verify-repairs 8
#   ~20-minute wall clock (review/fix plus a verify-only follow-up)
#
# Optional:
#   HI_MODEL   (default: the saved pipenetwork profile model)
#   HI_BIN     (default: freshly built ./target/debug/hi)
#   HI_LIVE_E2E_KEEP=1  keep the workdir on failure
#
# Exit: 0 pass · 1 fail · 2 skipped (no key).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

HI_BIN="${HI_BIN:-$PWD/target/debug/hi}"
FIXTURE="$PWD/bench/tui-smoke/scenarios/live_review_fix_chat/fixture"

has_pipe_credential() {
  python3 - <<'PY'
import json, os
from pathlib import Path

def nonempty(name):
    return bool(os.environ.get(name, "").strip())

if nonempty("PIPENETWORK_API_KEY") or nonempty("HI_API_KEY"):
    raise SystemExit(0)

auth_path = Path.home() / ".config" / "hi" / "auth.json"
if not auth_path.is_file():
    raise SystemExit(1)
try:
    data = json.loads(auth_path.read_text())
except Exception:
    raise SystemExit(1)
if not isinstance(data, dict):
    raise SystemExit(1)
for key, value in data.items():
    if "pipenetwork" not in str(key):
        continue
    access = value.get("access", "") if isinstance(value, dict) else ""
    if str(access).strip():
        raise SystemExit(0)
raise SystemExit(1)
PY
}

if ! has_pipe_credential; then
  echo "SKIP: no pipenetwork credential (set PIPENETWORK_API_KEY / HI_API_KEY, or save one with hi auth pipenetwork / the default pipenetwork profile)" >&2
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
  --profile pipenetwork
  --allow-unverified --no-save --no-memory --no-finalize
  --temperature 0 --max-tokens 16384 --max-steps 80
  --max-verify-repairs 8 --verify "cargo test --offline"
  --report "$REPORT"
)
if [[ -n "${HI_MODEL:-}" ]]; then
  HI_ARGS+=(--model "$HI_MODEL")
fi
if [[ -n "${PIPENETWORK_API_KEY:-${HI_API_KEY:-}}" ]]; then
  echo "== review-fix live e2e: credential=env model=${HI_MODEL:-profile} workdir=$WD/repo ==" >&2
else
  echo "== review-fix live e2e: credential=hi-client-auth-store model=${HI_MODEL:-profile} workdir=$WD/repo ==" >&2
fi

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

# Second turn: a verify-only follow-up whose wording contains the noun
# "change". A false-positive mutation classifier used to spend recovery on
# "no file changes" after a green cargo test and settle as no_progress.
VERIFY_PROMPT="Run cargo test to verify the welcome-message change didn't break anything."
VERIFY_REPORT="$WD/report-verify.json"
HI_ARGS=(
  --profile pipenetwork
  --allow-unverified --no-save --no-memory --no-finalize
  --temperature 0 --max-tokens 8192 --max-steps 24
  --max-verify-repairs 2 --verify "cargo test --offline"
  --report "$VERIFY_REPORT"
)
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
