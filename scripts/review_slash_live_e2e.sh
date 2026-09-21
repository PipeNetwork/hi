#!/usr/bin/env bash
# Live end-to-end for the /review slash command via its headless entry,
# `hi --spec-review`, against pipenetwork.ai.
#
# Fixture: the buggy std-only IRC chat server from live_review_fix_chat
# (Welcome never reaches the socket; KICK does not drop the target), plus a
# SPEC.md and plan.md written here. The spec lists Welcome and KICK, which the
# server attempts, and a `/topic` command that is deliberately not implemented
# and unchecked in the plan. One run therefore exercises both halves:
#
#   audit    -> coverage marks /topic missing; P0/P1 findings for the two bugs
#   fix loop -> the P0/P1 defects are fixed; `cargo test --offline` goes green
#   re-audit -> no open P0/P1; /topic is still missing (reported, never built)
#   exit 3   -> incomplete coverage, no open defects
#
# Credentials (first match wins):
#   1. PIPENETWORK_API_KEY / HI_API_KEY in the environment
#   2. The hi client's saved pipenetwork profile in ~/.config/hi
#
# Budget is sized for audit + fix + re-audit (up to 3 passes):
#   --max-tokens 16384 · --max-steps 80 · --max-verify-repairs 8
#   ~25-minute wall clock worst case
#
# Optional:
#   HI_PROFILE (default: `--provider pipenetwork`, which borrows the saved
#              Pipe credential from whichever profile targets pipenetwork)
#   HI_MODEL   (default: that profile's model)
#   HI_BIN     (default: freshly built ./target/debug/hi)
#   HI_LIVE_E2E_KEEP=1  keep the workdir on failure
#
# Exit: 0 pass · 1 fail · 2 skipped (no key).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

HI_BIN="${HI_BIN:-$PWD/target/debug/hi}"
FIXTURE="$PWD/bench/tui-smoke/scenarios/live_review_fix_chat/fixture"
EXPECTED_EXIT=3

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

WD="$(mktemp -d "${TMPDIR:-/tmp}/hi-review-e2e.XXXXXX")"
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

cat > "$WD/repo/SPEC.md" <<'MD'
# Chat server spec

A line-oriented TCP chat server. Each client connection is a session.

## Session

- On connect the server sends `Welcome to chat. Type HELP for commands.` as
  the first line before the client sends anything.
- `REGISTER <nick>` (alias `LOGIN`) sets the session nick and replies
  `OK registered as <nick>`.

## Channels

- `JOIN <channel>` subscribes the session and replies `OK joined <channel>`.
  The first member of a channel becomes its operator.
- `PRIVMSG <channel> :<text>` fans the message out to every subscribed
  session as `PRIVMSG <channel> <nick> :<text>`.
- `KICK <channel> <nick>`: only the operator may kick. The target receives
  `KICKED <channel>` and is removed from the channel, so later `PRIVMSG`
  traffic on that channel is never delivered to the kicked session.
- `TOPIC <channel> :<text>` (the `/topic` feature): the operator sets the
  channel topic; every member receives `TOPIC <channel> :<text>`, and a
  session that joins later receives the current topic right after
  `OK joined <channel>`.

## Verification

`cargo test --offline` runs `tests/integration.rs`, which covers Welcome,
registration, join, fan-out and kick delivery.
MD

cat > "$WD/repo/plan.md" <<'MD'
# Plan

- [x] Welcome line is sent first on connect
- [x] REGISTER / LOGIN sets the nick
- [x] JOIN subscribes and makes the first member operator
- [x] PRIVMSG fan-out to channel members
- [x] KICK removes the target from the channel fan-out
- [ ] /topic: TOPIC command with broadcast and replay on join
MD

# Offline verify in the child needs a lockfile; generate it here (no deps).
( cd "$WD/repo" && cargo generate-lockfile --offline >/dev/null 2>&1 || cargo generate-lockfile >&2 )
( cd "$WD/repo" && git init -q && git add -A && git -c user.email=t@t -c user.name=t commit -qm init )

echo "== preflight: fixture tests must fail before hi runs ==" >&2
if ( cd "$WD/repo" && cargo test --offline --test integration >/dev/null 2>&1 ); then
  echo "FAIL: buggy fixture already passed cargo test — e2e has nothing to fix" >&2
  exit 1
fi
if grep -qi 'topic' "$WD/repo/src/main.rs"; then
  echo "FAIL: fixture already mentions topic; the missing-item probe is void" >&2
  exit 1
fi

REPORT="$WD/report.json"
LOG="$WD/hi.log"
HI_ARGS=(
  --allow-unverified --no-save --no-memory --no-finalize
  --temperature 0 --max-tokens 16384 --max-steps 80
  --max-verify-repairs 8 --verify "cargo test --offline"
  --report "$REPORT"
  --spec-review-passes 3
  --spec-review
)
if [[ -n "${HI_PROFILE:-}" ]]; then
  HI_ARGS=(--profile "$HI_PROFILE" "${HI_ARGS[@]}")
else
  HI_ARGS=(--provider pipenetwork "${HI_ARGS[@]}")
fi
if [[ -n "${HI_MODEL:-}" ]]; then
  HI_ARGS=(--model "$HI_MODEL" "${HI_ARGS[@]}")
fi
if [[ -n "${PIPENETWORK_API_KEY:-${HI_API_KEY:-}}" ]]; then
  credential=env
else
  credential=hi-client-auth-store
fi
echo "== /review live e2e: credential=$credential route=${HI_PROFILE:+profile $HI_PROFILE}${HI_PROFILE:-provider pipenetwork} model=${HI_MODEL:-profile} workdir=$WD/repo ==" >&2

set +e
( cd "$WD/repo" && \
  XDG_STATE_HOME="$WD/xdg-state" \
  HI_DISABLE_UPDATE_CHECK=1 \
  HI_DISABLE_FEEDBACK=1 \
  "$HI_BIN" "${HI_ARGS[@]}" < /dev/null 2>&1 | tee "$LOG" >&2 )
hi_status=${PIPESTATUS[0]}
set -e

echo "-- hi exit $hi_status (expected $EXPECTED_EXIT) --" >&2
if [[ ! -f "$REPORT" ]]; then
  echo "FAIL: hi wrote no report at $REPORT" >&2
  exit 1
fi

python3 - "$REPORT" <<'PY' >&2
import json, sys
r = json.load(open(sys.argv[1]))
usage = (r.get("usage") or {}).get("session") or {}
review = r.get("review") or {}
print(
    "tokens session={}/{} review phase={} passes={} verdict={} open_blocking={} changed={}".format(
        usage.get("input_tokens"),
        usage.get("output_tokens"),
        review.get("phase"),
        review.get("passes"),
        review.get("verdict"),
        review.get("open_blocking"),
        review.get("changed_files"),
    )
)
for row in review.get("coverage") or []:
    print("  coverage {:<12} {}  {}".format(row.get("state"), row.get("item"), row.get("location") or "-"))
for f in review.get("findings") or []:
    print("  finding  [{}] {}  {}".format(f.get("severity"), f.get("title"), f.get("location") or "-"))
PY

echo "== postflight: the fix loop must have fixed the server without touching the tests ==" >&2
if ! grep -q 'read_until("Welcome")' "$WD/repo/tests/integration.rs"; then
  echo "FAIL: integration test no longer checks Welcome" >&2
  exit 1
fi
if ! grep -q 'after kick' "$WD/repo/tests/integration.rs"; then
  echo "FAIL: integration test no longer checks kick delivery" >&2
  exit 1
fi
if ! ( cd "$WD/repo" && cargo test --offline --test integration >&2 ); then
  echo "FAIL: cargo test --offline still fails after the fix loop ($WD/repo)" >&2
  exit 1
fi
if ! grep -q 'fix pass 1/' "$LOG"; then
  echo "FAIL: the audit never opened a fix pass, so it did not report the two bugs as P0/P1" >&2
  exit 1
fi
if grep -qi 'topic' "$WD/repo/src/main.rs"; then
  echo "FAIL: /topic was implemented by the fix loop; missing plan items must only be reported" >&2
  exit 1
fi

echo "== report: coverage marks /topic missing, no open P0/P1, exit $EXPECTED_EXIT ==" >&2
if ! python3 - "$REPORT" "$EXPECTED_EXIT" <<'PY' >&2
import json, sys
r = json.load(open(sys.argv[1]))
expected = int(sys.argv[2])
review = r.get("review") or {}
fail = []
if review.get("phase") != "done":
    fail.append("phase={} stop_reason={}".format(review.get("phase"), review.get("stop_reason")))
if int(review.get("passes") or 0) < 1:
    fail.append("no fix pass ran (passes={})".format(review.get("passes")))
if int(review.get("open_blocking") or 0) != 0:
    fail.append("open P0/P1 after the loop: {}".format(review.get("open_blocking")))
if "src/main.rs" not in (review.get("changed_files") or []):
    fail.append("src/main.rs not in changed_files={}".format(review.get("changed_files")))
if review.get("verdict") != "incomplete" or review.get("coverage_complete"):
    fail.append("verdict={} coverage_complete={}".format(review.get("verdict"), review.get("coverage_complete")))
topic = [row for row in review.get("coverage") or [] if "topic" in str(row.get("item", "")).lower()]
if not topic:
    fail.append("no coverage row mentions /topic")
elif any(row.get("state") != "missing" for row in topic):
    fail.append("/topic coverage is {} not missing".format([row.get("state") for row in topic]))
if int(review.get("exit_code", -1)) != expected:
    fail.append("report exit_code={} expected {}".format(review.get("exit_code"), expected))
for f in fail:
    print("FAIL:", f)
sys.exit(1 if fail else 0)
PY
then
  echo "FAIL: review report contract not met ($REPORT)" >&2
  exit 1
fi
if [[ "$hi_status" -ne "$EXPECTED_EXIT" ]]; then
  echo "FAIL: hi exited $hi_status, expected $EXPECTED_EXIT (incomplete coverage, no open P0/P1)" >&2
  exit 1
fi

echo "PASS: /review audited the spec (/topic missing), fixed the P0/P1 chat bugs, cargo test --offline is green, exit $EXPECTED_EXIT" >&2
status=0
exit 0
