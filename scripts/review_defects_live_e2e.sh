#!/usr/bin/env bash
# Positive live end-to-end for the defects-only /review path via its headless
# entry, `hi --spec-review --spec-review-audit-only`, against pipenetwork.ai.
#
# The scenario is the one a large repository with no plan.md/spec.md gets by
# default: the audit narrows to recent work (the uncommitted changes) and must
# actually find what is wrong in them. The fixture is a tiny, correct Rust
# ledger crate padded past the large-workspace limit (200 files) and committed
# clean; then two defects are planted in `src/lib.rs` as an uncommitted change:
#
#   * `totals` overwrites a name's running total instead of adding to it
#     (wrong results / lost entries)
#   * `render` drops the last entry when writing the ledger back (data loss)
#
# Neither is covered by the crate's tests, so the seed (`cargo test --offline`)
# passes and any finding has to come from the model reading the diff. Expected:
#
#   scope    -> 1 uncommitted file (git status), no plan/spec, defects only
#   audit    -> at least one P0/P1 finding cited at src/lib.rs, not seeded
#   read     -> 1/1 files in scope read (no partial audit)
#   tree     -> untouched apart from the planted change (audit turns are read-only)
#   exit 4   -> open P0/P1, audit only
#
# Credentials: the saved default profile in the local hi config (its sealed
# key in the private store), or PIPENETWORK_API_KEY / HI_API_KEY if exported.
# HI_PROFILE or HI_PROVIDER selects a different route.
#
# Budget is sized for one audit turn plus its continuations:
#   --max-tokens 8192 · --max-steps 30 · ~5-minute wall clock worst case
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
EXPECTED_EXIT=4
# One more than hi's large-workspace limit, so the audit narrows to git.
PAD_FILES=201

# shellcheck source=live_e2e_auth.sh
source "$SCRIPT_DIR/live_e2e_auth.sh"

if ! has_saved_hi_credential; then
  echo "SKIP: no saved hi credential (configure a profile with hi auth / hi login, or set PIPENETWORK_API_KEY / HI_API_KEY)" >&2
  exit 2
fi
if [[ ! -x "$HI_BIN" ]]; then
  echo "building hi (debug)…" >&2
  cargo build -p hi >&2 || exit 1
fi

WD="$(mktemp -d "${TMPDIR:-/tmp}/hi-review-defects-e2e.XXXXXX")"
cleanup() {
  if [[ -n "${HI_LIVE_E2E_KEEP:-}" && "${status:-1}" -ne 0 ]]; then
    echo "keeping workdir $WD" >&2
    return
  fi
  rm -rf "$WD"
}
status=1
trap cleanup EXIT

REPO="$WD/repo"
mkdir -p "$REPO/src" "$REPO/tests" "$REPO/docs/notes" "$WD/xdg-state"

cat > "$REPO/Cargo.toml" <<'TOML'
[package]
name = "ledger"
version = "0.1.0"
edition = "2021"

[dependencies]
TOML

cat > "$REPO/.gitignore" <<'GI'
/target/
/.hi/
GI

cat > "$REPO/src/lib.rs" <<'RS'
//! A tiny append-only ledger: `name,amount` lines, one entry per line. Every
//! line for a name counts; `totals` sums them and `render` writes every total
//! back out so the ledger can be rewritten without losing an entry.

use std::collections::BTreeMap;

/// `alice, 40` -> `("alice", 40)`. Blank names and bad amounts are skipped
/// by the caller, so this returns `None` for them.
pub fn parse_line(line: &str) -> Option<(String, i64)> {
    let (name, amount) = line.split_once(',')?;
    let name = name.trim();
    if name.is_empty() {
        return None;
    }
    Some((name.to_string(), amount.trim().parse().ok()?))
}

/// Running total per name over every parseable line, in name order.
pub fn totals(lines: &[&str]) -> BTreeMap<String, i64> {
    let mut out = BTreeMap::new();
    for line in lines {
        if let Some((name, amount)) = parse_line(line) {
            *out.entry(name).or_insert(0) += amount;
        }
    }
    out
}

/// The totals as ledger lines again, one `name,amount` line per entry.
pub fn render(totals: &BTreeMap<String, i64>) -> String {
    let mut out = String::new();
    for (name, amount) in totals {
        out.push_str(&format!("{name},{amount}\n"));
    }
    out
}
RS

cat > "$REPO/src/main.rs" <<'RS'
//! `ledger <file>`: print the running total per name.

use std::process::ExitCode;

fn main() -> ExitCode {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: ledger <file>");
        return ExitCode::from(2);
    };
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("ledger: {path}: {err}");
            return ExitCode::from(1);
        }
    };
    let lines: Vec<&str> = text.lines().collect();
    print!("{}", ledger::render(&ledger::totals(&lines)));
    ExitCode::SUCCESS
}
RS

# The tests cover distinct names only, so the planted defects do not fail
# them: the seed passes and the finding must come from reading the diff.
cat > "$REPO/tests/ledger.rs" <<'RS'
use ledger::{parse_line, totals};

#[test]
fn parses_name_and_amount() {
    assert_eq!(parse_line("alice, 40"), Some(("alice".to_string(), 40)));
    assert_eq!(parse_line(", 40"), None);
    assert_eq!(parse_line("alice, forty"), None);
    assert_eq!(parse_line("no comma"), None);
}

#[test]
fn totals_each_name() {
    let totals = totals(&["alice,40", "bob,2", "bad line"]);
    assert_eq!(totals.get("alice"), Some(&40));
    assert_eq!(totals.get("bob"), Some(&2));
    assert_eq!(totals.len(), 2);
}
RS

for i in $(seq 1 "$PAD_FILES"); do
  printf '# Note %s\n\nRelease note placeholder.\n' "$i" > "$REPO/docs/notes/n$i.md"
done

( cd "$REPO" && cargo generate-lockfile --offline >/dev/null 2>&1 || cargo generate-lockfile >&2 )
( cd "$REPO" && git init -q && git add -A && git -c user.email=t@t -c user.name=t commit -qm "ledger" )

echo "== preflight: the committed crate is correct and its tests pass ==" >&2
if ! ( cd "$REPO" && cargo test --offline >/dev/null 2>&1 ); then
  echo "FAIL: the clean fixture does not pass cargo test" >&2
  exit 1
fi

# Plant the defects as an uncommitted change to src/lib.rs.
python3 - "$REPO/src/lib.rs" <<'PY'
import sys
path = sys.argv[1]
text = open(path).read()
before = text
text = text.replace(
    "            *out.entry(name).or_insert(0) += amount;\n",
    "            out.insert(name, amount);\n",
)
text = text.replace(
    "    for (name, amount) in totals {\n",
    "    for (name, amount) in totals.iter().take(totals.len().saturating_sub(1)) {\n",
)
assert text != before and text.count("out.insert(name, amount)") == 1 and "saturating_sub(1)" in text
open(path, "w").write(text)
PY

if ! ( cd "$REPO" && cargo test --offline >/dev/null 2>&1 ); then
  echo "FAIL: the planted defects fail cargo test; the seed would hand them to the model" >&2
  exit 1
fi
PLANTED_SHA="$(sha256sum "$REPO/src/lib.rs" | cut -d' ' -f1)"
if [[ "$(cd "$REPO" && git status --porcelain)" != " M src/lib.rs" ]]; then
  echo "FAIL: expected exactly one uncommitted change (src/lib.rs), got:" >&2
  ( cd "$REPO" && git status --porcelain >&2 )
  exit 1
fi

REPORT="$WD/report.json"
LOG="$WD/hi.log"
HI_ARGS=(
  --allow-unverified --no-save --no-memory --no-finalize
  --temperature 0 --max-tokens 8192 --max-steps 30
  --verify "cargo test --offline"
  --report "$REPORT"
  --spec-review-audit-only
  --spec-review
)
apply_live_route
if [[ -n "${HI_MODEL:-}" ]]; then
  HI_ARGS=(--model "$HI_MODEL" "${HI_ARGS[@]}")
fi
echo "== /review defects-only live e2e: credential=$(live_credential_label) route=$(live_route_label) model=${HI_MODEL:-profile} workdir=$REPO ==" >&2

set +e
( cd "$REPO" && \
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
read = review.get("read") or {}
git = (review.get("inputs") or {}).get("git") or {}
print(
    "tokens session={}/{} review phase={} verdict={} open_blocking={} git={} read={}/{} partial={} verify={}".format(
        usage.get("input_tokens"),
        usage.get("output_tokens"),
        review.get("phase"),
        review.get("verdict"),
        review.get("open_blocking"),
        git.get("files"),
        len(read.get("files") or []),
        read.get("scope_total"),
        read.get("partial"),
        (review.get("verify") or {}).get("passed"),
    )
)
for f in review.get("findings") or []:
    print("  finding  [{}] {}  {}{}".format(f.get("severity"), f.get("title"), f.get("location") or "-", "  (seeded)" if f.get("from_verify") else ""))
PY

echo "== postflight: the audit is read-only; only the planted change is in the tree ==" >&2
if [[ "$(sha256sum "$REPO/src/lib.rs" | cut -d' ' -f1)" != "$PLANTED_SHA" ]]; then
  echo "FAIL: src/lib.rs changed during the audit (audit turns must not edit)" >&2
  exit 1
fi
if [[ "$(cd "$REPO" && git status --porcelain)" != " M src/lib.rs" ]]; then
  echo "FAIL: the audit left other changes in the tree:" >&2
  ( cd "$REPO" && git status --porcelain >&2 )
  exit 1
fi
if ! grep -q 'no plan/spec (defects only)' "$LOG"; then
  echo "FAIL: the run did not announce a defects-only audit" >&2
  exit 1
fi
if ! grep -q '1 uncommitted file (git status' "$LOG"; then
  echo "FAIL: the run did not narrow to the one uncommitted file" >&2
  exit 1
fi

echo "== report: git scope, an unseeded P0/P1 at src/lib.rs, every file read, exit $EXPECTED_EXIT ==" >&2
if ! python3 - "$REPORT" "$EXPECTED_EXIT" <<'PY' >&2
import json, sys
r = json.load(open(sys.argv[1]))
expected = int(sys.argv[2])
review = r.get("review") or {}
fail = []
if review.get("phase") != "done":
    fail.append("phase={} stop_reason={}".format(review.get("phase"), review.get("stop_reason")))
if not review.get("audit_only"):
    fail.append("audit_only={}".format(review.get("audit_only")))
if int(review.get("passes") or 0) != 0:
    fail.append("a fix pass ran in audit-only mode (passes={})".format(review.get("passes")))
git = (review.get("inputs") or {}).get("git") or {}
if (git.get("source") or {}).get("kind") != "uncommitted":
    fail.append("git scope source={} (expected uncommitted)".format(git.get("source")))
if git.get("files") != ["src/lib.rs"]:
    fail.append("git scope files={} (expected ['src/lib.rs'])".format(git.get("files")))
if (review.get("inputs") or {}).get("files"):
    fail.append("inputs.files={} (expected none: no plan/spec)".format(review["inputs"]["files"]))
if review.get("coverage"):
    fail.append("coverage rows in a defects-only audit: {}".format(review.get("coverage")))
findings = review.get("findings") or []
hits = [
    f for f in findings
    if f.get("severity") in ("P0", "P1")
    and str(f.get("location") or "").startswith("src/lib.rs")
    and not f.get("from_verify")
]
if not hits:
    fail.append("no unseeded P0/P1 finding cited at src/lib.rs; findings={}".format(
        [(f.get("severity"), f.get("title"), f.get("location"), f.get("from_verify")) for f in findings]))
if int(review.get("open_blocking") or 0) < 1:
    fail.append("open_blocking={}".format(review.get("open_blocking")))
verify = review.get("verify") or {}
if verify.get("command") != "cargo test --offline" or verify.get("passed") is not True:
    fail.append("verify seed={} (expected `cargo test --offline` passed)".format(verify))
read = review.get("read") or {}
if read.get("scope_total") != 1 or read.get("unread_count") != 0 or read.get("partial"):
    fail.append("read coverage {}/{} unread={} partial={} (expected 1/1, none unread)".format(
        len(read.get("files") or []), read.get("scope_total"), read.get("unread_count"), read.get("partial")))
if review.get("changed_files") or review.get("audit_changed_files"):
    fail.append("the audit changed files: changed={} audit_changed={}".format(
        review.get("changed_files"), review.get("audit_changed_files")))
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
  echo "FAIL: hi exited $hi_status, expected $EXPECTED_EXIT (open P0/P1, audit only)" >&2
  exit 1
fi

echo "PASS: the defects-only audit narrowed to the uncommitted src/lib.rs, read it, reported the planted defect as P0/P1, changed nothing, exit $EXPECTED_EXIT" >&2
status=0
exit 0
