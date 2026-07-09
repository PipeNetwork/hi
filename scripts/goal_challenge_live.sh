#!/bin/zsh
# A real /goal challenge through the dashboard's goal-row engine, emulated
# faithfully: worktree row, --goal first turn, per-turn --report, parent
# auto-continue with GOAL_CONTINUE_PROMPT while the goal is active, verify-gated
# merge after every turn (held merges keep driving), base refresh after merges,
# stall guard at 2. Low --max-steps forces the multi-turn cadence.
setopt NULL_GLOB
set -u
# Usage: PIPENETWORK_API_KEY=... scripts/<name>.sh [hi-binary] [workdir]
HIBIN="${1:-$PWD/target/release/hi}"
KEY="${PIPENETWORK_API_KEY:-${HI_API_KEY:-}}"
[ -z "$KEY" ] && { echo "SKIP: set PIPENETWORK_API_KEY" >&2; exit 2; }
SB="${2:-$(mktemp -d /tmp/hi-fleet-test.XXXXXX)}"
echo "workspace: $SB"
GD="$SB/goal-challenge"; rm -rf "$GD" "$GD"-wt; mkdir -p "$GD"
cd "$GD" || exit 1
git init -q
git -c user.email=t@t -c user.name=t commit -q --allow-empty -m base

OBJECTIVE="Build 'kan', a complete kanban board CLI in Python in this directory: storage.py (persist boards to kanban.json), models.py (Board/Column/Card with validation), cli.py using argparse subcommands (init, add, move, list, delete, stats) with tag filtering on list, unit tests covering storage, models, and every CLI subcommand under tests/, and a README.md with usage examples. python3 -m unittest discover -s tests -t . must pass."
VERIFY="python3 -m unittest discover -s tests -t ."
DRIVE="Continue the long-horizon goal: complete the active sub-goal now, then update the plan with update_plan — including any newly discovered steps."
S="$SB/goal-challenge.jsonl"; R="$SB/goal-challenge.report.json"
rm -f "$S" "$R"
MAX_TURNS=8

BASE=$(git rev-parse HEAD)
WT="$GD-wt"
git worktree add -q "$WT" "$BASE"

goal_field() { python3 -c "import json,sys
try:
  g=json.load(open('$R')).get('$1' if False else 'goal')
  print(g.get(sys.argv[1]) if g else '')
except Exception: print('')" "$1" 2>/dev/null; }
tokens() { python3 -c "
try:
  import json; print(json.load(open('$R')).get('total_tokens',0))
except Exception: print(0)" 2>/dev/null; }
goal_json() { python3 -c "
try:
  import json; g=json.load(open('$R')).get('goal'); import sys; print(json.dumps(g,sort_keys=True) if g else '')
except Exception: print('')" 2>/dev/null; }

clean_wt() { find "$WT" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null; }
merge_try() {
  clean_wt
  git -C "$WT" add -A 2>/dev/null
  if [ -z "$(git -C "$WT" diff --cached --name-only "$BASE")" ]; then echo "nochange"; return; fi
  # verify gate in the worktree (dashboard: launcher.verify)
  if ! ( cd "$WT" && PYTHONDONTWRITEBYTECODE=1 sh -c "$VERIFY" >/dev/null 2>&1 ); then
    echo "held"; return
  fi
  clean_wt; git -C "$WT" add -A 2>/dev/null
  if git -C "$WT" diff --cached --binary "$BASE" | git apply --whitespace=nowarn 2>/dev/null; then
    # post-merge: refresh base (dashboard: checkpoint + reset_to)
    git add -A 2>/dev/null && git -c user.email=t@t -c user.name=t commit -qm "merge" 2>/dev/null
    BASE=$(git rev-parse HEAD)
    git -C "$WT" reset --hard -q "$BASE" 2>/dev/null
    echo "merged"
  else
    echo "applyfail"
  fi
}

echo "== /goal challenge: kanban CLI · max-steps 25/turn · verify-gated merges =="
STALL=0; PREV_GOAL=""
for TURN in $(seq 1 $MAX_TURNS); do
  if [ "$TURN" = 1 ]; then
    ARGS=(--goal "$OBJECTIVE"); PROMPT="$OBJECTIVE"; KIND="dispatch"
  else
    ARGS=(); PROMPT="$DRIVE"; KIND="drive"
  fi
  START=$(date +%s)
  ( cd "$WT" && HI_API_KEY="$KEY" "$HIBIN" --provider pipenetwork --model ipop/coder-balanced \
      --session-file "$S" --report "$R" --max-steps 25 \
      --verify "$VERIFY" --max-verify 2 "${ARGS[@]}" \
      "$PROMPT" < /dev/null > "$SB/goal-challenge-t$TURN.log" 2>&1 & p=$!
    ( sleep 420; kill -9 $p 2>/dev/null ) & w=$!; wait $p 2>/dev/null; kill $w 2>/dev/null )
  ELAPSED=$(( $(date +%s) - START ))
  DONE=$(goal_field done); TOTAL=$(goal_field total); STATUS=$(goal_field status); PAUSED=$(goal_field paused)
  TOK=$(tokens); GJ=$(goal_json)
  MERGE=$(merge_try)
  echo "turn $TURN [$KIND] ${ELAPSED}s · goal $DONE/$TOTAL ($STATUS) · tok $TOK · merge: $MERGE"
  # stall guard (dashboard: next_drive_stall)
  if [ "$KIND" = "drive" ] && [ "$GJ" = "$PREV_GOAL" ]; then
    STALL=$((STALL+1))
    if [ "$STALL" -ge 2 ]; then echo "⏸ drive parked — no progress for 2 turns"; break; fi
  else
    STALL=0
  fi
  PREV_GOAL="$GJ"
  # continue predicate (dashboard: should_auto_drive via report)
  if [ "$STATUS" != "Active" ] || [ "$PAUSED" = "True" ]; then
    echo "◎ goal no longer active ($STATUS) — drive stops"
    break
  fi
done

echo ""
echo "== final merge state (force any residue) =="
MERGE=$(merge_try); echo "final merge_try: $MERGE"
echo ""
echo "== ground truth in the REAL tree =="
ls *.py README.md 2>/dev/null | tr '\n' ' '; echo ""
ls tests/ 2>/dev/null | tr '\n' ' '; echo ""
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -t . 2>&1 | tail -2
echo "== exercise the CLI for real =="
PYTHONDONTWRITEBYTECODE=1 python3 cli.py init 2>&1 | head -2
PYTHONDONTWRITEBYTECODE=1 python3 cli.py add todo "ship the fleet" --tag urgent 2>&1 | head -2
PYTHONDONTWRITEBYTECODE=1 python3 cli.py list 2>&1 | head -6
echo "== session/report =="
wc -l < "$S" | xargs echo "session lines:"
git worktree remove --force "$WT" 2>/dev/null
echo "== worktree cleaned =="
