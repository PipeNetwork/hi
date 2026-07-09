#!/bin/zsh
setopt NULL_GLOB
# Real-fleet test: 5 concurrent agents, worktree-isolated, dashboard merge
# protocol (auto-merge disjoint, hold overlaps, force-merge), final ground truth.
set -u
# Usage: PIPENETWORK_API_KEY=... scripts/<name>.sh [hi-binary] [workdir]
HIBIN="${1:-$PWD/target/release/hi}"
KEY="${PIPENETWORK_API_KEY:-${HI_API_KEY:-}}"
[ -z "$KEY" ] && { echo "SKIP: set PIPENETWORK_API_KEY" >&2; exit 2; }
SB="${2:-$(mktemp -d /tmp/hi-fleet-test.XXXXXX)}"
echo "workspace: $SB"
FL="$SB/fleet-real"; rm -rf "$FL" "$FL"-wt*; mkdir -p "$FL"
cd "$FL" || exit 1
git init -q

# --- The project: a small shop lib with real work to do ---
cat > pricing.py <<'EOF'
def total_with_tax(subtotal, rate):
    # BUG: rate applied twice
    return subtotal + subtotal * rate * rate
EOF
cat > cart.py <<'EOF'
class Cart:
    def __init__(self):
        self.items = {}

    def add_item(self, name, price, qty=1):
        self.items[name] = self.items.get(name, 0) + qty * price

    def total(self):
        return sum(self.items.values())
EOF
cat > shipping.py <<'EOF'
def shipping_cost(weight_kg):
    if weight_kg <= 0:
        raise ValueError("weight must be positive")
    if weight_kg <= 1:
        return 5.0
    if weight_kg <= 10:
        return 5.0 + (weight_kg - 1) * 1.5
    return 18.5 + (weight_kg - 10) * 1.0
EOF
mkdir -p tests
cat > tests/test_pricing.py <<'EOF'
import unittest
from pricing import total_with_tax

class TestPricing(unittest.TestCase):
    def test_tax(self):
        self.assertAlmostEqual(total_with_tax(100, 0.1), 110.0)

if __name__ == "__main__":
    unittest.main()
EOF
touch tests/__init__.py
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
git add -A && git -c user.email=t@t -c user.name=t commit -qm base
BASE=$(git rev-parse HEAD)
echo "== base $BASE — suite at base (expect pricing FAIL): =="
python3 -m unittest discover -s tests -t . 2>&1 | tail -1

# --- The fleet: 5 rows, task 2 and goal-row 5 both touch cart.py ---
declare -a TASKS
TASKS[1]="fix the bug in pricing.py: total_with_tax must be subtotal + subtotal*rate (tax applied once); tests/test_pricing.py must pass via: python3 -m unittest discover -s tests -t ."
TASKS[2]="add a remove_item(name) method to the Cart class in cart.py (raise KeyError if absent) and add tests/test_cart_remove.py covering it; run python3 -m unittest discover -s tests -t . to confirm nothing else breaks"
TASKS[3]="write thorough unit tests for shipping.py in tests/test_shipping.py covering all tiers and the error case; they must pass via python3 -m unittest discover -s tests -t ."
TASKS[4]="write a README.md documenting the pricing, cart, and shipping modules with short usage examples for each"
GOAL5="add a discounts feature: a discounts.py module with percent_off(total, pct) and bulk_discount(qty, unit_price) functions, tests in tests/test_discounts.py, and a total_with_discount(pct) method on Cart in cart.py that uses percent_off"

echo "== dispatching 5 rows concurrently =="
PIDS=()
for i in 1 2 3 4 5; do
  WT="$FL-wt$i"
  git worktree add -q "$WT" "$BASE"
  S="$SB/fleet-real-$i.jsonl"; R="$SB/fleet-real-$i.report.json"
  rm -f "$S" "$R"
  if [ "$i" = 5 ]; then
    ( cd "$WT" && HI_API_KEY="$KEY" "$HIBIN" --provider pipenetwork --model ipop/coder-balanced \
        --session-file "$S" --report "$R" --max-steps 40 --goal "$GOAL5" \
        "Begin working the goal." < /dev/null > "$SB/fleet-real-$i.log" 2>&1 ) &
  else
    ( cd "$WT" && HI_API_KEY="$KEY" "$HIBIN" --provider pipenetwork --model ipop/coder-balanced \
        --session-file "$S" --report "$R" --max-steps 40 \
        "${TASKS[$i]}" < /dev/null > "$SB/fleet-real-$i.log" 2>&1 ) &
  fi
  PIDS+=($!)
  echo "  dispatched #$i (pid ${PIDS[-1]})"
done
# Global watchdog: kill stragglers after 8 minutes.
( sleep 480; for p in "${PIDS[@]}"; do kill -9 "$p" 2>/dev/null; done ) &
WATCH=$!
for p in "${PIDS[@]}"; do wait "$p" 2>/dev/null; done
kill "$WATCH" 2>/dev/null

echo ""
echo "== per-row results (tokens · goal · changed files) =="
typeset -A CHANGED
for i in 1 2 3 4 5; do
  R="$SB/fleet-real-$i.report.json"
  TOK=$(python3 -c "import json;print(json.load(open('$R')).get('total_tokens',0))" 2>/dev/null || echo "?")
  GOAL=$(python3 -c "import json;g=json.load(open('$R')).get('goal');print(f\"{g['done']}/{g['total']}\" if g else '-')" 2>/dev/null || echo "?")
  WT="$FL-wt$i"
  find "$WT" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
  git -C "$WT" add -A 2>/dev/null
  FILES=$(git -C "$WT" diff --cached --name-only "$BASE" | tr '\n' ' ')
  CHANGED[$i]="$FILES"
  echo "  #$i tok=$TOK goal=$GOAL changed: $FILES"
done

echo ""
echo "== dashboard merge protocol (arrival order 1..5) =="
typeset -A MERGED_FILES
merge_row() {
  local i=$1 WT="$FL-wt$1"
  find "$WT" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
  git -C "$WT" add -A 2>/dev/null
  git -C "$WT" diff --cached --binary "$BASE" | git apply --whitespace=nowarn 2>/dev/null
}
for i in 1 2 3 4 5; do
  overlap=""
  for j in "${(@k)MERGED_FILES}"; do
    for f in ${=CHANGED[$i]}; do
      case " ${MERGED_FILES[$j]} " in *" $f "*) overlap="$overlap #$j($f)";; esac
    done
  done
  if [ -z "${CHANGED[$i]// /}" ]; then
    echo "  #$i: no changes — nothing to merge"
  elif [ -n "$overlap" ]; then
    echo "  #$i: ⇡ HELD — overlaps$overlap"
    HELD=$i
  else
    if merge_row "$i"; then
      echo "  #$i: ✓ auto-merged (${CHANGED[$i]})"
      MERGED_FILES[$i]="${CHANGED[$i]}"
    else
      echo "  #$i: ✗ apply failed"
    fi
  fi
done
if [ -n "${HELD:-}" ]; then
  echo "  → force-merging held #$HELD (the 'm' key):"
  if merge_row "$HELD"; then
    echo "    ✓ forced merge applied cleanly (non-overlapping hunks)"
  else
    echo "    ✗ forced merge CONFLICTED (dashboard would show 'merge failed (m retries)')"
  fi
fi

echo ""
echo "== ground truth: merged real tree =="
ls *.py README.md 2>/dev/null | tr '\n' ' '; echo ""
ls tests/ | tr '\n' ' '; echo ""
python3 -m unittest discover -s tests -t . 2>&1 | tail -2
echo ""
echo "== sessions on disk (each row resumable) =="
ls -la "$SB"/fleet-real-*.jsonl 2>/dev/null | awk '{print "  " $NF, $5 " bytes"}'
for i in 1 2 3 4 5; do git worktree remove --force "$FL-wt$i" 2>/dev/null; done
echo "== worktrees cleaned =="
