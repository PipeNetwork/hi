#!/usr/bin/env bash
# Live stall hunt: current hi binary against a copy of ~/chat.
# Builds debug hi (not a leftover target/release/hi) and runs the ignored
# live_complex_app cases with the operator's configured provider.
#
# Requires HI_LIVE=1 and a configured provider (typically pipenetwork in
# ~/.config/hi). Each live turn is capped at 8 minutes inside the tests.
#
# Exit: 0 pass · 1 fail · 2 skipped (HI_LIVE unset).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

if [[ "${HI_LIVE:-}" != "1" ]]; then
  echo "SKIP: set HI_LIVE=1 to run live stall e2e" >&2
  exit 2
fi

echo "building current hi (debug)…" >&2
cargo build -p hi >&2 || exit 1

exec cargo test -p hi --test live_complex_app -- --ignored --test-threads=1 --nocapture
