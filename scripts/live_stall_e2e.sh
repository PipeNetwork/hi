#!/usr/bin/env bash
# Live stall hunt: current hi binary against in-repo unique-file, IRC,
# web-register, large chat-app, and Linux kernel subset fixtures. Builds debug
# hi (not a leftover target/release/hi) and runs the ignored live_complex_app
# cases using the pipenetwork credential already configured for interactive hi
# (~/.config/hi), or PIPENETWORK_API_KEY / HI_API_KEY when those are set.
#
# HI_LIVE=0 opts out. Each live turn is capped at 12 minutes inside the tests
# (20 minutes for the Linux subset). Filter with LIVE_E2E_FILTER (default:
# live_e2e). The Linux cases fetch a pinned v6.6 tarball into target/live-linux
# unless HI_LINUX_SRC points at an existing tree with lib/math/int_sqrt.c.
#
# Exit: 0 pass · 1 fail · 2 skipped (HI_LIVE=0).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

if [[ "${HI_LIVE:-}" == "0" || "${HI_LIVE:-}" == "false" || "${HI_LIVE:-}" == "off" ]]; then
  echo "SKIP: HI_LIVE=0" >&2
  exit 2
fi

FILTER="${LIVE_E2E_FILTER:-live_e2e}"

echo "building current hi (debug)…" >&2
cargo build -p hi >&2 || exit 1

echo "running live e2e filter=${FILTER} using configured hi pipenetwork credential …" >&2
exec cargo test -p hi --test live_complex_app "${FILTER}" -- --ignored --test-threads=1 --nocapture
