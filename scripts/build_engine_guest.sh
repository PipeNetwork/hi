#!/usr/bin/env bash
# Build the reference decision engine as a Component Model artifact.
# The output is intentionally unsigned; release packaging must sign the
# generated manifest payload with the production Ed25519 key.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT_DIR=${1:-"$ROOT/target/engine"}
MODULE="$OUT_DIR/engine.wasm"
MANIFEST="$OUT_DIR/engine.manifest.json"

command -v wasm-tools >/dev/null 2>&1 || {
  echo "wasm-tools is required; install it with: cargo install wasm-tools" >&2
  exit 2
}

rustup target list --installed | grep -qx 'wasm32-unknown-unknown' || {
  echo "wasm32-unknown-unknown is required; install it with: rustup target add wasm32-unknown-unknown" >&2
  exit 2
}

mkdir -p "$OUT_DIR"
cargo build --manifest-path "$ROOT/Cargo.toml" -p hi-engine-guest \
  --target wasm32-unknown-unknown --release

CORE_MODULE="$ROOT/target/wasm32-unknown-unknown/release/hi_engine_guest.wasm"
[[ -f "$CORE_MODULE" ]] || {
  echo "guest build did not produce $CORE_MODULE" >&2
  exit 1
}

wasm-tools component new "$CORE_MODULE" --output "$MODULE"

if command -v shasum >/dev/null 2>&1; then
  SHA256=$(shasum -a 256 "$MODULE" | awk '{print $1}')
else
  SHA256=$(sha256sum "$MODULE" | awk '{print $1}')
fi
VERSION=${HI_ENGINE_GUEST_VERSION:-0.1.0}
REVISION=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)

python3 - "$MANIFEST" "$SHA256" "$VERSION" "$REVISION" <<'PY'
import json
import pathlib
import sys

manifest = {
    "api_major": 1,
    "api_minor": 0,
    "guest_version": sys.argv[3],
    "state_schema_version": 1,
    "supported_features": [],
    "required_capabilities": [],
    "module_sha256": sys.argv[2],
    "signature_hex": None,
    "build_revision": sys.argv[4] or None,
}
pathlib.Path(sys.argv[1]).write_text(
    json.dumps(manifest, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
PY

echo "built $MODULE"
echo "manifest $MANIFEST"
echo "sha256 $SHA256"
