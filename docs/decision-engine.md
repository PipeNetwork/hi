# Hot-swappable decision engine

`hi` has a native decision path and an optional Wasmtime Component Model
decision engine. The native substrate remains authoritative for providers,
tools, approvals, checkpoints, persistence, workspace mutations, cancellation,
and terminal rendering. A guest only receives bounded protocol events and
returns host-validated actions; the engine WIT world intentionally has no host
imports.

## Rollout defaults

Native is the default until deterministic replay parity and soak gates have
passed. Select the experimental engine explicitly with either:

```text
HI_ENGINE_MODE=wasm HI_ENGINE_MODULE=/path/to/engine.wasm hi
```

or in an interactive session:

```text
/engine status
/engine wasm /path/to/engine.wasm
/engine reload
/engine watch on
/engine native
```

`/config engine ...` is an alias for the same control surface. `HI_ENGINE_MODE=native`
is an emergency bypass. Development modules must opt into unsigned loading
with `HI_ENGINE_ALLOW_UNSIGNED=1`; production modules require a trusted
Ed25519 public key in `HI_ENGINE_TRUSTED_KEYS` (comma-separated hexadecimal
keys).

## Building a local component

The guest crate is deliberately small and does not receive WASI capabilities.
Install the target and `wasm-tools`, then run:

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-tools
scripts/build_engine_guest.sh
HI_ENGINE_MODE=wasm HI_ENGINE_ALLOW_UNSIGNED=1 \
  HI_ENGINE_MODULE="$PWD/target/engine/engine.wasm" hi
```

The script emits `engine.wasm` and a sibling `engine.manifest.json`. The
manifest contains the API major/minor, state schema, module hash, and build
metadata. It is unsigned for local development; release packaging must sign
the manifest payload after the hash is generated.

## Lifecycle guarantees

Reload compiles and validates a candidate in isolation, rejects unsupported
API/state versions, mismatched hashes, untrusted signatures, imports, missing
exports, and oversized modules, then puts the candidate in a pending slot.
Each turn is pinned to the active generation. A pending generation activates
only after all turns using the old generation finish. The optional watcher is
off by default and debounces module/manifest replacement.

Fuel, guest memory, a two-second default guest-step deadline, action payloads,
input payloads, and action counts are bounded. Every host action carries an
idempotency key. A guest trap does not
replay an already-issued effect; the host retains the previous known-good
generation and reports one concise failure status.

The current migration seam invokes the guest at turn start for protocol and
lifecycle validation while the existing native turn loop continues to own
provider/tool orchestration. Moving additional decisions behind the broker is
gated on differential replay tests, so enabling module loading cannot silently
change ordinary turns.

The stable protocol types live in
[`crates/hi-engine-api/src/lib.rs`](../crates/hi-engine-api/src/lib.rs), the
Wasmtime lifecycle and validation in
[`crates/hi-engine-host/src/lib.rs`](../crates/hi-engine-host/src/lib.rs), and
the reference guest in
[`crates/hi-engine-guest/src/lib.rs`](../crates/hi-engine-guest/src/lib.rs).
