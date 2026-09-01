# Hardening Plan: unsafe policy, async-blocking audit, symlink-check consolidation

Three independent workstreams from the codebase review. Each is self-contained and
lands with its own verification. The core workspace now owns workstreams A–C;
the backend-specific exception and GGUF split belong in the separately released
[`hi-local-runtime`](https://github.com/PipeNetwork/hi-local-runtime) workspace.
Order matters only for the GGUF split (do it last, after that tree is otherwise
green, to keep the diff reviewable).

---

## Workstream A — Workspace-wide unsafe policy

**Goal:** prevent `unsafe` from creeping into core crates that don't need it.
The extracted runtime workspace owns the legitimate mmap/deallocation exceptions
for its backend crates; the core workspace is the target of this guardrail.

**Approach:** workspace lint table + explicit per-crate opt-out, rather than
touching 20 `lib.rs` files.

1. `Cargo.toml` (workspace root) — add:
   ```toml
   [workspace.lints.rust]
   unsafe_code = "deny"
   ```
2. Each crate `Cargo.toml` — add `[lints] workspace = true`.
   - Apply it to the core workspace crates that do not intentionally use unsafe.
   - The runtime workspace must keep its own explicit exceptions documented in
     its crate manifests; do not add those packages back to core just to satisfy
     this historical hardening checklist.
   - Confirm there is no existing `[lints]` table to merge with before appending.
3. Note: workspace `[lints]` apply to all targets including tests, so the test
   `unsafe` blocks in `hi-eval/src/artifacts.rs` (the `set_var` calls) will need a
   targeted `#[allow(unsafe_code)]` on those test fns, or keep them and add the
   allow. Check `hi-eval/artifacts.rs:66,78`.

**Verify:** `cargo check --workspace` and `cargo test -p hi-eval` (the crate with
test-only unsafe). A clean build is the pass gate.

**Risk:** low. Only failure mode is a test/helper `unsafe` we didn't spot; the
compiler will name it.

---

## Workstream B — Async-blocking process audit (narrowed)

**Original finding narrowed by inspection.** The grep hits split into three
buckets; only bucket 1 is a real defect.

1. **Genuine async-context blocking (fix these).** Verify each is reached from an
   async task and convert to `tokio::process::Command` (or wrap in
   `tokio::task::spawn_blocking` if the call is inherently sync):
   - `hi-tools/src/checkpoint.rs` (many git calls — check whether `checkpoint`
     runs on the agent's async turn loop).
   - `hi-tools/src/tools/mod.rs` git calls (194, 400–476).
   - `hi-cli/src/candidate_merge.rs` — the `Command::new("git")` calls are in
     **sync** fns (`candidate_transaction_mode`, `apply_patch_to_scratch`,
     `tree`), so only fix if their caller is async; trace the call path first.

2. **Acceptable / leave alone (document, don't change):**
   - `hi-eval/src/skeptic_detector.rs` `mine()` and `runner.rs`/`agent_path.rs` —
     sync fns driving a batch CLI; blocking is fine off the runtime.
   - `hi-agent/src/local_skeptic.rs:62` `nvidia-smi -L` — one-shot startup probe.
   - `hi-tui/src/loops.rs:2395` — **test helper** (`fn git(...)` inside tests).
   - `hi-bootstrap`, `hi-candidate` — sync `main` binaries.

3. **Outcome:** a short list of conversions (likely just hi-tools git paths if
   they're async-reachable), plus a one-line comment on each intentionally-sync
   site so the next audit doesn't re-flag them.

**Verify:** for each converted site, run the owning crate's tests
(`cargo test -p hi-tools`, `-p hi-cli`). No conversions → the deliverable is the
annotated audit + comments instead.

**Risk:** low-medium. Converting sync→async git calls can change error/output
capture semantics; keep `.output()` capture identical.

---

## Workstream C — Consolidate symlink-escape checks in hi-eval

**Goal:** `hi-eval/src/artifacts.rs` repeats the same "reject a symlink that
escapes its root" logic at **lines 220, 530, 966** (oracle / candidate / source),
each with its own `symlink_metadata` + `file_type().is_symlink()` + context
string. Three copies drift.

1. Extract one helper, e.g.
   `fn reject_escaping_symlink(path: &Path, root: &Path, kind: &str) -> Result<()>`
   that: takes `symlink_metadata`, if it's a symlink canonicalizes and confirms the
   target stays under `root`, else bails with a `"unsafe {kind} symlink {path}"`
   message (preserving the existing wording so tests still match).
2. Replace the three call sites with the helper.
3. Confirm the canonicalization semantics match what each site currently does —
   the oracle/candidate/source checks may differ subtly (some only check
   `is_symlink()` and bail outright rather than resolving the target). **Preserve
   per-site strictness**; if one site must reject *any* symlink while another
   resolves-and-checks, keep two helpers or a `mode` flag rather than weakening a
   check.

**Verify:** `cargo test -p hi-eval` — the existing tests at `artifacts.rs:1211-1270`
(`unsafe symlink` escape cases) must still pass and still produce the same error
messages.

**Risk:** low. Pure refactor; the safety property is covered by existing tests.
Main care is not loosening any site's policy.

---

## Workstream D — Split the runtime GGUF crate (do last)

**Goal:** In `hi-local-runtime/crates/hi-gguf`, split the 14,404-line single file
(parser + tokenizer + Qwen weight-name tables +
dequantizers + tests). Split along existing item boundaries; **no logic changes**,
pure `mod` extraction with `pub use` re-exports so the public API is unchanged.

Natural seams (from the item map):
- `tokenizer.rs` — `GgufTokenizer` (1113) + summary (1763) + BPE/merge logic
  (~1113–1775, plus the streaming byte-fallback decoders near 7700–7870). ~2–3k lines.
- `weights.rs` (or `qwen_names.rs`) — the large block of `qwen_*_weight_names` /
  `_bias_names` fns (2375–4068+). ~2k lines, mechanical to move.
- `dequantize.rs` — `GgufTensorType` (666) + `dequantize_tensor_as_f32` (1063) +
  the per-format `dequantize_*` / `read_*_tensor` fns (6299–6800+). ~1.5k lines.
- `lib.rs` retains: `GgufFile`, mmap/shard handling, `GgufDirectReader` +
  `AlignedBlockBuf` (the unsafe), `MetadataValue`, `TensorInfo`, parsing.
- Tests: the big `#[cfg(test)] mod tests` (7926–14404, ~6.5k lines) can move to
  `src/tests.rs` or stay split per-module. Moving to a top-level `tests.rs` keeps
  `lib.rs` small.

**Method:** in the external runtime repository, move one region at a time, `cargo check -p hi-gguf` after each, add
`pub use` in `lib.rs` to keep `hi-gguf::X` paths stable for the ~5 dependent
crates. Update the line-count ratchet baseline after the split (swap the parent
path for any still-oversized children).

**Verify:** from `hi-local-runtime`, run `cargo test -p hi-gguf` (the crate is
heavily tested) plus `cargo check` on its local runtime dependents.

**Risk:** medium. Zero logic change, but high churn and the tokenizer/dequantize
code interleaves — boundaries must be drawn where item dependencies allow. Do it
independently verifiable, one module per commit.

---

## Sequencing & effort

| WS | Effort | Churn | Gate |
|----|--------|-------|------|
| A unsafe lint | S | ~20 Cargo.toml + root | `cargo check --workspace` |
| B async audit | S–M | a few files | per-crate tests |
| C symlink helper | S | 1 file | `cargo test -p hi-eval` |
| D gguf split | L | 1 crate | `cargo test -p hi-gguf` + dependents |

A → C → B → D is a sensible order (small/green first, big refactor last, on a
clean tree). A, B, C are independent and could land in any order.
