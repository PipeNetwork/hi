---
name: rust-workspace
description: Cargo workspace coding loop — target one crate, never cargo test the world, use mid-turn seals
scope: global
---

# Rust workspace

## When to use
Editing a Cargo workspace (multi-crate `Cargo.toml` with `[workspace]`), or any Rust package under `crates/`.

## Prerequisites
- `cargo` on PATH
- Prefer package-local checks; full workspace suites only at turn-end verify if configured

## Procedure
1. **Orient** with `repo_map` / `find_symbol` (not blind `list`/`grep`) for the task identifiers.
2. **Scope** the change to the owning crate (`crates/<name>/`). Note the package name from its `Cargo.toml`.
3. **Edit** with `edit` / `multi_edit` for single-file hunks; `apply_patch` only for multi-file coordination. Do not `write`-overwrite large sources.
4. **Mid-turn feedback** (automatic): after mutations, hi runs LSP diagnostics then `cargo check --quiet --manifest-path crates/<name>/Cargo.toml`. When the task mentions tests, it may also run package-local `cargo test --quiet --manifest-path …`.
5. **Do not** run `cargo test` or `cargo check` at the workspace root unless you intentionally need every member. Prefer:
   ```bash
   cargo check --quiet --manifest-path crates/<name>/Cargo.toml
   cargo test --quiet --manifest-path crates/<name>/Cargo.toml
   # or from workspace root when metadata is clear:
   cargo test -p <package-name> --quiet
   ```
6. **Finish** only after green verify (or a clear reason verify does not apply). Failed verify output is structured — fix `file:line` causes first.

## Pitfalls
- `cargo test` with no `-p` / `--manifest-path` on a large monorepo burns minutes and floods context.
- Editing `workspace.dependencies` / root `Cargo.toml` can invalidate many crates — expect broader checks.
- Mid-turn seals skip re-running the same package check at turn-end only if nothing else mutated that package.

## Verification
Package-local `cargo check` then `cargo test` for the touched crate; rely on hi WorkspaceRepair for the configured pipeline after tools stop.
