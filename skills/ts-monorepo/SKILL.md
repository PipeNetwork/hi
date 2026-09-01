---
name: ts-monorepo
description: JS/TS package coding loop — package-local tsc and npm test, never repo-wide npm test by default
scope: global
---

# TS / JS monorepo

## When to use
Trees with `package.json` (and often `tsconfig.json`) — apps, packages, or a single Node package at repo root.

## Prerequisites
- `npm` (or the project's package manager) when typecheck/tests are required
- Prefer the scripts the package already defines

## Procedure
1. **Orient** with `repo_map` / `find_symbol` for components, exports, and routes named in the task.
2. **Scope** to the nearest `package.json` directory (e.g. `packages/ui`, `apps/web`).
3. **Edit** with `edit` / `multi_edit`; use `apply_patch` for multi-file API moves.
4. **Mid-turn feedback** (automatic):
   - Typecheck: `npm --prefix <pkg> run typecheck` if a `typecheck` script exists, else `npm --prefix <pkg> exec -- tsc --noEmit` when `tsconfig.json` is present
   - Tests (when task is test-gated): `npm --prefix <pkg> test --silent`
5. **Manual checks**:
   ```bash
   npm --prefix packages/ui run typecheck --silent
   npm --prefix packages/ui test --silent
   ```
   Prefer package scripts (`lint`, `typecheck`, `test`) over ad-hoc global `tsc`/`jest` unless that is what the package uses.
6. **Dependencies**: do not add packages unless asked; match existing import style (ESM/CJS, path aliases).

## Pitfalls
- `npm test` at the monorepo root may run every package — use `--prefix <pkg>` (or `pnpm --filter` / `yarn workspace` if the repo standard is not npm).
- Missing `npm`/node skips mid-turn checks silently; failures are real non-zero exits only.
- Large `write` overwrites of source files are refused — patch in place.

## Verification
Package-local typecheck then test; turn-end verify may skip sealed green `affected-typecheck:` / `affected-test:` stages for packages still clean at the same revision.
