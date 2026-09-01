---
name: dep-audit
description: Audit and patch language-ecosystem dependencies via the real CLI (cargo/npm/pip). Not a connector catalog.
scope: global
---

# Dependency audit

## When to use
The user asks to audit, bump, or patch dependencies; a lockfile changed; or supply-chain review of the current crate/package. Do not run this unattended as a standing upgrade bot.

## Permissions
Stay on the session `/permissions` ladder. Audits are read-mostly (`cargo audit`, `npm audit`, `pip-audit`) and can run in Auto. Applying upgrades, publishing, or `npm audit fix --force` is mutating: keep Ask/Auto confirms; never standing-approve `bash`. Unattended goals must not auto-apply version bumps — park for `/inbox` if a confirm is required.

## Procedure
1. **Detect** the ecosystem from repo markers (`Cargo.lock`, `package-lock.json` / `pnpm-lock.yaml`, `poetry.lock` / `requirements*.txt`). Target one package, not the world.
2. **Audit** with the real CLI via `bash`:
   ```bash
   cargo audit
   npm audit --prefix <pkg>
   pip-audit -r requirements.txt
   ```
   Install the auditor only if missing *and* the user wants that (do not silently `curl | sh`).
3. **Patch** the smallest lockfile/manifest change that clears the advisory. Re-run the same audit. Do not rewrite unrelated pins.
4. **Report** advisory ids, affected packages, and whether a compatible fix exists. Do not open Jira/Slack/GitHub issues unless the user asked.

## Pitfalls
- Workspace-root `cargo update` / `npm audit fix` can churn every member.
- Yanked crates and `>=` ranges are not a license to jump majors.
- Do not add a first-party audit tool; this skill is the recipe.

## Verification
Re-run the same auditor until it is clean or remaining findings are accepted by the user. Then package-local test/check for the touched crate.
