---
name: pytest-package
description: Python package coding loop — pytest on the package root, ruff when configured, no full-tree thrash
scope: global
---

# Pytest package

## When to use
Python trees with `pyproject.toml`, `setup.py`/`setup.cfg`, `pytest.ini`, or `tox.ini` as the package root.

## Prerequisites
- `python3` / `pytest` on PATH when tests are required
- Optional: `ruff` when `ruff.toml` / `[tool.ruff]` is present

## Procedure
1. **Orient** with `repo_map` / `find_symbol` for the module or test names in the task.
2. **Find the package root** (nearest directory with pyproject/setup/pytest.ini) — not necessarily the repo root.
3. **Edit** sources under that package with `edit` / `multi_edit`. Keep tests next to code or under the package's test layout the project already uses.
4. **Mid-turn feedback** (automatic): hi may run `ruff check <pkg>` when ruff is configured, and when the task is test-gated run `pytest -q <pkg>` on affected packages only.
5. **Manual checks** when you need them:
   ```bash
   pytest -q <package_or_path>
   # optional lint
   ruff check <package_or_path>
   ```
   Avoid `pytest` with no path on a huge monorepo unless that is the project convention.
6. **Imports**: match existing layout (`src/`, flat package, `PYTHONPATH=.`). Prefer editing the failing module the oracle/import path already uses.

## Pitfalls
- Running pytest from the wrong cwd loses imports — stay at the workspace root and pass the package path.
- Rewriting whole modules with `write` on large files is rejected; use `edit`.
- Mid-turn pytest is skipped silently if `pytest` is not installed; install or run via the project's venv if needed.

## Verification
`pytest -q` on the affected package path; fix assertion failures from structured output (`file:line`) before claiming done.
