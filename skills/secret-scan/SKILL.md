---
name: secret-scan
description: Find likely secrets in the working tree or diff before commit. Never print secret values. Not a connector catalog.
scope: global
---

# Secret scan

## When to use
Before `/commit`, after adding env/config/auth code, or when the user asks to check for leaked keys. Not a substitute for a hosted scanner.

## Permissions
Stay on the session `/permissions` ladder. Do not standing-approve `bash`. If the scan would post findings outside this repo, or you need a broader crawl than the current diff, switch to `/permissions ask` (or wait for the confirm overlay in Ask/Auto).

## Procedure
1. **Scope** to the change: `git diff` / `git diff --cached` first. Full-tree grep only when the user asks.
2. **Search** with `grep`/`glob` for high-signal shapes (API key assignments, `BEGIN PRIVATE KEY`, `.env` bodies, cloud access tokens). Prefer existing tools over a new scanner CLI.
3. **Do not** print secret values, paste them into chat, write them to memory, or send them through `browser_exec` / MCP / research. Cite `path:line` and the *kind* of secret only.
4. **Fix** by rotating the credential out-of-band (user), removing it from the tree, and adding the path to `.gitignore` when it is a local env file. Do not invent a first-party Slack/Jira/Gmail notifier.

## Pitfalls
- Echoing a match is exfiltration. Truncate.
- `.env.example` placeholders are not findings.
- Do not `git add` files that still contain live secrets.

## Verification
Re-run the scoped grep on the same paths after the fix. The diff should no longer contain the flagged assignment.
