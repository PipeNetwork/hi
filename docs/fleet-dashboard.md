# `/dashboard` — concurrent agents

Aliases: `/fleet`, `/agents-dashboard`, `Ctrl+\`.

A full-screen TUI roster for dispatching and steering several **independent**
coding sessions from one screen. You are the **manager** (the hi session you
opened the dashboard from). Each row is a **sub-agent**: an in-process
`hi-harness` turn loop, not a child `exec hi`, and not a Luna/`spawn_subagent`
tree.

```
  role        model                             where
  manager     pipe/deepseek-v4-flash-0731       this session · Esc
  sub-agent   pipe/gpt-6                        next dispatch · Ctrl+M    + New sub-agent

  #  role        model                             state       task
  1⠋  sub-agent   pipe/gpt-6                        working     fix login
  2○  sub-agent   pipe/deepseek-v4-flash-0731       idle        review tests

  ╭ dispatch a sub-agent ─────────────────────────────────────────────╮
  │❯                                                                  │
  ╰ you are the manager · type a prompt · Enter starts a sub-agent ───╯
  enter:sub-agent  │  ctrl+s:open  │  ctrl+m:sub model  │  ?:help
```

## How to set manager vs sub-agent

There is no picker. **You are the manager.** Change the manager model with
`/model` in the session, then reopen `/dashboard`.

Each Enter on `+ New sub-agent` creates a new independent session. `Ctrl+M`
sets the **next** sub-agent's model (`pipe/gpt-6` on Pipe Network, or the
manager's model). Prefix a prompt with `/model pipe/gpt-6 …` for one dispatch.
Rows do not call each other; you peek/reply to feed them work.

Optional OpenAI: an `openai` profile in config is used only for `openai/…`
model ids. Missing `OPENAI_API_KEY` is not an error — gpt-6 stays on Pipe.

## Isolation

`Ctrl+W` (git repos only) makes the **next** dispatch a `git worktree`. There
is **no auto-merge**. Closing the dashboard does not kill workers or delete
worktrees; `Ctrl+X` twice on a row removes that row (and its worktree).

Dashboard JSONL lives under
`~/.local/share/hi/projects/<workspace>/dashboard/` so it does not show up in
`/sessions`.

## Keys

| key | does |
|---|---|
| `Enter` (dispatch box) | start a **new** sub-agent |
| `Ctrl+S` | dispatch/reply **and** attach |
| `↑`/`↓` | select a row (the box becomes reply) |
| empty `Enter` on a row | attach (details view) |
| typed `Enter` on a row | reply now, or FIFO-queue if that row is working |
| `Tab` | list ↔ dispatch/peek input |
| `Ctrl+M` | next sub-agent: `pipe/gpt-6` ↔ manager model |
| `Ctrl+W` | next dispatch uses a git worktree |
| `Ctrl+X` | cancel the selected turn; twice in 2s deletes the row |
| `Ctrl+C` | cancel the selected row, or close the dashboard |
| `?` | cheat sheet |
| `Esc` / `Ctrl+\` | step back; close hides the roster (workers keep running) |

`HI_DASHBOARD_MAX_WORKING` (default 8) caps concurrent working rows.
