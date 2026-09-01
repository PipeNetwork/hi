# ADR 002: Interactive tool admission bar

- Status: accepted
- Date: 2026-07-29

## Context

The interactive agent reaches the world through a small structured tool catalog
(`hi-tools`) on top of ordinary developer surfaces: files, shell, git-shaped
diffs, HTTP fetch/search, and real language toolchains invoked via `bash`.

Models are trained on those human protocols. Every first-class tool adds JSON
Schema tokens, decision load, and another surface that can rot when the real CLI
moves. Empirically, large advertised sets degrade small-model tool calling
(latency and malformed calls). Growing a parallel “agent protocol” or plugin
runtime for capabilities that already exist as CLIs fights the product bet:

> Richer capabilities come from subprocess CLI tools the model invokes via
> `bash` rather than a plugin runtime.

MCP, subagent coordination, and the RSI tool-host are optional or separate trust
domains. They must not become the default way to wrap everyday developer work.

## Decision

**Human protocol first.** Before adding a built-in tool, prefer:

1. **`bash` + a real CLI** the user already has, and/or
2. a **skill** (`skills/*/SKILL.md` or user/project skills) that teaches the recipe.

Admit a **new first-class tool** only when at least one holds:

| Gate | Meaning |
|------|---------|
| **Structure** | The transcript needs parseable, stable results—not free-form CLI noise. |
| **Safety** | Confirmations, sandbox class, or side-effect labeling the shell path cannot express cleanly. |
| **Reliability** | Materially better than the human path on real turns (e.g. unique-hunk `edit` vs `sed`/heredoc). |

### Advertisement (schema tax is part of admission)

New tools default to **capability-gated or injected** advertisement (same pattern
as `explore` / `delegate` / `task` / `use_tool` / `skill`), not unconditional
membership in the always-on global set.

| Tier | When |
|------|------|
| Inject / feature-gated | Default for new families |
| Global `TOOL_SPECS` | Almost every coding turn needs it |
| `MINIMAL_TOOL_SPECS` | Only if weak models must keep the coding loop without the full set |
| `PROTECTED_TOOLS` | Almost never for new names—floor is the core workspace loop only |

Census trim (`hi tools trim`) remains a **post-hoc, human-gated** diet. It is not
a substitute for pre-merge admission.

### What a tool change must include

1. **Admission note** (PR body or comment above the spec): human alternative
   considered; which gate (structure / safety / reliability) fails for that
   alternative; advertisement tier; minimal-set yes/no; side-effect class.
2. **Mechanical checklist**
   - `ToolSpec` in `catalog.rs` (or inject helper) — description reads like a
     man page and says when *not* to use `bash`
   - `TOOL_CATALOG` metadata row
   - dispatch arm in `tools/mod.rs`
   - side-effect pin in catalog tests (no silent additions)
   - protected / conditional / inject decision

Thin wrappers around `cargo`, `git`, `gh`, `docker`, etc. are **skills or bash**,
not tools.

### Worked examples

| Proposal | Verdict |
|----------|---------|
| `cargo_test` / `gh_pr` / `docker_ps` | **Reject** → skill or `bash` |
| `edit` / `apply_patch` | **Admit** — reliability + confirmable mutations |
| `bash` | **Admit** + protect — human-protocol escape hatch |
| `repo_map` / `find_symbol` | **Admit** only while structure/speed beats `list`/`grep`; keep dynamic |
| `memory_search` / `get` / `update` / `forget` | **Admit** inject-only — markdown `.hi/memory.md` / `~/.config/hi/memory.md` with stable `[#n]` bullets. Not RSI SQLite. Off with `--no-memory`. |
| `browser_click` for coding-core | **Reject** — new protocol surface; not the coding job |
| `browser_exec` | **Admit** inject/feature-gated — Safety + Reliability vs `bash`+curl for page/login/live-UI. On by default; `[browser] enabled = false` hides it. Advertised only on matching tasks. Never global/`MINIMAL`/`PROTECTED`. Side-effect: network. Cloud metadata and link-local stay blocked even with `allow_private_urls`. In Ask/Auto, `browser_exec` and `use_tool` go through `ConfirmationRequest::External` (never standing-approve browser; MCP may grant server+tool for the session). |
| `delegate` / `explore` / `task` | **Admit** inject — isolation/verify are not one shell command |

### Non-goals

- Not a ban on provider function calling (structured tool calls remain the
  envelope that makes FS/shell safe and parseable).
- Not a redesign of RSI `hi-protocol` / `hi-tool-host` (separate trust domain;
  see [ADR 001](001-rsi-runtime-boundary.md)).
- Not automatic catalog deletion or auto-trim.
- Not requiring an RFC for every tool—only the admission note + checklist.

## Consequences

- Default growth path is skills and bash recipes; the advertised set stays small.
- Reviewers and catalog pin tests share one written bar.
- Existing tools are grandfathered until intentionally reclassified; the bar
  applies to **new names** and **promotions** (skill → tool, inject → global,
  normal → protected).
- Catalog teeth: every `TOOL_CATALOG` row carries `ToolAdmission` +
  `alternative`. The current set is fully classified under
  Structure/Safety/Reliability (there is no `Legacy` variant), and tests reject
  empty alternatives so new tools must pick a real gate (or stay bash/skill).
- First-party Pipe MCP is inject-only via the same two gateway schemas. New
  Pipe tools must be explicitly allowlisted for the agent; nested model-call
  tools stay code-denied.

## See also

- [Architecture: interactive agent vs RSI control plane](../architecture.md)
- [Shell sandbox](../sandbox.md)
- `crates/hi-tools/src/lib.rs` — bash-over-plugins crate policy
- `crates/hi-tools/src/catalog.rs` — `TOOL_SPECS`, `PROTECTED_TOOLS`, side-effect pins
- `crates/hi-cli/src/tool_trim.rs` — human-gated census trim
- `crates/hi-agent/src/skills.rs` — skills as the reject path for thin workflows
