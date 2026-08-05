# Architecture: interactive agent vs RSI control plane

`hi` is one product with two trust domains. They share goals (verify work,
bound cost, record evidence) but **must not be conflated in code or docs**.

## Interactive path (default `hi` CLI)

```
hi-cli → hi-agent → hi-ai (providers)
                 → hi-tools (+ hi-lsp)
                 → hi-tui
```

| Concern | Crate / type | Role |
|--------|---------------|------|
| Turn loop | `hi-agent` (`run_turn` / `TurnPhase`) | Setup → (Model → Tools → Steer)* → WorkspaceRepair → Settle → Finalize → Done |
| Workspace repair | `hi_agent::verify::WorkspaceRepairVerifier` | compile/lint/test stages; failures feed the model |
| Review repair | `hi_agent::steering::ReviewRepairMode` | answer-quality nudges in Steer (not shell stages) |
| Session memory | `hi_agent::memory` | markdown bullets (`.hi/memory.md`, user global) |
| Runtime | process-local `WorkspaceRuntime` | tools, ledger, LSP, checkpoints |
| Shell sandbox | `hi_tools::sandbox` (`HI_SANDBOX`) | default workspace write confine (`off` to disable); see [sandbox.md](sandbox.md) |

This path is what developers run day to day. Verification here is a **workspace
repair gate**, not a cryptographic attestation. CLI RSI hooks stay thin
(`hi-cli` `rsi_bootstrap`) — descriptors, budgets, trace observation only.

Built-in tools stay a thin remote control over human developer surfaces (files,
shell, real CLIs). Adding to the catalog follows
[ADR 002: tool admission](adr/002-tool-admission.md).

## RSI control plane (managed / supervisor)

See [ADR 001](adr/001-rsi-runtime-boundary.md). The bootstrap worker lives
outside this repo; candidate `hi` accepts a managed descriptor only under
`--rsi-managed`.

```
hi-rsi-runtime          shared budget, identity, report types
├── hi-agent-runtime    WorkflowExecutor / trusted stage driver
├── hi-verifier         AttestingVerifier + Attestor
├── hi-memory           RsiMemoryStore (SQLite, tenant-scoped)
├── hi-protocol         wire contracts
└── hi-replay           replay over the runtime
```

| Concern | Crate / type | Role |
|--------|---------------|------|
| Attested verification | `hi_verifier::AttestingVerifier` | hashed `VerificationReport`; supervisor attests |
| Durable memory | `hi_memory::RsiMemoryStore` | candidate hypotheses vs supervisor-verified entries |
| Workflow | `hi_agent_runtime::WorkflowExecutor` | budgeted stage machine, not the interactive loop |

`hi-cli` depends on `hi-rsi-runtime` for managed descriptors, shared budgets,
and trace observation only. It does **not** drive `WorkflowExecutor` or
`AttestingVerifier` on the interactive path.

## Naming rule

Prefer the disambiguated names in new code and docs:

- `WorkspaceRepairVerifier` (alias: `RepairVerifier`) for turn-loop compile/test repair
- `ReviewRepairMode` / `ReviewRepairState` for read-only answer-quality repair
- `AttestingVerifier` when you mean RSI attestation
- `RsiMemoryStore` when you mean control-plane SQLite memory
- “session memory” when you mean markdown `hi_agent::memory`

Historical type aliases (`RepairVerifier`, `hi_verifier::Verifier`,
`hi_memory::MemoryStore`) remain for compatibility.



## Ownership boundary (do not merge the two SMs)

| Path | State machine | Owner crate | Trust domain |
|------|---------------|-------------|--------------|
| Interactive coding | `hi_agent::TurnPhase` / `run_turn` | `hi-agent` (+ `hi-tools`) | User workstation; undo/checkpoint; workspace repair |
| RSI managed/candidate | `hi_agent_runtime::WorkflowExecutor` | `hi-rsi-runtime` types + runtime/verifier | Bootstrap-attested; budgets; attestation |

**Keep both.** They share vocabulary (verify, checkpoint, budget) but not authority:
merging them would either weaken RSI attestation or over-constrain the REPL.

Interactive code may *observe* RSI (`RsiControl`, managed descriptor, trace sinks)
but must not call `WorkflowExecutor` or `AttestingVerifier`. RSI candidate code must
not depend on `hi-agent`'s turn loop.

See [ADR 001](adr/001-rsi-runtime-boundary.md).

## Local inference

`hi-local` (+ `hi-local-core` / `hi-cuda` / `hi-mlx` / `hi-gguf`) is an
OpenAI-compatible **sidecar**. The agent talks to it like any other provider;
GPU crates are not linked into `hi-agent`.

## Benchmark evidence boundary

Harness diagnostics reuse the existing `hi-agent` turn report, change ledger,
checkpoint store, and `hi-trace` observer; they do not create a parallel
telemetry or artifact system. Reports remain additive `schema_version: 2`.

`failure_mode` identifies where a run stopped while `FailKind` remains the
quality bucket. Provider policy blocks retain provider code and HTTP status and
are neither compatibility fallbacks nor circuit-breaker health failures.
Provider refusal signals are authoritative; ordinary plain-text refusal-like
language is not classified heuristically.

Each concrete request attempt can emit a bounded wire audit covering route,
model, token parameter, sampling/reasoning fields, tools, strict schema,
tool-choice, compatibility fallback, and accepted/rejected status.
Payload-changing retries receive a new request identity, and `auto` probing is
limited to an explicit unsupported output-token-field error.

Reasoning remains provider-neutral in the transcript, with explicit requested,
received, replayed, signed-replay, and fallback telemetry. Tool channels are
reported as `native`, `text_fallback`, `mixed`, or `none`; DeepSeek reasoning
content and Anthropic signed thinking blocks stay provider-specific only at the
adapter boundary.

Partial artifacts use atomic writes, existing checkpoints, and content-addressed
evidence around mutation, verification, provider failure, cancellation,
completion, and rollback. Full local traces are enabled for evaluation and
explicit diagnostics; normal metadata traces do not persist raw payloads.
Nothing uploads a local trace implicitly or auto-commits intermediate edits.
