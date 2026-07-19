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
| Turn loop | `hi-agent` (`run_turn` / `TurnPhase`) | Setup → (Model → Tools → Steer)* → WorkspaceRepair → Settle → Finalize |
| Workspace repair | `hi_agent::verify::WorkspaceRepairVerifier` | compile/lint/test stages; failures feed the model |
| Review repair | `hi_agent::steering::ReviewRepairMode` | answer-quality nudges in Steer (not shell stages) |
| Session memory | `hi_agent::memory` | markdown bullets (`.hi/memory.md`, user global) |
| Runtime | process-local `WorkspaceRuntime` | tools, ledger, LSP, checkpoints |
| Shell sandbox | `hi_tools::sandbox` (`HI_SANDBOX`) | opt-in write confine; see [sandbox.md](sandbox.md) |

This path is what developers run day to day. Verification here is a **workspace
repair gate**, not a cryptographic attestation. CLI RSI hooks stay thin
(`hi-cli` `rsi_bootstrap`) — descriptors, budgets, trace observation only.

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

## Local inference

`hi-local` (+ `hi-local-core` / `hi-cuda` / `hi-mlx` / `hi-gguf`) is an
OpenAI-compatible **sidecar**. The agent talks to it like any other provider;
GPU crates are not linked into `hi-agent`.
