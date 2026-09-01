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

### Trace trust boundary: local tamper-evidence, not anchored identity

`hi-trace` gives **local tamper-evidence only**. The event hash chain and the
per-event `trace_id == manifest.trace_id` binding detect corruption, reorder,
and foreign-journal splices *of the files on disk*. Self-hosted runs add a
real but **local-only** signature: `LocalAttestor` signs the terminal
`root_hash` with an ed25519 key persisted on the same machine
(`$XDG_STATE_HOME/hi/trace-signing-key`, owner-only), emitting
`local-signed:<hex-sig>`. That proves the trace has not been modified since
signing — but the key is readable by the same principal that wrote the trace,
so it is still **not** external authenticity. A worker-anchored signature over
the trace, made with a key the candidate cannot read, remains the worker's job
and lives outside this repo.

External anchoring uses the `Attestor` seam in two places:
`hi_verifier::Attestor` (report-level: `AttestingVerifier` hashes each report
and calls `attestor.attest(hash)`) and `hi_trace::TraceAttestor` (trace-level:
`TraceWriter::with_attestor` signs the terminal `root_hash` at finalize,
recorded in the manifest's `attestation` field). A managed deployment supplies
an implementation that binds the evidence to the signed control-plane manifest
the worker already verified. The only in-repo impls are test stubs and
`LocalAttestor`, whose `local-signed:` label marks self-hosted output as not
worker-attested evidence.

Two rules follow:

- Treat a passing `validate_trace` as "this local trace is internally
  consistent," never as "this trace is authentic." A passing local-signature
  check adds "unmodified since signing on this machine" — still not external
  authenticity, which requires the worker to have recorded the trace root
  out-of-band or signed it via a production `Attestor`.
- Any code that consumes a managed trace for a trust decision must require a
  worker-anchored attestation, not just a valid chain or a local signature.

The `hi trace` CLI surfaces this boundary. `hi trace list` shows recent runs
with an `INTEGRITY` column (`ok`/`TAMPERED`, from `validate_trace`) and an
`ATTESTATION` column (the label scheme: `local-signed`, `unattested`, or
a worker scheme), so tampered or unattested runs are visible at a glance.
`hi trace show [id]` prints one run's detail with the integrity status inline,
and `hi trace verify [id]` runs the integrity gate and, for `local-signed`
traces, validates the ed25519 signature against the local key (reporting
`signature: ok` / `MISMATCH` / `unverifiable`). None of these establish
authenticity — they report local consistency, the local signature, and the
attestation label.

The workflow side mirrors this. `hi workflow run` attests each verification
report through the `hi_verifier::Attestor` seam; the self-hosted
`LocalAttestor` signs the report hash with the **same** local ed25519 key
(`$XDG_STATE_HOME/hi/trace-signing-key`, fallback `$HOME/.local/state/hi/`),
so a self-hosted report is tamper-evident but not worker-attested. The final
signed report is persisted to `<state_root>/workflow/<plan>-<hash>/report.json`,
and `hi workflow verify [report.json]` recomputes the unsigned report hash and
validates the signature — resolving the latest persisted report when no path
is given, and failing hard on a forged or tampered signature (the signature is
the report's only integrity mechanism; there is no hash chain to fall back on).

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

`hi-local` is an OpenAI-compatible **sidecar** released from the separate
[`hi-local-runtime`](https://github.com/PipeNetwork/hi-local-runtime) repository.
The agent talks to it like any other provider; GPU crates are not linked into
`hi-agent` or the core workspace.

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
