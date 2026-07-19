# ADR 001: RSI runtime trust boundary

- Status: accepted
- Date: 2026-07-19

## Decision

The immutable bootstrap is the separately deployed `rsi-hi-worker` in the RSI
control-plane repository. It verifies candidate manifests and revocations,
pins artifacts, owns credentials and leases, constructs isolation, brokers
inference over a worker-owned Unix socket, sequences authoritative backend
events, and invokes the independent evaluator. Candidate packages contain the
`hi` executable and evolvable agent modules; they do not link the bootstrap.

For each process launch, the bootstrap derives an expiring JSON runtime
descriptor from the verified manifest and effective lease. `hi` accepts it only
with `--rsi-managed`, rejects unknown or unsafe fields, verifies that its
effective role/tool/budget settings do not exceed it, and binds the descriptor,
run, task, candidate, manifest, executable, and repository snapshot hashes into
the managed trace. The bootstrap independently checks that provenance before
uploading evidence.

Local bootstrap/candidate messages use bounded Serde JSON contracts. Inference
continues over authenticated HTTP on a Unix socket; future brokered tool
messages will use length-prefixed JSON on a separate Unix socket. Protocol
major and schema versions fail closed.

The initial DigitalOcean worker profile is the approved strict namespace
equivalent because nested KVM is unavailable: dedicated Unix identities, user,
mount, PID, and network namespaces, cgroups v2, seccomp, a read-only base,
disposable worktree, no device passthrough, default-deny egress, and forced
process-tree teardown. Source candidates additionally require no network
allowlist. The descriptor records this as `namespace`; a candidate cannot
downgrade it. Firecracker remains an optional stronger profile on KVM hosts.

Repository intelligence caches are tenant/repository scoped and keyed by the
repository tree hash, Rust toolchain, Cargo metadata version, and analyzer
version. Compiler diagnostics may be cached; source bodies and semantic chunks
follow the task evidence policy.

The default Rust verification matrix is workspace integrity, formatting,
required-feature `cargo check`, targeted and affected-crate tests, workspace
tests when budget allows, warnings-denied Clippy, secret/security scans, and an
opaque evaluator entry point. Only the supervisor/evaluator can attach an
attestation.

Interactive local use does not upload full traces implicitly. Managed RSI runs
always retain the required operational evidence; remote RSI activation provides
the explicit upload and retention notice. Human and pull-request outcomes are
associated by the public run ID and remain supporting evidence rather than
promotion authority.

## Consequences

The candidate can vary without recompiling or granting credentials to the
bootstrap. Candidate-authored trace content remains evidence, not a trusted
verdict; tool, accounting, and verification authority stays outside the
candidate process. Physical compromise of the one droplet remains a shared
host trust-domain limitation and is handled operationally with short-lived
credentials, separate identities, signed artifacts, and reproducibility.

## See also

- [Architecture: interactive agent vs RSI control plane](../architecture.md) — crate map and naming (`RepairVerifier` vs `AttestingVerifier`, session memory vs `RsiMemoryStore`).
