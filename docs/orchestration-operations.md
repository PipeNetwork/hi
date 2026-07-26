# Orchestration operations

## Scheduler presets

Set `HI_SCHEDULER_PRESET` to `conservative`, `balanced` (default), or
`throughput`. Conservative mode disables adaptive scheduling and warm workers;
individual concurrency variables still provide explicit caps.

Rollback switches:

- `HI_ADAPTIVE_SCHEDULER=0`
- `HI_WARM_WORKERS=0`
- `HI_PARALLEL_DELEGATES=1`

Run `hi doctor` to inspect configuration and `hi metrics` for local p50/p95
orchestration latency. `hi workflow run plan.md` drives a markdown plan of
objectives through the workflow engine (see `docs/plan-workflows.md`).
Resolve-rate benchmarking and the evidence-driven tuning loop are covered in
`docs/benchmarking.md`. `hi --benchmark-orchestration` runs the deterministic
microbenchmark. CI can set `HI_BENCH_BASELINE_{1,2,4,8}_MS` and
`HI_BENCH_MAX_REGRESSION_PERCENT`.

## Recovery and cancellation

Startup removes scheduler and verification-flight artifacts older than one
hour. Live artifacts are protected by owner checks and RAII cleanup. Capacity
waits longer than two seconds report whether they are blocked by FIFO queueing
or memory pressure. Destination mutation remains single-writer and verification
failure rolls it back.

## Security

Child environments remove credential-shaped variables before adding only the
explicit provider key. Reports and caches must contain timings and hashes, not
raw credentials. State roots should remain user-private. Treat prompts, child
logs, patches, and reports as sensitive local data.

## Validation

Before increasing concurrency, run package-local tests plus cancellation,
timeout, stale-owner, verifier-mutation, merge-conflict, and worker-panic tests.
Use Miri for lock-sensitive standard-library code where supported and platform
CI for Linux/macOS process cleanup behavior.
