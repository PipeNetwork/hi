# Profile-driven evaluations

`hi eval` separates source adapters, immutable task imports, execution, and
evidence. Start with a copied `manifest.example.toml`:

```text
hi eval import --manifest evals/manifest.toml --profile smoke
hi eval run --manifest evals/manifest.toml --profile smoke
hi eval status --profile smoke
hi eval report --profile smoke
```

The initial adapter boundary accepts directories containing the existing
schema-v2 `task.toml` files or normalized `package.toml`/`package.json` task
packages. The route catalog already reserves the current Harbor, Terminal-Bench,
DeepSWE, StableBench, Arena-Hard, OpenAI Evals, SWE-bench, GeneBench, GraphWalks,
MRCR, HealthBench, GDPval, SWE-Atlas, GPQA, BrowseComp, ARC-AGI-3, and Agents'
Last Exam routes. Format-specific readers can be added without changing the
runner contract.

Every profile result carries a claim level: `official`, `public_reproduction`,
`smoke`, or `evidence_only`. A continuous named reward is retained separately
from binary pass classification.

## Harness profile (process + budget)

`bench/harness` scores *how* hi ran, not just whether the hidden oracle passed.
Each task has a `judge.toml` sidecar. CI replays committed `tape/*.json` with
no API key:

```text
cargo run -p hi-eval -- judge --suite bench/harness
scripts/check_harness_regression.sh
```

Live (optional): `HI_MODEL=… cargo run -p hi-eval -- bench/harness --configs=verify --trials=1 --artifacts=artifacts/harness`.
Set `HI_STATE_DIR` so process/budget fails append `harness_process` /
`harness_budget` findings. See `bench/harness/README.md`. The Monday
scheduled workflow runs this as its own job, not mixed with `bench/tasks`.

## Hygiene profile

`hi bench swe` scores hidden tests only. Merge-quality regressions (extra files,
dependency-manifest edits, whole-file rewrites) are invisible there. The
`hygiene` profile is a separate smoke corpus beside SWE:

```text
hi eval import --manifest evals/manifest.toml --profile hygiene --dry-run
hi eval import --manifest evals/manifest.toml --profile hygiene
hi eval run --manifest evals/manifest.toml --profile hygiene
```

Tasks live under `bench/tasks/hygiene-*`. Each oracle fails extra files
(`allowed_changes`), unexpected dependency-manifest edits, or a file that grew
past a byte cap. Do not fold this into SWE grading.

## Docker/Harbor profiles

Use `backend = "docker"` (or `"harbor"`) for a task package whose
`environment` is an OCI image or Dockerfile. Docker Desktop is supported; the
candidate and verifier run in separate fresh Linux containers with the
candidate workspace mounted at `/workspace` and evidence at `/evidence`.

Each Docker arm must provide `command`, which is executed inside the image via
`/bin/sh -lc`. For final-message tasks, the input is mounted at
`/input/eval-input.json`, and the command should write either
`/evidence/hi-report.json` with `assistant_response` or
`/evidence/final_message.txt`. Verifiers receive `HI_EVAL_FINAL_MESSAGE`,
`HI_EVAL_OUTPUT`, and `HI_EVAL_ARTIFACTS` when applicable.

Images are checked locally and are never pulled implicitly. Run `docker pull`
or build the image before `hi eval prepare`. Docker's disabled network policy
maps to `--network none`; scoped host allowlists are rejected until a policy
enforcement proxy is available. Docker Desktop does not normally support
per-container overlay storage quotas, so storage is recorded as requested but
not enforced unless `HI_DOCKER_ENFORCE_STORAGE=1` is set; on unsupported
drivers that mode fails explicitly.

Useful overrides are `HI_DOCKER_BIN=/path/to/docker` and
`HI_DOCKER_ENFORCE_STORAGE=1`.
