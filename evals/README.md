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
