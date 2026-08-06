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
