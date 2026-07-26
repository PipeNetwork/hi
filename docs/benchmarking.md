# Benchmarking and evidence-driven tuning

hi's codegen quality is managed by measurement, not intuition: a resolve-rate
regression gate over real-world issues, plus corpus rigs that pin the
heuristics most prone to silent drift. This doc is the operating manual.

## The resolve-rate gate: `hi bench swe`

```bash
hi bench swe                                  # every runnable Rust instance
hi bench swe --repo sharkdp__fd --limit 5     # quick sample
hi bench swe --instances id1,id2 --retries 2  # targeted re-runs / experiments
```

Each Multi-SWE-bench instance runs end-to-end: clone at the base SHA → hi
one-shot with the pinned prompt → standard-protocol grading. Verdicts:

- **RESOLVED** — every hidden fail-to-pass test passes, zero *introduced*
  failures.
- **NOT_RESOLVED** — no introduced failures, but the hidden tests still fail.
- **FAILED** — the change broke tests that passed at base.
- **INFRA** — the harness couldn't grade (clone/patch failure); excluded from
  the denominator.

Grading protocol details that keep the number honest:

- **Hidden tests own their files.** The agent's edits to any path the hidden
  test patch touches are reverted before grading — its *source* fix is what's
  judged (its own added tests would otherwise conflict textually).
- **Baseline attribution.** A would-be FAILED is re-graded against the base
  repository's own failures (agent changes dropped, hidden tests re-applied):
  only failures the agent *introduced* count. Pre-existing red suites — flaky
  or environment-dependent tests — cannot convert a correct fix into FAILED.
- **The prompt is pinned and test-guarded.** A controlled A/B showed that
  adding "Do NOT modify any existing test files" to the prompt flipped the
  intent classifier into read-only preflight and zeroed the resolve rate.
  Grading handles agent test edits structurally, so the prompt never mentions
  tests; a unit test fails if the clause regrows.
- `--retries N` re-runs a non-resolved instance from a fresh clone with the
  previous attempt's failing-test names in the prompt.

Evidence lands under `<state-root>/bench/`: `scorecard.jsonl` (verdict +
attempts per instance) and `runs/<instance>/` (prompt, transcript, agent
diff, grade logs). Failure transcripts are the tuning loop's raw material.

**Baseline (2026-07-26, ipop/coder-balanced, one-shot):** 142 gradable
instances → 55 RESOLVED (~39%). Per-repo: clap 43% · fd 43% · nushell 42% ·
ripgrep 29% · bat 14% · tracing 0%. 59% of failures were near-misses
(≤ 2 failing tests). Compare new runs against this line.

## Corpus rigs (ignored tests, run on demand)

Each rig replays a heuristic against a corpus of real-world data and reports
coverage or false positives. All are `#[ignore]`d tests driven by an env var:

| Rig | Guards | Invocation |
|---|---|---|
| Failure digests | verify_digest parsers (rustc, libtest, nextest, pytest, go, tracebacks) | `HI_DIGEST_CORPUS=<dir of .log> cargo test -p hi-agent --lib digest_corpus -- --ignored --nocapture` (`HI_DIGEST_SHOW=<file>` renders one digest) |
| Intent classification | read-only misclassification of implementation prompts | `HI_INTENT_CORPUS=<prompts.jsonl> cargo test -p hi-agent --lib intent_corpus -- --ignored --nocapture` |
| Convergence signatures | thrashing detection stability across reruns | `HI_CONVERGENCE_CORPUS=<pairs.jsonl> cargo test -p hi-agent --lib convergence_corpus -- --ignored --nocapture` |
| Impact notes | reverse-reference co-change prediction | `HI_IMPACT_CORPUS=<records.jsonl> cargo test -p hi-agent --lib impact_corpus -- --ignored --nocapture` |

Corpus sources that worked well: rust-lang/rust `tests/ui/*.stderr` (sparse
clone; the compiler's own diagnostic catalog), Hugging Face agent-trajectory
datasets (real failure logs and test-run sequences via the datasets-server
rows API), SWE-bench problem statements (ground-truth implementation prompts),
and Multi-SWE-bench gold patches (ground-truth multi-file co-change). Keep
corpora local (scratch or cache); commit only compact distilled fixtures.

## The tuning loop

1. Use hi on real work; run `hi metrics` — its tuning-signals section sweeps
   recent transcripts for digest parser gaps, repair thrashing, and impact-note
   activity.
2. `hi bench swe` for the resolve-rate trend before/after meaningful changes.
3. Every finding gets a *structural* fix (no phrase-list patches), a corpus or
   unit test that would catch its regression, and a re-measurement.
4. Measure before building: proposals that fail measurement die. (A planned
   near-duplicate loop detector was dropped when 121 transcripts showed
   successful runs near-dup almost as often as failed ones — 87% vs 91%.)

## Known caveats

- Verdicts depend on the model configured for the session; compare trends
  under one model, not absolutes across models.
- One suite-level flake during the baseline-attribution run can mask a real
  single-test regression; acceptable for a trend metric.
- `cargo test` grading is Rust-specific for now (`--lang rust`).
