# Running hi on Terminal-Bench 2.0

[Terminal-Bench](https://github.com/harbor-framework/terminal-bench) is the
standard benchmark for terminal coding agents; [Harbor](https://github.com/harbor-framework/harbor)
is its official harness. Each task runs in its own Docker environment; Harbor
installs the agent into the container, hands it the task instruction, and
scores the resulting container state with the task's own tests — outcome-only,
so intermediate behavior earns no credit. This is the setting studied in
"Failure as a Process" (arXiv:2607.09510), whose 21 model–scaffold baselines
span 19–45% pass rate.

## One-time setup

1. **Harbor** (needs Python ≥3.12 and Docker running):

   ```bash
   uv tool install harbor
   ```

2. **A Linux build of `hi`** matching the task-image architecture. The
   published Terminal-Bench 2.0 task images are **linux/amd64 only** — even on
   Apple Silicon (Docker runs them under Rosetta), the binary must be x86_64.
   The workspace's GPU crates don't build in the container, but the binary
   crate builds alone; the reproducible path is a containerized build:

   ```bash
   docker run --rm --platform linux/amd64 -v "$PWD":/src:ro \
     -v "$PWD/target-linux-amd64":/out \
     rust:1-bookworm cargo build --release -p hi \
     --manifest-path /src/Cargo.toml --target-dir /out
   export HI_AGENT_BINARY="$PWD/target-linux-amd64/release/hi"
   ```

   (An emulated build on Apple Silicon takes several minutes; it only needs
   to happen once per hi revision.)

## Running

```bash
export PIPENETWORK_API_KEY=...        # or the provider your model needs
scripts/terminal_bench.sh sample      # ~10 tasks, plumbing + cost check
scripts/terminal_bench.sh full        # full dataset — mind the spend
scripts/terminal_bench.sh task 'git*' # tasks matching a glob
```

Model selection: `HI_TB_MODEL=<provider>/<model>` (default
`pipenetwork/ipop/coder-balanced`). Known providers: `pipenetwork`,
`anthropic`, `openrouter`, `openai` — the adapter maps them to `hi --provider`
and forwards the matching API key from your environment into the container.

Results land under `jobs/terminal-bench/<job-name>/` — per-trial directories
with the agent transcript (`hi.txt`), hi's own `--report` telemetry
(`hi-report.json`, including `repeated_verify_failures` and the trajectory
timeline), and Harbor's verifier verdicts. Harbor prints the aggregate pass
rate at the end of the job.

## How the adapter works

`integrations/terminal_bench/hi_agent.py` implements Harbor's
`BaseInstalledAgent`:

- `install()` uploads `$HI_AGENT_BINARY` into the container at `/opt/hi/hi`
  and symlinks it onto `PATH`.
- `run()` executes one `hi --plain --no-save --allow-unverified` turn with the
  task instruction, `--report` pointed at Harbor's log directory, and
  `XDG_CONFIG_HOME` isolated so a config file baked into a task image can
  never shadow the model routing. The command never fails the trial on hi's
  exit code — scoring is the verifier's job.
- `populate_context_post_run()` reads the downloaded report and fills Harbor's
  token accounting plus a `hi_telemetry` metadata block.

## Caveats

- **Architecture**: the uploaded binary must match the task images. Harbor on
  Apple Silicon uses aarch64 images by default; build accordingly.
- **Timeouts**: Terminal-Bench tasks carry their own generous agent timeouts;
  hi's dynamic step caps apply within them. Use `--timeout-multiplier` on
  `harbor run` for slow models.
- **Cost**: the full dataset is hundreds of agent turns. Always run `sample`
  first and extrapolate spend before `full`.
