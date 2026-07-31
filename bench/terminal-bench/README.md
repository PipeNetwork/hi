# hi on Terminal-Bench 2.1

Runs the `hi` coding agent against [Terminal-Bench 2.1](https://www.tbench.ai/docs/run-terminal-bench-2-1)
via [Harbor](https://github.com/harbor-framework/terminal-bench). The adapter
uploads a locally built static Linux binary into each task container and
invokes one-shot mode (`hi --provider … --model … <instruction>`); Harbor
scores the container afterwards with the task's own tests.

## Prerequisites

- Docker running
- `uv` (installs Harbor and this adapter into `.venv/`)
- A provider API key exported (e.g. `ANTHROPIC_API_KEY`)

## Quick start

```sh
cd bench/terminal-bench

# 1. Static Linux build of hi (host arch, via rust:alpine in Docker)
./build-linux.sh

# 2. Optional: validate the Harbor/Docker rig without an agent or API cost
uvx harbor run -d terminal-bench/terminal-bench-2-1 -a oracle -l 5

# 3. First real run: 5 tasks
./run.sh -m anthropic/claude-opus-4-8 -k 5
```

Useful `harbor run` flags (all pass through `./run.sh`):

- `-k N` — run only N tasks; start small, a full 89-task run is expensive
- `--include-task-name <name>` — a single task
- `-n N` — N concurrent trials (local Docker: keep small; add `--env daytona`
  with `DAYTONA_API_KEY` for cloud sandboxes)
- `--max-retries 3 --retry-include ApiRateLimitError` — retry rate-limited trials

## How the invocation maps to hi

- `-m provider/model` → `hi --provider <provider> --model <model>`; the model
  part may itself contain `/` (e.g. `openai/moonshotai/kimi-k2` routes
  provider `openai` to model `moonshotai/kimi-k2`).
- API keys pass through per provider exactly as hi resolves them
  (`ProviderName::key_envs`): e.g. anthropic reads `HI_API_KEY` then
  `ANTHROPIC_API_KEY`.
- `--no-save --no-memory` so nothing persists across tasks, and
  `--allow-unverified` so a completed turn exits 0 even when hi's own verifier
  didn't pass — the benchmark's tests are the ground truth here. A non-zero
  exit still fails the trial (surfaced by Harbor as an agent error).
- Full agent output is captured to `/installed-agent/hi-output.txt` in the
  container for debugging.

Set `HI_TB_BINARY=/path/to/hi` to test a different binary than
`dist/hi-linux`.

## Leaderboard notes

Submissions must use the fixed 2.1 task environments/timeouts and include
public trajectories — see the tbench.ai submission docs before reporting
numbers.
