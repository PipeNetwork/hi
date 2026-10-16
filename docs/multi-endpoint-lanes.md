# Multi-endpoint task lanes

`hi` talks to one OpenAI-compatible inference URL by default (`--base-url` /
the selected profile). That path is unchanged.

When you have several peer GPUs on a LAN — for example four GB100 boxes each
serving vLLM (typically Qwen3.8-27B-NVFP4; one box may run DeepSeek-R1) — you
can opt into **task-lane parallelism**:

1. One workspace, one `hi` session, one transcript.
2. The profile `base_url` is the **coordinator**. It plans, merges, and does
   non-path work (open-ended shell, whole-repo questions).
3. Independent coding work is split into **narrow lanes** (tight sub-tasks with
   only the files that lane needs) and fanned across the other endpoints in
   parallel.
4. Optional path prefixes/globs *hint* which GPU a lane should use. They do
   not pin a crate to a box forever.
5. Lanes that would write overlapping files are serialized. Disjoint lanes on
   different endpoints run together so tokens/sec can scale with GPU count.
6. If one endpoint is down, that lane returns a clear shard error and the
   others keep going.

This is not round-robin load balancing of a single conversation stream, and it
is not “each crate lives on its own box.”

## Configure four LAN vLLM boxes

In `./hi.toml` or `~/.config/hi/config.toml`:

```toml
[profiles.four-box]
provider = "ollama"                      # OpenAI-compatible wire; no cloud key
model = "Qwen3.8-27B-NVFP4"              # must match vLLM --served-model-name
base_url = "http://192.168.1.10:8000/v1" # coordinator (and default leftover work)
api_key = "local"                        # vLLM ignores this; pair it with the URL
max_tokens = 8192
tool_mode = "auto"

[[profiles.four-box.endpoints]]
id = "gpu-a"
base_url = "http://192.168.1.10:8000/v1"
default = true
paths = ["crates/a/**", "crates/hi-cli/**"]

[[profiles.four-box.endpoints]]
id = "gpu-b"
base_url = "http://192.168.1.11:8000/v1"
paths = ["crates/b/**", "crates/hi-harness/**"]

[[profiles.four-box.endpoints]]
id = "gpu-c"
base_url = "http://192.168.1.12:8000/v1"
paths = ["crates/c/**", "crates/hi-tools/**"]

[[profiles.four-box.endpoints]]
id = "gpu-d"
base_url = "http://192.168.1.13:8000/v1"
model = "DeepSeek-R1"                    # optional override on this box only
paths = ["crates/d/**", "docs/**"]
```

Then:

```bash
hi -p four-box "split the remaining work into tight lanes and implement it"
```

`hi --show-config -p four-box` lists the coordinator URL and each lane
endpoint. Omit `[[endpoints]]` (or use any other profile) and `hi` is the
usual single-endpoint session.

The lab profile in `hi.toml` is `sparks`. The 512GB Mac Studio
(`100.122.74.84`, endpoint `mac-512gb`) coordinates and is the default
worker. It serves GLM-5.3 Flash through mlx_vlm on port 8080; the model
id is the on-disk path `/Users/david/models/glm-5-3-flash-mlx-mixed-4_8bit`.
`spark-7466` (`100.116.163.116`), `spark-7a60` (`100.81.119.49`), and
`gx10-3607` (`100.113.145.127`) are peer workers on the same tree, with
no path prefixes. Each of those runs the `hi-lane` user service:
`hi-local-v2` on port 8000 serving Qwen3-Coder-30B-A3B (Q4_K_M) as model
id `qwen3-coder-30b-a3b`, with `HI_CUDA_PREFIX_CACHE=1`. That cache turns a
repeated 2.7k-token prefill from 9s into 0.2s, which matters for write
lanes because every tool round resends the lane so far. `gx10-1e82`
(`100.74.5.66`) is offline.

```bash
hi --show-config -p sparks
hi -p sparks "split the remaining work into tight lanes and implement it"
```

Each endpoint field:

| Field | Role |
|---|---|
| `id` | Label in the transcript (`gpu-a`, …) |
| `base_url` | OpenAI-compatible origin (`…/v1`) |
| `model` | Optional; inherits the profile model |
| `api_key` / `api_key_env` / `api_key_ref` | Optional; inherits the profile key |
| `paths` | Prefixes (`crates/a`) or globs (`crates/a/**`, `*.md`) used as assignment hints |
| `default` | Where a single unmatched lane lands. Several unmatched lanes spread across free GPUs, one each, before any box takes a second lane |

`provider = "ollama"` is the usual choice for LAN vLLM because it does not
require a real API key. `provider = "openai"` works too if you pair
`api_key` / `api_key_ref` with the profile (custom origins cannot borrow
ambient OpenAI/OpenRouter keys).

Project-local `hi.toml` that points at non-loopback URLs still needs folder
trust, same as any other custom `base_url`.

## How it differs from `--base-url`

| | Single endpoint | Lane pool |
|---|---|---|
| Config | `--base-url` / profile `base_url` only | profile `base_url` **plus** `[[endpoints]]` |
| Session | One model stream | Coordinator stream + N slim lane streams |
| Parallelism | Tool calls on one GPU | Independent lanes on different GPUs |
| Context | Full session transcript | Each lane gets the task + named files only |
| Failure | The turn fails if that URL is down | One shard errors; other GPUs stay usable |

`--base-url http://192.168.1.10:8000/v1` still means “this whole conversation
goes to that one server.” Adding `[[endpoints]]` is what turns four boxes
into peer workers.

## What the coordinator does

When the pool is present, `hi` injects a `lane` tool (ADR 002: feature-gated,
not in the always-on catalog). The coordinator is prompted to emit several
`lane` calls in one response:

- `task` — one concrete sub-task
- `paths` — files/directories that lane owns (keep write scopes disjoint)
- `context_files` — the only files preloaded into that worker
- `readonly` — investigate without taking the write lock against other lanes.
  A readonly lane gets the preloaded files and no tools, and answers in one
  completion.

The harness then:

1. Matches each lane to an endpoint. An explicit `endpoint` wins, then a
   path hint. Lanes that match neither spread across free GPUs, one per box.
   A single unmatched lane stays on the `default` worker. Among equally
   loaded GPUs, a lane whose first file matches the last lane a box ran goes
   back to that box, where its prefix is still cached.
2. Runs lanes on free GPUs, one at a time per box, skipping write-path
   overlaps. A GPU that finishes starts the next lane immediately.
3. Runs the worker with a **slim** prompt: worker instructions, then the
   preloaded files, then the owned paths, then the task. Files come first so
   lanes over the same files share a cacheable prefix. Readonly lanes stop
   after that one reply. Write lanes get read, write, edit, grep, and
   foreground bash, at most four rounds. Readonly output is capped at 384
   tokens and write-lane output at 1536. Tool results fed back into the lane
   are clipped. The lane result lists changed files and the edits that
   applied.
4. Merges a short summary back into the parent session. The coordinator's
   system prompt and tool list are identical on every request, so a server
   with a prefix cache (mlx_vlm, vLLM) reuses the earlier prefill for the
   merge and for any follow-up wave.

Lane workers cannot spawn further lanes. `/undo` still restores the parent
turn’s git checkpoint. Pipe Network login, `/verify`, and the TUI are
unchanged when `[[endpoints]]` is absent.

## Write scope and permissions

Lane sessions start in always-approve. `--confirm-edits` (or `/permissions ask`)
confirms every lane diff in one overlay. Auto mode applies safe file edits and
asks for the rest together.

Lane file edits stay inside the declared `paths`. A write, edit, or patch
whose target sits outside that set is refused. A lane that names no paths may
write anywhere in the workspace and is scheduled alone against other writers.
Readonly lanes inspect the tree.

A write lane sees a small tool list: `read`, `write`, `edit`, `grep`, and
foreground `bash`. A readonly lane sees none. Inside a declared scope, shell
is read-only or a build/test command (`cargo`, `pytest`, `go`, `npm`, and the
usual siblings). Those builds
get their own `CARGO_TARGET_DIR` under `.hi/lane-targets/<gpu>/`, so two lanes
can test at the same time. Other mutating shell is refused. A lane with no
paths may run mutating shell, one command at a time.

The coordinator prompt lists each endpoint id, model, and path hint. Pass
`endpoint` on the `lane` tool to pin a GPU (for example the DeepSeek box).

A free GPU starts the next non-conflicting lane without waiting for a slower
peer. Before the first lane, `hi` probes `GET /models` on each box. A transport
failure or HTTP 5xx takes that GPU out and moves the lane once onto the next
healthy box. `hi --show-config` prints `up` or `down` for each endpoint.

While a lane runs, the TUI keeps one live line per lane above the transcript
(endpoint, current tool, last line) and replaces that line in place. The tool
result includes a short diff and that lane's token counts. The session status
line gains a `lanes gpu-a 12/4, gpu-b 8/3` suffix (input/output) after a lane
round.
