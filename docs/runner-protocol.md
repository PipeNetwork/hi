# Guest runner protocol v1

`hi runner` uses newline-terminated JSON on stdin and stdout. Keep its pipes,
stderr, session, and managed recovery journal on the customer VM. This interface
uses the existing harness; stdout is never human terminal output. It does not
enroll a VM, acquire a server execution lease, or publish Git revisions.

The first input frame starts a run:

```json
{"version":1,"operation":"start","run_id":"11111111-1111-4111-8111-111111111111","workspace":"/workspace/run-11111111","state_dir":"/home/pipe/.local/state/pipe-code/11111111","credential_file":"/home/pipe/.local/state/pipe-code/11111111/inference-key","prompt":"Fix the failing test","call_micros":1000000,"run_micros":20000000,"execution_seconds":3600}
```

Frames are at most 1 MiB including their newline. Send prompts only through stdin.
The credential file must be owned by the executing user with no group/other access.
State and credentials must be outside the checkout. The caller keeps stdin open;
EOF, malformed control frames, SIGINT, and SIGTERM request cancellation.

Subsequent frames are `{"version":1,"operation":"status"}` and
`{"version":1,"operation":"cancel"}`. Cancellation acknowledgment means a
request was received. Wait for the terminal event and supervise the process tree
before reporting confirmed execution stop.

Output frames have `version`, `event`, and `data`. Events include `ready`,
`assistant_text`, `assistant_reasoning`, `assistant_end`, `tool_started`,
`tool_call`, `tool_stream`, `tool_result`, `status`, `turn_end`, `error`,
`cancellation_requested`, and `terminal`. Tool lifecycle events carry the existing
harness tool-call ID when available. Detailed fields contain customer content.
They must not be copied into central metadata or logs.

The terminal status is `completed`, `cancelled`, `timed_out`, or `requires_action`.
Test summaries and changed files are client-reported evidence. `completed` is a
harness result, not permission to publish or proof that tests passed.

To resume, send the same first frame with `operation: "resume"` and omit `prompt`.
The original session, managed journal, credential identity, limits, and absolute
deadline must remain available. Resume rejects a changed run ID, workspace,
credential path, endpoint, or limits. A local file lock excludes another process
using the same state directory. The guest supervisor must separately enforce
one run per VM, renew the fenced server lease every 15 seconds, and cancel the
dedicated process group on lease loss. A missing or ambiguous journal requires
reconciliation; starting another run is not automatic recovery.

Managed requests use balanced `pipe/auto`, required buffered verification, retrieval
off, and the existing local recovery journal. No ambient interactive configuration,
transcript sync, ordinary-model fallback, or unbudgeted Typesafe/JeV path is enabled.
Server authorization and aggregate billing enforcement remain necessary; local
configuration does not replace them.
