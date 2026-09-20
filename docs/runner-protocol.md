# Guest runner protocol v1

`hi runner` uses newline-terminated JSON on stdin and stdout. Keep its pipes,
stderr, session, and managed recovery journal on the customer VM. This interface
uses the existing harness; stdout is never human terminal output. It does not
enroll a VM, acquire a server execution lease, or publish Git revisions.

The first input frame starts a run. The supervisor fills `lease_until_boot_ms`
from the actual guest boot clock; the illustrative value below is not a valid lease:

```json
{"version":1,"operation":"start","run_id":"11111111-1111-4111-8111-111111111111","attempt_id":"22222222-2222-4222-8222-222222222222","lease_until_boot_ms":123456789,"workspace":"/workspace/run-11111111","state_dir":"/home/pipe/.local/state/pipe-code/11111111","credential_file":"/home/pipe/.local/state/pipe-code/11111111/inference-key","prompt":"Fix the failing test","call_micros":1000000,"run_micros":20000000,"execution_seconds":3600}
```

Frames are at most 1 MiB including their newline. Send prompts only through stdin.
The supervisor may supply `deadline_unix_ms`, the original server deadline. It
must be in the future and cannot extend `execution_seconds`; resume must preserve
it exactly. The `ready` event returns that same millisecond deadline.
The credential file must be owned by the executing user with no group/other access.
State and credentials must be outside the checkout. The caller keeps stdin open;
EOF, malformed control frames, SIGINT, and SIGTERM request cancellation.
Execution requires Linux cgroup v2 with `cgroup.kill`. Before exec, the supervisor
places hi in a delegated `pipe-code-<attempt_id>` cgroup. hi checks that binding.
`lease_until_boot_ms` is an absolute `CLOCK_BOOTTIME` deadline no more than 60
seconds away. It counts VM suspension and cannot gain time in a delayed pipe.
Send `{"version":1,"operation":"lease","sequence":1,"lease_until_boot_ms":123456999}`
after each successful server renewal, incrementing the sequence. Expired leases
cannot revive a process. Both the dispatch guard and an independent timer enforce
the bound; hi kills its attempt cgroup even if its parent is stopped. Nonblocking
output preserves that fence when stdout fills.

Subsequent frames are `{"version":1,"operation":"status"}` and
`{"version":1,"operation":"cancel"}`. Cancellation acknowledgment means a
request was received. Wait for the terminal event and supervise the process tree
before reporting confirmed execution stop.

Output frames have `version`, `event`, and `data`. Events include `ready`,
`assistant_text`, `assistant_reasoning`, `assistant_end`, `tool_started`,
`tool_call`, `tool_stream`, `tool_result`, `status`, `turn_end`, `error`,
`cancellation_requested`, `checkpoint_required`, and `terminal`. Tool lifecycle events carry the existing
harness tool-call ID when available. Detailed fields contain customer content.
They must not be copied into central metadata or logs.

After a completed mutation batch and durable tool receipts, hi emits
`checkpoint_required` with a fresh `checkpoint_id`. It dispatches no further
tools or inference until the supervisor journals/pushes the exact checkpoint and
returns `{"version":1,"operation":"checkpoint_ack","checkpoint_id":"<UUID>","source_sha":"<40-hex SHA>"}`.
A wrong, duplicate, or expired receipt cancels execution. Resume first reconciles
any outstanding checkpoint independently of tool execution.

The terminal status is `completed`, `cancelled`, `timed_out`, or `requires_action`.
Test summaries and changed files are client-reported evidence. `completed` is a
harness result, not permission to publish or proof that tests passed. Before
emitting it, hi fsyncs `terminal-<attempt_id>.json` in its private state directory.
The supervisor can recover a lost final stdout response from that exact receipt.
hi closes the complete attempt cgroup after durable terminal output; its process
may therefore exit with SIGKILL even after a completed result. The supervisor
checks the receipt and confirms the cgroup is empty, rather than treating an exit
code as execution or publication evidence.

To resume, send the same first frame with `operation: "resume"` and omit `prompt`.
The original session, managed journal, credential identity, limits, and absolute
deadline must remain available. Resume rejects a changed run ID, workspace,
credential path, endpoint, or limits. A local file lock excludes another process
using the same state directory. The guest supervisor must separately enforce
one run per VM, renew the fenced server lease every 15 seconds, and cancel the
dedicated process group on lease loss. A missing or ambiguous journal requires
reconciliation; starting another run is not automatic recovery.

Use `operation: "inspect"` with the original binding and no prompt to inspect
recovery without inference or tool dispatch. The terminal event has status
`inspected`, the original deadline, and bounded recovery metadata: the journal's
BLAKE3 and SHA-256 hashes, ambiguous tool count, unretained response count, unresolved call count
and `can_resume`. An unsubmitted prepared request is safe to submit once; a
submitted request whose response is missing requires reconciliation. Inspection
also works after the execution deadline, but does not authorize execution.
Completed runner output includes the same `recovery` object. Its optional
`accepted_final_key` is the original final-answer idempotency key; the server must
verify that key under this run's credential before accepting publication evidence.

Managed requests use balanced `pipe/auto`, required buffered verification, retrieval
off, and the existing local recovery journal. No ambient interactive configuration,
transcript sync, ordinary-model fallback, or unbudgeted Typesafe/JeV path is enabled.
Server authorization and aggregate billing enforcement remain necessary; local
configuration does not replace them.
