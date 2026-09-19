# Managed coding with Pipe Auto

Select `hi --model pipe/auto "Fix the bug and run the tests"` or select `pipe/auto`
in the model picker. The default model is unchanged. A paired project key and
available inference credits are required; customers do not need a TypeSafe key.
If the project has not reached the coding rollout, hi explains the unavailability
and stops. There is no fallback to ordinary inference.

A saved profile may override the defaults:

```toml
[profiles.pipenetwork.managed]
profile = "balanced"
call_budget_usd = "1.000000"
turn_budget_usd = "20.000000"
deadline_ms = 180000
```

Amounts are exact decimal USD strings. Every generation and verification attempt,
including unsuccessful attempts and compaction, consumes the same turn budget at
provider cost. Project/key/credit limits may reduce the per-call allowance. Unknown
costs remain reserved. Resuming a turn preserves its original settings and budget.
Managed coding disables optional direct TypeSafe helper calls; the managed server
performs required review.

The server reviews each assistant proposal and final answer before releasing it.
Tools still execute locally under hi's normal permission checks, sandbox and
checkpoints. Verification checks **client-reported execution evidence**; it is not
independent server execution or proof that code is correct. A test-success claim
needs a corresponding successful reported test command after the latest edit.
Repository/tool output is untrusted data, including instructions embedded in files.

Coding accepts text and function tools, with a 1 MiB request ceiling, 64,000 total
context tokens, up to 8,192 output tokens, and a 180-second call deadline, narrowed
by available providers. The initial context accounting uses a conservative UTF-8
byte bound, so effective usable text can be smaller than a model-token estimate.
Required verification uses bounded evidence references and may reject a request
whose necessary evidence is missing. Server web retrieval is off.

Servers retain accounting, hashes and finite verification receipts, not prompts,
tool contents or generated answers. hi keeps its recovery journal locally beside
the session as `<session>.managed.json`, with owner-only permissions and durable
writes. Protect it as you would the local conversation transcript.

On interruption, hi queries `/v1/pipe/requests/by-key` using the original
`Idempotency-Key`. Cancellation uses `POST /v1/pipe/requests/by-key/cancel`, including
before response headers arrive. Both require the original project authentication.
A cancellation submitted before admission prevents a later racing submission.
Status and cancellation remain available when new managed admission is disabled.

If a response was lost after server completion, the server reports
`result_not_retained`. hi stops with the idempotency key and journal path instead
of regenerating. Inspect the request status and local repository before starting
another separately budgeted turn. Never delete the journal to reset a budget.
Accepted local responses and completed tool results are reused on resume. A tool
journaled as started without a completed result requires manual reconciliation of
its effects; hi will not repeat it automatically. Interrupted or malformed SSE,
missing terminal metadata and truncated tool arguments cannot authorize tools.
