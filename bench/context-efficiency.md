# Context efficiency regression fixtures

These deterministic fixtures measure fixes to avoidable request and tool-output
overhead. They do not measure live provider charges or model coding pass rates.

| Fixture | Before | After | Preserved evidence |
| --- | ---: | ---: | --- |
| 800 passing Rust tests with `expected_error` in their names | 5,034 output characters | 507 output characters | Test totals and result; separate regression retains real multiline failures |
| 100,000-character single-line read, configured budget 512 | 100,006 output characters | 512 output characters | Line number, explicit truncation, instructions for inspecting the remainder |
| Two tool-enabled requests to a server rejecting usage streaming and frequency penalty | 8 HTTP requests / 10,016 request bytes | 4 HTTP requests / 4,968 request bytes | Identical messages and tool schemas; explicit compatibility settings remain authoritative |
| Eight retired signed-thinking blocks in a synthetic conversation | 38,211 serialized bytes / 9,251 estimated input tokens | 5,219 serialized bytes / 1,059 estimated input tokens | Stable system prefix, recent user request, source, and failure evidence |
| Program selects a small answer after a large read and a failed read | 5,037 output bytes / 1,262 estimated input tokens | 193 output bytes / 51 estimated input tokens | Selected answer, call statuses, and failed-read diagnostics |
| Folding 100 tiny source reads | 26,183 serialized bytes / 2,990 estimated message tokens | 20,639 serialized bytes / 1,505 estimated message tokens | Complete original read contents; replacement notices are skipped when larger |
| Fallback grep with 40 matches, 80 source lines, and 3 context lines | 5,033 output characters / 25 distinct matches retained | 2,431 output characters / all 40 matches retained | Overlapping context appears once, each matching line is marked correctly |

The thinking fixture measures removal of obsolete signatures from already-elided
reasoning. It is also a correctness fix: the old representation paired changed
reasoning text with its original signature. Recent signed reasoning remains intact.
The request fixture counts rejected HTTP requests; fewer rejected requests should
not be interpreted as equivalent savings in billed tokens.

Verification excerpts now cap each source line and read only bounded regular
files inside the workspace. A 240,000-byte minified source fixture produces a
611-byte excerpt with a truncation notice. A 1,000-failure fixture produces a
197-byte digest while retaining every failure in its comparison signature and
leaving the stage output available to the repair loop.

Context integrity tests also cover incremental background logs, recent image
attachments through compaction, and legacy session repair. These protections
matter because removing unique evidence can waste subsequent coding attempts.

Program results no longer duplicate successful intermediate payloads. Large
selected results use a valid JSON preview with explicit truncation instead of
clipping the serialized envelope into malformed JSON.

Read folding now pairs each result with its originating call occurrence. Reused
call IDs, including legacy duplicate IDs in a batch, cannot retire newer source
or unrelated error evidence. The tiny-read comparison measures the old folding
behavior against the fixed behavior on the same transcript; it does not imply
that every conversation shrinks by this proportion.

Search output bypasses test-log condensation so source text quoting test results
cannot suppress ordinary matches. Search includes hidden configuration, excludes
the harness's dependency cache, and no longer silently stops at 200 matches in a
file when the configured output budget could retain more.

Reproduce the focused fixtures from the repository root:

```sh
cargo test -p hi-tools --no-default-features --lib token_tests -- --nocapture
cargo test -p hi-tools --no-default-features --lib read::formatting::tests -- --nocapture
cargo test -p hi-ai --no-default-features --lib compatibility_tests -- --nocapture
cargo test -p hi-agent --no-default-features --lib context_integrity -- --nocapture
cargo test -p hi-agent --no-default-features --lib verify_digest::source -- --nocapture
cargo test -p hi-agent --no-default-features --lib program_selected_output -- --nocapture
cargo test -p hi-agent --no-default-features --lib folding_tiny_reads_never_inflates_request_context -- --nocapture
cargo test -p hi-tools --no-default-features --lib read::search_tests::dense_fallback_grep_context_fixture -- --nocapture
```

Provider transport tests require permission to bind a local test server. For live
coding-quality and token comparisons, use the paired evaluation procedure in
`bench/README.md` with the same model, tasks, retry budget, and environment.
