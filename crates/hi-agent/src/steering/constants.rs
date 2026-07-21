//! Nudge strings, context limits, and preflight patterns referenced by
//! [`nudges`](super::nudges) and [`preflight`](super::preflight).

/// Sent when the model re-issues the exact same tool call as the previous
/// round. The command already ran and its output is in the history just above —
/// re-running it will only produce the same result. This nudges the model to act
/// on that output (edit the code, move on, or finish) instead of looping.
pub(crate) const REPEAT_NUDGE: &str = "You just ran that exact command last round and its output is already \
in the conversation above — running it again will only repeat the same result. Act on that output \
now: make the edit it points to, move to the next step, or if the task is already complete, stop \
and give your final recap. Do not re-run the same command.";

/// Synthetic tool result recorded for a call the repeat guard skipped. The
/// skipped call stays in the transcript paired with this result (provider-safe)
/// so the model sees exactly what happened to the call it just made — stripping
/// the call left weak models convinced the tool layer was broken ("my tool
/// calls aren't producing visible output") and they gave up instead of
/// correcting course.
pub(crate) const SKIPPED_REPEATED_CALL_RESULT: &str = "[not executed: this call is identical to \
the one you made last round. Its result is unchanged and already shown above — act on that result \
instead of re-issuing the call.]";

/// Synthetic tool result for a repeated, unchanged `update_plan` call. Models
/// are told to keep re-posting the plan as statuses change, so an identical
/// re-post is a common weak-model stall: harmless bookkeeping, but zero
/// progress. Point the model at executing the plan instead.
pub(crate) const SKIPPED_PLAN_REPOST_RESULT: &str = "[not executed: this plan is already recorded \
exactly as posted — re-posting an unchanged plan does nothing. Execute the plan's next step now \
with your other tools; call update_plan again only when a step's status changes.]";

/// Synthetic tool result for a repeated, unchanged bookkeeping call other than
/// `update_plan` (today: `record_decision`). Same stall pattern as the plan
/// re-post: meta-work instead of work.
pub(crate) const SKIPPED_BOOKKEEPING_REPOST_RESULT: &str = "[not executed: this bookkeeping call \
is already recorded from your previous identical call — recording it again does nothing. Do the \
actual work now with your repository tools (read, list, grep, bash, edit).]";

/// Sent when the model re-posts an identical `update_plan` call instead of
/// working. The generic [`REPEAT_NUDGE`] ("you just ran that exact command…
/// act on its output") reads as nonsense for a bookkeeping call whose output
/// is a one-line ack, and confused models into believing their tools were
/// broken. This names the actual problem and the concrete next action.
pub(crate) const PLAN_REPOST_NUDGE: &str = "You re-posted the same plan without doing any work. \
The plan is already recorded — do not call update_plan again until a step's status actually \
changes; bookkeeping tools are unavailable for your next action. Execute the first incomplete \
plan step now using your other tools (read, list, grep, bash, edit).";

/// Sent when the model repeats identical bookkeeping calls (`update_plan`,
/// `record_decision`) instead of working. Observed live: withholding only
/// `update_plan` made the model slide to `record_decision` and repeat that
/// instead — so the nudge (and the one-round tool withholding that accompanies
/// it) covers the whole bookkeeping family.
pub(crate) const BOOKKEEPING_REPOST_NUDGE: &str = "You repeated a bookkeeping call \
(update_plan/record_decision) that was already recorded, without doing any work. Those records \
are saved; bookkeeping tools are unavailable for your next action. Do the actual work now: \
inspect files with read/list/grep, run a command with bash, or make an edit.";

pub(crate) const NO_EVIDENCE_REVIEW_NUDGE: &str = "This read-only review has no inspected evidence yet. \
Do not finalize. Use read-only inspection tools first, then answer from the inspected evidence. \
If inspection is impossible, explain which inspection failed and what remains unknown.";
pub(crate) const READ_ONLY_SAFE_CONTEXT_WINDOW: u32 = 12_000;
pub(crate) const READ_ONLY_PREFLIGHT_GREP_MAX_LINES: usize = 32;
pub(crate) const READ_ONLY_PREFLIGHT_DIFF_MAX_LINES: usize = 160;
pub(crate) const SECURITY_PREFLIGHT_EXTRA_READ_LIMIT: u32 = 90;
pub(crate) const DEFAULT_PREFLIGHT_EXTRA_READ_LIMIT: u32 = 120;
pub(crate) const READ_ONLY_PREFLIGHT_MAX_EXTRA_READS: usize = 3;
pub(crate) const NO_EVIDENCE_SECURITY_NUDGE: &str = "This security review has no inspected evidence yet. \
Do not finalize. Search for unsafe, unwrap, expect, panic!, command execution, filesystem/env \
access, and secret/token/auth patterns, then read the most relevant matching files before answering.";
pub(crate) const NO_EVIDENCE_STATUS_NUDGE: &str = "This status review has no inspected evidence yet. \
Do not finalize. Inspect git status or diff summary, workspace manifests, README/docs if present, \
main crate or module entrypoints, and tests before making status claims.";
pub(crate) const NO_EVIDENCE_GAP_NUDGE: &str = "This gap or roadmap review has no inspected evidence yet. \
Do not finalize. Inspect manifests, owning modules, tests, and TODO/FIXME or missing-coverage \
search results before naming gaps or build-next work.";
pub(crate) const REVIEW_DEEPEN_NUDGE: &str = "This read-only review only has a directory listing so far. \
Do not finalize yet. Use a targeted search or read relevant files, then answer from the inspected \
evidence. If deeper inspection is impossible, explain which files or searches could not be checked.";
pub(crate) const SECURITY_DEEPEN_NUDGE: &str = "This security review only has a directory listing so far. \
Do not finalize yet. Search for unsafe, unwrap, expect, panic!, command execution, filesystem/env \
access, and secret/token/auth patterns, then read the most relevant matching files before answering.";
pub(crate) const STATUS_DEEPEN_NUDGE: &str = "This status review only has a directory listing so far. Do \
not finalize yet. Inspect git status or diff summary, workspace manifests, README/docs if present, \
main crate or module entrypoints, and tests before making status claims.";
pub(crate) const GAP_DEEPEN_NUDGE: &str = "This gap or roadmap review only has a directory listing so far. \
Do not finalize yet. Inspect manifests, owning modules, tests, and TODO/FIXME or missing-coverage \
search results before naming gaps or build-next work.";
pub(crate) const CONCRETE_REVIEW_NUDGE: &str = "Your read-only review answer did not cite concrete files or \
modules from the inspected evidence. Do not use mutating tools. Answer again with bounded findings \
tied to inspected paths and a brief Limits section naming what remains unknown.";
pub(crate) const READ_AFTER_SEARCH_NUDGE: &str = "The targeted search result is already in the transcript. \
Do not rerun the same search and do not use mutating tools. Read the most relevant matching file, \
then answer from that inspected file. If you cannot pick a file to read, explain that limitation \
and answer only from the search output.";

/// Default file-read + targeted-search caps for read-only review turns. Listings
/// and diffs remain useful context, but they do not increase this count.
/// Sized for multi-crate workspaces: enough greps + file reads to ground findings
/// before the sprawl nudge forces an answer; still bounded so review turns cannot
/// churn toward max_steps on distinct-file thrash.
///
/// These are the **base** caps per intent. The effective cap is computed by
/// [`active_read_only_inspection_cap`] which applies a task-type multiplier and
/// a project-size ceiling. See [`inspection_cap_multiplier`] for the scaling
/// system.
pub(crate) const REVIEW_INSPECTION_CAP: u32 = 32;
pub(crate) const STATUS_INSPECTION_CAP: u32 = 20;
pub(crate) const ROADMAP_INSPECTION_CAP: u32 = 28;
pub(crate) const GAPS_INSPECTION_CAP: u32 = 28;
pub(crate) const SECURITY_INSPECTION_CAP: u32 = 40;

/// How many *additional* read-only inspection rounds are allowed after the
/// sprawl nudge before the turn stops incomplete.
pub(crate) const MAX_INSPECTION_SPRAWL_NUDGES: u32 = 2;

// ── Inspection cap scaling system ──────────────────────────────────────────
//
// The base caps above are task-blind and project-blind. The effective cap is:
//
//   effective = min(base * task_multiplier, project_size_ceiling) + soft_cap_extension
//
// where:
//   - task_multiplier scales the base for broad-scope tasks (review/audit)
//     vs. narrow ones (status check)
//   - project_size_ceiling raises the upper bound for large repos
//   - soft_cap_extension lets the agent request more budget with justification

/// Multiplier applied to the base inspection cap for each intent. Broad-scope
/// tasks (review, security, gaps, roadmap) get a higher multiplier so they can
/// cover more ground; status stays at 1.0 since it's a quick health check.
pub(crate) fn inspection_cap_multiplier(intent: super::types::ReviewIntent) -> f64 {
    match intent {
        super::types::ReviewIntent::Review => 1.5,
        super::types::ReviewIntent::Security => 1.5,
        super::types::ReviewIntent::Gaps => 1.25,
        super::types::ReviewIntent::Roadmap => 1.25,
        super::types::ReviewIntent::Status => 1.0,
    }
}

/// Project-size-aware ceiling for the inspection cap. Small projects get a
/// lower ceiling (no need for 100 reads on a 10-file repo); large repos get a
/// higher upper bound. The ceiling is applied *after* the task multiplier but
/// *before* the soft-cap extension.
///
/// `indexed_file_count` is the number of source files the repo-intelligence
/// indexer found (0 when indexing is unavailable or the project is empty).
pub(crate) fn inspection_cap_project_ceiling(indexed_file_count: u32) -> u32 {
    if indexed_file_count == 0 {
        // Unknown size — be generous so we don't starve legitimate review.
        120
    } else if indexed_file_count < 50 {
        40
    } else if indexed_file_count < 200 {
        80
    } else if indexed_file_count < 1000 {
        120
    } else {
        200
    }
}

/// How many *additional* inspection attempts the soft-cap extension grants
/// when the agent requests more budget with justification. The extension is
/// granted in chunks so the agent must re-justify if it still needs more.
pub(crate) const SOFT_CAP_EXTENSION_GRANT: u32 = 20;

/// Maximum number of soft-cap extension grants per turn. After this many
/// extensions the turn must answer from gathered evidence.
pub(crate) const MAX_SOFT_CAP_EXTENSIONS: u32 = 3;

/// Weight applied to context-efficient tools when counting inspection attempts.
/// `explore`, `repo_map`, and `find_symbol` aggregate many files into a concise
/// summary, so they cost less against the cap than a direct `read` or `grep`.
/// A weight of 4 means 4 such calls count as 1 inspection attempt.
pub(crate) const CONTEXT_EFFICIENT_TOOL_WEIGHT: u32 = 4;

/// A mutation-capable turn may inspect a bounded amount of evidence before it
/// must attempt the requested edit. This protects against models that keep
/// reading/planning indefinitely while repeatedly promising to act.
pub(crate) const MUTATION_DISCOVERY_ROUND_CAP: u32 = 10;
pub(crate) const MUTATION_DISCOVERY_ROUNDS_PER_NUDGE: u32 = 2;
pub(crate) const MAX_MUTATION_DISCOVERY_NUDGES: u32 = 2;

/// Sent when the model re-reads files it already inspected earlier this turn
/// (a multi-step read cycle like A→B→C→A→B→C that evades the exact-match
/// repeat guard). The file contents are already in the transcript above —
/// re-reading will only reproduce them. Nudges the model to act on what it
/// already has instead of cycling until the step cap.
pub(crate) const REREAD_NUDGE: &str = "You already read these files earlier this turn and their contents \
are already in the conversation above — reading them again will only repeat the same output. Act on \
that output now: make the edit it points to, move to the next step, or if the task is already complete, \
stop and give your final recap. Do not re-read files you have already inspected.";
/// Sent when a wait-and-check poll ("sleep 300 && du …") returns byte-identical
/// output to an earlier poll: whatever the model is waiting on has stopped
/// changing, so blind re-polling is no longer progress. Points the model at
/// diagnosing the stalled process instead of quitting or looping.
pub(crate) const WAIT_POLL_STATIC_NUDGE: &str = "Your wait-and-check command returned exactly the same \
output as before — whatever you are waiting on has not progressed since the last check. Do not simply \
re-run the same poll. Check the underlying process directly (bash_output on its handle, its log file, or \
the process list), fix what is stuck if you can, or if the wait is genuinely still in progress use a much \
longer interval. If you cannot make progress now, stop and report the current state and what remains.";
pub(crate) const SECURITY_BROAD_SEARCH_NUDGE: &str = "This security review searched and read some evidence, \
but it has not covered all required pattern families yet. Do not use mutating tools. Search for \
unsafe/unwrap/expect/panic, command execution/filesystem/env access, and secret/token/auth \
patterns, then answer only from concrete inspected evidence with a Limits section for unsearched \
areas.";
pub(crate) const SECURITY_SCOPE_NUDGE: &str = "The security answer made repo-wide all-clear claims that are \
broader than the inspected files and search results support. Do not use mutating tools. Answer \
again with findings explicitly bounded to the searched patterns and inspected files, and name any \
broader security claims that remain unverified.";
pub(crate) const GAP_SEARCH_OVERCLAIM_NUDGE: &str = "The gap or roadmap answer claimed there were no \
TODO/FIXME/missing gaps even though the targeted search returned matches. Do not use mutating \
tools. Answer again from the inspected files and search matches, with Limits for broader roadmap \
claims.";
pub(crate) const SECURITY_PREFLIGHT_PATTERN: &str = "unsafe|unwrap\\(|expect\\(|panic!|std::process|process::Command|Command::new|spawn\\(|std::fs|fs::|read_to_string|std::env|env::|secret|token|auth|api_key|apikey|password|credential|bearer";
pub(crate) const GAP_PREFLIGHT_PATTERN: &str =
    "TODO|FIXME|todo!|unimplemented!|missing|gap|needs coverage|not implemented";
pub(crate) const IMPLEMENTATION_NO_CHANGES_NUDGE: &str = "This is an implementation request, but no \
successful file changes are in the transcript yet. Do not finalize. Inspect the workspace if \
needed, then create or edit the necessary files with write/edit/multi_edit/apply_patch or a \
project-local scaffold command.";
pub(crate) const IMPLEMENTATION_MISSING_VALIDATION_NUDGE: &str = "Files changed for this implementation \
request, but no successful noninteractive validation command ran after the last change. Do not \
finalize. Run the detected build/test/check command now, then finish with changed files and the \
validation command.";
pub(crate) const IMPLEMENTATION_SCAFFOLD_ONLY_NUDGE: &str = "This implementation request has only scaffold \
or dependency/setup changes so far. Do not finalize yet. Edit the actual source/config files that \
implement the requested behavior, then run validation after the final edit.";
pub(crate) const IMPLEMENTATION_EMPTY_TUI_NUDGE: &str = "The implementation preflight found no project \
manifest. This is a TUI request, so scaffold the Rust binary in the current directory now with \
`cargo init --bin .`, then add Ratatui/Crossterm, implement the requested behavior, and validate with \
`cargo test` or `cargo check`.";
pub(crate) const POST_TOOL_EMPTY_RESPONSE_NUDGE: &str = "The previous model response after the tool \
results was empty. Continue from the returned tool output now. If more workspace inspection is \
needed, use the available tools; otherwise answer or implement the next concrete step. Do not \
repeat the same read-only calls unless their prior output lacks the needed details.";
pub(crate) const TOOL_PROTOCOL_RETRY_NUDGE: &str = "The previous response was rejected by the provider \
because it was not a valid tool turn. Continue using exactly valid tool calls from the available \
schemas. For multi-line file creation, prefer `apply_patch` with `*** Add File` hunks, or call \
`write` with JSON arguments containing `path` and `content`. For shell commands, call `bash` with \
a JSON `command`. Do not put malformed JSON, markdown fences, or prose inside a tool call.";
pub(crate) const TOOL_PROTOCOL_TEXT_FALLBACK_NUDGE: &str = "Structured tool calls have been rejected \
repeatedly by the provider. For this next response only, do not use provider/function tool calling. \
Emit exactly one plain-text tool call in this XML-ish format and no markdown fences:\n\
<tool_call>write<arg_key>path</arg_key><arg_value>src/main.rs</arg_value><arg_key>content</arg_key><arg_value>file contents here</arg_value></tool_call>\n\
For shell commands use:\n\
<tool_call>bash<arg_key>command</arg_key><arg_value>cargo test</arg_value></tool_call>\n\
Keep the edit compact; a minimal working vertical slice is better than a huge invalid tool call.";
