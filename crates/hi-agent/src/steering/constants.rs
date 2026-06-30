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

pub(crate) const NO_EVIDENCE_REVIEW_NUDGE: &str = "This read-only review has no inspected evidence yet. \
Do not finalize. Use read-only inspection tools first, then answer from the inspected evidence. \
If inspection is impossible, explicitly say the evidence is insufficient.";
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
evidence. If deeper inspection is impossible, explicitly say the evidence is insufficient.";
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
tied to inspected paths, or explicitly say the evidence is insufficient.";
pub(crate) const READ_AFTER_SEARCH_NUDGE: &str = "The targeted search result is already in the transcript. \
Do not rerun the same search and do not use mutating tools. Read the most relevant matching file, \
then answer from that inspected file. If you cannot pick a file to read, explicitly say the \
evidence is insufficient.";
pub(crate) const SECURITY_BROAD_SEARCH_NUDGE: &str = "This security review searched and read some evidence, \
but it has not covered all required pattern families yet. Do not use mutating tools. Search for \
unsafe/unwrap/expect/panic, command execution/filesystem/env access, and secret/token/auth \
patterns, then answer only from concrete inspected evidence or explicitly say the evidence is \
insufficient.";
pub(crate) const SECURITY_SCOPE_NUDGE: &str = "The security answer made repo-wide all-clear claims that are \
broader than the inspected files and search results support. Do not use mutating tools. Answer \
again with findings explicitly bounded to the searched patterns and inspected files, or explicitly \
say the evidence is insufficient for broader security claims.";
pub(crate) const GAP_SEARCH_OVERCLAIM_NUDGE: &str = "The gap or roadmap answer claimed there were no \
TODO/FIXME/missing gaps even though the targeted search returned matches. Do not use mutating \
tools. Answer again from the inspected files and search matches, or explicitly say the evidence is \
insufficient for broader roadmap claims.";
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
`cargo init --bin .`, then add Ratatui/Crossterm, implement the estimator, and validate with \
`cargo test` or `cargo check`.";
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

