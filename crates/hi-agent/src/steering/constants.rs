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

/// Synthetic tool result when the model pages a file that was already returned
/// in full. This is **not** an identical call (offsets differ), so telling the
/// model "this call is identical" makes weak models argue with the tool layer
/// instead of implementing. Live: DeepSeek Flash read SPEC.md, then tried
/// offset pages and spent the next round reasoning that dedup was broken.
pub(crate) const SKIPPED_COMPLETED_FILE_REREAD_RESULT: &str = "[not executed: this file was already \
returned in full earlier this turn — there is no paging footer, so you already have every line. \
The contents are in the conversation above. Do not re-read or page it. Use those contents and \
continue the task (create/edit files, run commands).]";

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
/// In-turn elision occupancy for classified read-only reviews. 12k fired as
/// soon as the system prompt + skill pack landed, stubbing every tool result
/// and sending the model hunting for evidence it no longer had (live: 45+
/// serial RSI rounds on a 403-notice review). 24k still stubs a long inspect
/// loop; recent results stay verbatim via [`crate::IN_TURN_KEEP_TOOL_RESULTS`].
pub(crate) const READ_ONLY_SAFE_CONTEXT_WINDOW: u32 = 24_000;
/// In-turn elision cap for mutation/coding loops. Without this, a 200k catalog
/// window delays stubbing old tool results until occupancy is huge, so a long
/// edit loop resends every payload. Recent results stay verbatim via
/// [`crate::IN_TURN_KEEP_TOOL_RESULTS`].
pub(crate) const MUTATION_SAFE_CONTEXT_WINDOW: u32 = 32_000;
pub(crate) const READ_ONLY_PREFLIGHT_GREP_MAX_LINES: usize = 32;
pub(crate) const READ_ONLY_PREFLIGHT_DIFF_MAX_LINES: usize = 160;
pub(crate) const SECURITY_PREFLIGHT_EXTRA_READ_LIMIT: u32 = 90;
pub(crate) const DEFAULT_PREFLIGHT_EXTRA_READ_LIMIT: u32 = 120;
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

/// Sent when the model re-reads files it already inspected earlier this turn
/// (a multi-step read cycle like A→B→C→A→B→C that evades the exact-match
/// repeat guard). The file contents are already in the transcript above —
/// re-reading will only reproduce them. Nudges the model to act on what it
/// already has instead of cycling indefinitely.
pub(crate) const REREAD_NUDGE: &str = "You already read these files earlier this turn and their contents \
are already in the conversation above — reading them again will only repeat the same output. If the \
first read had no \"read more with offset N\" footer, you already have the entire file; do not page it. \
Act on that output now: make the edit it points to, move to the next step, or if the task is already complete, \
stop and give your final recap. Do not re-read files you have already inspected.";
/// Sent when a wait-and-check poll ("sleep 300 && du …") returns byte-identical
/// output to an earlier poll: whatever the model is waiting on has stopped
/// changing, so blind re-polling is no longer progress. Points the model at
/// diagnosing the stalled process instead of quitting or looping.
pub(crate) const WAIT_POLL_STATIC_NUDGE: &str = "Your wait-and-check command returned exactly the same \
output as before — whatever you are waiting on has not progressed since the last check. Do not simply \
re-run the same poll. Check the underlying process directly (bash_output on its handle — pass wait_secs \
to block for new output instead of re-polling — its log file, or the process list), fix what is stuck if \
you can, or if the wait is genuinely still in progress use a much longer interval. If you cannot make \
progress now, stop and report the current state and what remains.";
/// Sent when the turn has spent its waiting budget: several consecutive tool
/// rounds did nothing but watch still-running background work (with or without
/// fresh output — a live progress bar makes every poll look new). Babysitting a
/// long process one model round at a time is the most expensive failure mode
/// observed in real transcripts (hundreds of rounds re-polling two downloads).
/// Steer the model to either block once server-side or end the turn honestly.
pub(crate) const BACKGROUND_WAIT_STATUS_NUDGE: &str = "The background process is still running. Stop \
polling it round after round. If it should produce output or finish within a few minutes, make ONE \
bash_output call with wait_secs (up to 600) to block until then. Otherwise stop now and give a concise \
final status: the work remains in progress, what has been completed so far, and what remains once it \
finishes. Do not claim completion or failure, and do not keep watching a process that will run for a \
long time.";
/// Sent when the model keeps polling after [`BACKGROUND_WAIT_STATUS_NUDGE`] —
/// the next round is forced tool-free so the status answer actually lands.
pub(crate) const BACKGROUND_WAIT_FINAL_NUDGE: &str = "The background process is still running and you \
were already asked to stop polling it. Give your final status answer now: state that the work remains \
in progress, what has been completed so far, and what remains. Do not call any tools and do not claim \
completion or failure.";
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
successful file changes are in the transcript yet. Do not finalize a diagnosis. Inspect the \
workspace if needed, then create or edit the necessary files with write/edit/multi_edit/apply_patch \
or a project-local scaffold command. If after inspection the task genuinely requires no edits, \
state plainly that no file changes are needed and explain why.";
pub(crate) const IMPLEMENTATION_MISSING_VALIDATION_NUDGE: &str = "Files changed for this implementation \
request, but no successful noninteractive validation command ran after the last change. Do not \
finalize. Run the detected build/test/check command now, then finish with changed files and the \
validation command.";
pub(crate) const REQUESTED_VALIDATION_NUDGE: &str = "The user explicitly asked you to run validation, \
but no successful build/test/check command is in the transcript. Run the requested command now and \
report its actual result. Do not claim completion without tool evidence.";
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
pub(crate) fn tool_protocol_retry_nudge(
    tools: &[hi_ai::ToolSpec],
    tool_mode: hi_ai::ToolMode,
) -> String {
    let mut names = tools
        .iter()
        .filter(|_| !matches!(tool_mode, hi_ai::ToolMode::ChatOnly))
        .map(|tool| tool.name.as_str())
        .collect::<Vec<_>>();
    names.sort_unstable();
    names.dedup();

    let availability = if names.is_empty() {
        "No tools are available in this request. Answer in plain text without making a tool call."
            .to_string()
    } else {
        format!(
            "The only available tool names for this retry are: {}. If none fits, answer in plain text.",
            names
                .iter()
                .map(|name| format!("`{name}`"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };

    format!(
        "The previous response was rejected by the provider because it was not a valid tool turn. \
Use only tools declared in the current request; a tool remembered from an earlier turn may not be \
available now. {availability} Follow the selected tool's exact JSON schema. Do not use an absent \
tool name, malformed JSON, markdown fences, or prose inside a tool call."
    )
}

pub(crate) fn tool_validation_retry_nudge(tool: &str, error: &str) -> String {
    format!(
        "The `{tool}` tool call was rejected by client-side schema validation: {error}. Emit a new `{tool}` call with a complete JSON object that satisfies the declared schema. Do not repeat the invalid arguments; include every required property and use the correct JSON types."
    )
}

pub(crate) fn unavailable_tool_retry_nudge(
    rejected_tools: &[&str],
    admitted_tools: &[String],
) -> String {
    let rejected = rejected_tools
        .iter()
        .map(|name| format!("`{name}`"))
        .collect::<Vec<_>>()
        .join(", ");
    let availability = admitted_tool_names(admitted_tools.iter().map(String::as_str));
    format!(
        "The previous call used {rejected}, which is not admitted by this request's sealed tool \
envelope. Changing its arguments or spelling it in XML/plain text cannot grant access. {availability} \
Use one of those admitted tools with its exact JSON schema. If none can perform the task, answer \
plainly with the limitation instead of emitting another unavailable tool."
    )
}

pub(crate) fn tool_protocol_text_fallback_nudge(
    tools: &[hi_ai::ToolSpec],
    tool_mode: hi_ai::ToolMode,
) -> String {
    if matches!(tool_mode, hi_ai::ToolMode::ChatOnly) || tools.is_empty() {
        return "The sealed envelope for this request admits no executable tools. Do not emit a \
structured, JSON, or XML/plain-text tool call; answer in plain text instead."
            .to_string();
    }
    let availability = admitted_tool_names(tools.iter().map(|tool| tool.name.as_str()));
    let example = xmlish_admitted_tool_example(tools).unwrap_or_else(|| {
        "Use `<tool_call>ADMITTED_TOOL_NAME` followed by complete `<arg_key>` / `<arg_value>` pairs from that tool's schema and `</tool_call>`.".to_string()
    });
    format!(
        "Structured tool calling did not produce an executable tool call in the previous attempts. \
For this response only, emit exactly one plain-text call using a tool from the current sealed \
envelope and no markdown fences. {availability} Here is the XML-ish shape using one of those \
admitted schemas; replace only the example values with the real arguments you need:\n{example}\n\
Do not name any tool outside the admitted list. If none fits, answer in plain text."
    )
}

fn xmlish_admitted_tool_example(tools: &[hi_ai::ToolSpec]) -> Option<String> {
    const PREFERRED: &[&str] = &[
        "apply_patch",
        "edit",
        "write",
        "bash",
        "read",
        "grep",
        "list",
    ];
    let tool = PREFERRED
        .iter()
        .find_map(|name| tools.iter().find(|tool| tool.name == *name))
        .or_else(|| tools.first())?;
    let required = tool
        .parameters
        .get("required")
        .and_then(|value| value.as_array())
        .or_else(|| {
            tool.parameters
                .get("oneOf")?
                .as_array()?
                .first()?
                .get("required")?
                .as_array()
        })?;
    let fields = required
        .iter()
        .filter_map(|value| value.as_str())
        .filter(|key| {
            tool.parameters["properties"][*key]["type"]
                .as_str()
                .is_some_and(|kind| kind == "string")
        })
        .collect::<Vec<_>>();
    if fields.len() != required.len() || fields.is_empty() {
        return None;
    }
    let args = fields
        .iter()
        .map(|key| {
            let value = match *key {
                "path" => "path/to/file",
                "command" => "command to run",
                "pattern" => "search pattern",
                "patch" => "patch text",
                "old_string" => "existing text",
                "new_string" | "content" => "replacement text",
                _ => "value",
            };
            format!("<arg_key>{key}</arg_key><arg_value>{value}</arg_value>")
        })
        .collect::<String>();
    Some(format!("<tool_call>{}{args}</tool_call>", tool.name))
}

fn admitted_tool_names<'a>(names: impl IntoIterator<Item = &'a str>) -> String {
    let mut names = names.into_iter().collect::<Vec<_>>();
    if names.is_empty() {
        return "No tools are admitted in this request.".to_string();
    }
    names.sort_unstable();
    names.dedup();
    format!(
        "The only admitted tool names are: {}.",
        names
            .iter()
            .map(|name| format!("`{name}`"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

#[cfg(test)]
mod protocol_retry_tests {
    use super::{
        tool_protocol_retry_nudge, tool_protocol_text_fallback_nudge, tool_validation_retry_nudge,
        unavailable_tool_retry_nudge,
    };
    use hi_ai::{ToolMode, ToolSpec};

    fn tool(name: &str) -> ToolSpec {
        let field = match name {
            "read" => "path",
            "grep" => "pattern",
            _ => "value",
        };
        ToolSpec {
            name: name.to_string(),
            description: String::new(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {(field): {"type": "string"}},
                "required": [field]
            }),
        }
    }

    #[test]
    fn protocol_retry_guidance_only_names_tools_in_the_request() {
        let nudge = tool_protocol_retry_nudge(&[tool("read"), tool("grep")], ToolMode::ReadOnly);

        assert!(nudge.contains("`grep`, `read`"));
        assert!(!nudge.contains("`bash`"));
        assert!(!nudge.contains("`write`"));
        assert!(nudge.contains("tool remembered from an earlier turn"));
    }

    #[test]
    fn protocol_retry_guidance_for_tool_free_requests_requires_text() {
        let nudge = tool_protocol_retry_nudge(&[tool("read")], ToolMode::ChatOnly);

        assert!(nudge.contains("No tools are available"));
        assert!(nudge.contains("plain text"));
        assert!(!nudge.contains("`bash`"));
    }

    #[test]
    fn validation_retry_guidance_names_the_rejected_tool_and_schema_error() {
        let nudge = tool_validation_retry_nudge(
            "read",
            "invalid tool arguments: 'path' is a required property",
        );
        assert!(nudge.contains("`read`"));
        assert!(nudge.contains("'path' is a required property"));
        assert!(nudge.contains("complete JSON object"));
    }

    #[test]
    fn unavailable_tool_guidance_does_not_prescribe_the_rejected_tool() {
        let nudge = unavailable_tool_retry_nudge(&["bash"], &["read".to_string()]);

        assert!(nudge.contains("`bash`"));
        assert!(nudge.contains("not admitted"));
        assert!(nudge.contains("only admitted tool names are: `read`"));
        assert!(!nudge.contains("new `bash` call"));
        assert!(nudge.contains("cannot grant access"));
    }

    #[test]
    fn text_fallback_is_derived_from_read_only_admitted_tools() {
        let nudge =
            tool_protocol_text_fallback_nudge(&[tool("read"), tool("grep")], ToolMode::ReadOnly);

        assert!(nudge.contains("`grep`, `read`"));
        assert!(!nudge.contains("`bash`"));
        assert!(!nudge.contains("`write`"));
        assert!(nudge.contains("<tool_call>read<arg_key>"));
    }

    #[test]
    fn text_fallback_for_chat_only_never_suggests_a_call() {
        let nudge = tool_protocol_text_fallback_nudge(&[tool("read")], ToolMode::ChatOnly);

        assert!(nudge.contains("admits no executable tools"));
        assert!(nudge.contains("answer in plain text"));
        assert!(!nudge.contains("<tool_call>"));
    }
}
