//! Synthetic denial messages and waiting/dry-run classification.

use crate::{ConfirmationResult, PARKED_TOOL_RESULT};

#[cfg(test)]
use crate::agent::turn::helpers::synthetic_tool_outcome;

pub(super) fn workspace_mutation_intent(
    calls: &[(String, String, String)],
    dirty_paths: Option<Vec<String>>,
) -> hi_workspace::MutationIntent {
    let policies = calls
        .iter()
        .map(|(_, name, arguments)| concrete_policy(name, arguments))
        .collect::<Vec<_>>();
    let replay_class = combined_replay_class(policies.iter().map(|policy| policy.replay_class));
    hi_workspace::MutationIntent {
        effect_scope: combined_effect_scope(policies.iter().map(|policy| policy.effect_scope)),
        replay_class,
        dirty_paths: dirty_paths.map(|paths| paths.into_iter().map(Into::into).collect()),
        description: Some(format!(
            "tool batch: {}",
            calls
                .iter()
                .map(|(_, name, _)| name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

pub(super) fn workspace_program_intent(
    name: &str,
    arguments: &str,
    dirty_paths: Option<Vec<String>>,
) -> hi_workspace::MutationIntent {
    let policy = concrete_policy(name, arguments);
    hi_workspace::MutationIntent {
        effect_scope: policy.effect_scope,
        replay_class: combined_replay_class(std::iter::once(policy.replay_class)),
        dirty_paths: dirty_paths.map(|paths| paths.into_iter().map(Into::into).collect()),
        description: Some(format!("program tool: {name}")),
    }
}

pub(super) fn workspace_operation_requires_settlement(name: &str, arguments: &str) -> bool {
    if name == "bash_kill" {
        // This controls a writer admitted under an existing JobPermit. Kill
        // and reap it first; its terminal callback unlocks reconciliation.
        return false;
    }
    let policy = concrete_policy(name, arguments);
    policy.effect_scope != hi_workspace::EffectScope::ReadOnly
        || policy.replay_class != hi_tools::catalog::ReplayClass::PureWorkspace
}

/// A terminal poll is the publication boundary for a managed live writer.
/// Its lifecycle callback has already moved the matching workspace job to
/// `DurabilityPending`, so the poll must obtain a reconciliation receipt before
/// its result can enter the provider transcript.
pub(super) fn terminal_background_requires_reconciliation(
    entries: &[crate::ToolCallEntry],
    pending: &[hi_tools::BackgroundJobId],
) -> bool {
    entries.iter().any(|entry| {
        entry.background.as_ref().is_some_and(|background| {
            matches!(
                background.state,
                hi_tools::BackgroundState::Exited
                    | hi_tools::BackgroundState::Killed
                    | hi_tools::BackgroundState::Failed
            ) && pending.iter().any(|job| job.handle == background.id)
        })
    })
}

/// Convert the executor's typed per-call outcomes into the operation report
/// consumed by the workspace controller. Durability code must never invent a
/// successful execution merely because its own scan/commit succeeded.
pub(super) fn workspace_execution_report(
    intent: &hi_workspace::MutationIntent,
    entries: &[crate::ToolCallEntry],
    expected_calls: usize,
) -> hi_workspace::ExecutionReport {
    let complete = entries.len() == expected_calls;
    let disposition = if !complete {
        hi_workspace::ExecutionDisposition::Indeterminate
    } else if entries
        .iter()
        .any(|entry| entry.status == hi_tools::ToolStatus::Cancelled)
    {
        hi_workspace::ExecutionDisposition::Cancelled
    } else if entries
        .iter()
        .any(|entry| entry.status != hi_tools::ToolStatus::Succeeded)
    {
        hi_workspace::ExecutionDisposition::Failed
    } else {
        hi_workspace::ExecutionDisposition::Succeeded
    };
    let live_process_observed = entries.iter().any(|entry| {
        entry.background.as_ref().is_some_and(|background| {
            matches!(
                background.state,
                hi_tools::BackgroundState::Started | hi_tools::BackgroundState::Running
            )
        })
    });
    let workspace_may_have_changed = entries.iter().any(|entry| entry.effects.mutation_applied)
        || live_process_observed
        || (!complete && intent.effect_scope == hi_workspace::EffectScope::LiveWriter);
    let external_effect_may_have_occurred = intent.replay_class
        != hi_workspace::ReplayClass::PureWorkspace
        && (!complete
            || entries
                .iter()
                .any(|entry| entry.status != hi_tools::ToolStatus::Denied));
    let mut changed_paths = entries
        .iter()
        .flat_map(|entry| entry.effects.file_changes.iter())
        .map(|change| std::path::PathBuf::from(&change.path))
        .collect::<Vec<_>>();
    changed_paths.sort();
    changed_paths.dedup();
    let detail = match disposition {
        hi_workspace::ExecutionDisposition::Succeeded => None,
        hi_workspace::ExecutionDisposition::Indeterminate => Some(format!(
            "executor retained {} of {expected_calls} typed tool outcomes",
            entries.len()
        )),
        other => {
            let failed = entries
                .iter()
                .filter(|entry| entry.status != hi_tools::ToolStatus::Succeeded)
                .map(|entry| format!("{}:{:?}", entry.tool, entry.status))
                .collect::<Vec<_>>()
                .join(", ");
            Some(format!("tool batch execution was {other:?}: {failed}"))
        }
    };
    hi_workspace::ExecutionReport {
        disposition,
        workspace_may_have_changed,
        external_effect_may_have_occurred,
        content_digest: None,
        changed_paths,
        artifacts: Vec::new(),
        detail,
    }
}

/// Preserve the restricted program host's real terminal result while keeping
/// its dynamically selected nested calls behind one conservative operation.
/// `effect_may_have_occurred` is set by the host as soon as it dispatches any
/// nested call whose policy requires settlement.
pub(super) fn workspace_program_execution_report(
    intent: &hi_workspace::MutationIntent,
    outcome: &hi_workflow::ProgramOutcome,
    effect_may_have_occurred: bool,
) -> hi_workspace::ExecutionReport {
    let disposition = match outcome {
        hi_workflow::ProgramOutcome::Cancelled { .. } => {
            hi_workspace::ExecutionDisposition::Cancelled
        }
        hi_workflow::ProgramOutcome::Failed { .. } => hi_workspace::ExecutionDisposition::Failed,
        hi_workflow::ProgramOutcome::Succeeded { calls, .. }
            if calls.iter().any(|call| call.status == "cancelled") =>
        {
            hi_workspace::ExecutionDisposition::Cancelled
        }
        hi_workflow::ProgramOutcome::Succeeded { calls, .. }
            if calls.iter().any(|call| call.status != "succeeded") =>
        {
            hi_workspace::ExecutionDisposition::Failed
        }
        hi_workflow::ProgramOutcome::Succeeded { .. } => {
            hi_workspace::ExecutionDisposition::Succeeded
        }
    };
    let detail = match outcome {
        hi_workflow::ProgramOutcome::Failed { error, .. } => Some(error.clone()),
        hi_workflow::ProgramOutcome::Cancelled { .. } => Some("program cancelled".into()),
        hi_workflow::ProgramOutcome::Succeeded { .. }
            if disposition != hi_workspace::ExecutionDisposition::Succeeded =>
        {
            Some(format!("nested program execution was {disposition:?}"))
        }
        hi_workflow::ProgramOutcome::Succeeded { .. } => None,
    };
    hi_workspace::ExecutionReport {
        disposition,
        workspace_may_have_changed: effect_may_have_occurred
            && intent.effect_scope == hi_workspace::EffectScope::LiveWriter,
        external_effect_may_have_occurred: effect_may_have_occurred
            && intent.replay_class != hi_workspace::ReplayClass::PureWorkspace,
        content_digest: None,
        changed_paths: Vec::new(),
        artifacts: Vec::new(),
        detail,
    }
}

fn concrete_policy(name: &str, arguments: &str) -> hi_tools::catalog::ToolPolicy {
    if name == "bash" {
        return hi_tools::protocol::classify_shell_tool_arguments(arguments).policy;
    }
    let mut policy = hi_tools::tool_metadata(name)
        .map(|metadata| metadata.policy)
        .unwrap_or_else(hi_tools::catalog::ToolPolicy::conservative);
    // A synchronous delegate prepares in isolation, then the parent applies
    // its candidate before returning the tool result.
    if name == "delegate" {
        policy.effect_scope = hi_workspace::EffectScope::LiveWriter;
    }
    policy
}

fn combined_replay_class(
    classes: impl IntoIterator<Item = hi_tools::catalog::ReplayClass>,
) -> hi_workspace::ReplayClass {
    let mut idempotent_external = false;
    for class in classes {
        match class {
            hi_tools::catalog::ReplayClass::NonReplayableExternal => {
                return hi_workspace::ReplayClass::NonReplayableExternal;
            }
            hi_tools::catalog::ReplayClass::IdempotentExternal => idempotent_external = true,
            hi_tools::catalog::ReplayClass::PureWorkspace => {}
        }
    }
    if idempotent_external {
        hi_workspace::ReplayClass::IdempotentExternal {
            key: hi_workspace::IdempotencyKey::new(uuid::Uuid::new_v4().to_string()),
        }
    } else {
        hi_workspace::ReplayClass::PureWorkspace
    }
}

fn combined_effect_scope(
    scopes: impl IntoIterator<Item = hi_workspace::EffectScope>,
) -> hi_workspace::EffectScope {
    let mut candidate = false;
    for scope in scopes {
        match scope {
            hi_workspace::EffectScope::LiveWriter => return hi_workspace::EffectScope::LiveWriter,
            hi_workspace::EffectScope::CandidateOnly => candidate = true,
            hi_workspace::EffectScope::ReadOnly => {}
        }
    }
    if candidate {
        hi_workspace::EffectScope::CandidateOnly
    } else {
        hi_workspace::EffectScope::ReadOnly
    }
}

/// Whether a call is compatible with a turn that is merely waiting on live
/// background work: the poll itself, plan bookkeeping, and non-mutating,
/// non-validating shell probes (log tails, size checks, process listings).
/// Real work — edits, builds, tests, file reads — is deliberately not
/// wait-flavored, so interleaved genuine progress resets the waiting streak.
pub(super) fn wait_flavored_call(
    name: &str,
    arguments: &str,
    output: &hi_tools::ToolOutcome,
) -> bool {
    match name {
        "bash_output" | "update_plan" => true,
        "bash" => {
            !output.effects.mutation_applied
                && !crate::steering::implementation_tool_call_validates(name, arguments)
        }
        _ => false,
    }
}

/// Human-readable "planned action" line for a dry-run tool call. Pure so the
/// dry-run path is unit-testable without a live agent/runtime.
pub(super) fn dry_run_message(name: &str, path: &str, mutates: bool) -> String {
    let target = if path.is_empty() {
        String::new()
    } else {
        format!(" on {path}")
    };
    let kind = if mutates { "mutating" } else { "read-only" };
    format!("[dry-run] would run `{name}`{target} ({kind}; not executed)")
}

#[cfg(test)]
mod dry_run_tests {
    use super::*;

    #[test]
    fn mutating_call_reports_path_and_mutation() {
        let msg = dry_run_message("edit", "src/main.rs", true);
        assert_eq!(
            msg,
            "[dry-run] would run `edit` on src/main.rs (mutating; not executed)"
        );
    }

    #[test]
    fn read_only_call_reports_read_only() {
        let msg = dry_run_message("read", "src/main.rs", false);
        assert_eq!(
            msg,
            "[dry-run] would run `read` on src/main.rs (read-only; not executed)"
        );
    }

    #[test]
    fn call_without_path_omits_target() {
        let msg = dry_run_message("bash", "", true);
        assert_eq!(msg, "[dry-run] would run `bash` (mutating; not executed)");
    }
}

#[cfg(test)]
mod workspace_policy_tests {
    use super::*;

    fn call(name: &str, arguments: &str) -> (String, String, String) {
        ("call-1".into(), name.into(), arguments.into())
    }

    #[test]
    fn opaque_shell_is_non_replayable_but_static_inspection_is_pure() {
        let opaque = workspace_mutation_intent(
            &[call("bash", r#"{"command":"echo hi > output.txt"}"#)],
            None,
        );
        assert_eq!(
            opaque.replay_class,
            hi_workspace::ReplayClass::NonReplayableExternal
        );

        let inspection =
            workspace_mutation_intent(&[call("bash", r#"{"command":"rg TODO src"}"#)], None);
        assert_eq!(
            inspection.replay_class,
            hi_workspace::ReplayClass::PureWorkspace
        );
        assert!(!workspace_operation_requires_settlement(
            "bash",
            r#"{"command":"rg TODO src"}"#
        ));
        assert!(workspace_operation_requires_settlement(
            "web_search",
            r#"{"query":"release"}"#
        ));
    }

    #[test]
    fn mixed_batch_uses_the_most_conservative_replay_class() {
        let intent = workspace_mutation_intent(
            &[
                call("edit", r#"{"path":"src/lib.rs"}"#),
                call("web_search", r#"{"query":"release"}"#),
                call("unknown_dynamic_tool", "{}"),
            ],
            Some(vec!["src/lib.rs".into()]),
        );
        assert_eq!(
            intent.replay_class,
            hi_workspace::ReplayClass::NonReplayableExternal
        );
        assert_eq!(intent.dirty_paths.unwrap().len(), 1);

        let delegate = workspace_mutation_intent(&[call("delegate", "{}")], None);
        assert_eq!(delegate.effect_scope, hi_workspace::EffectScope::LiveWriter);
    }

    #[test]
    fn kill_reconciles_only_after_its_existing_writer_job_is_reaped() {
        assert!(!workspace_operation_requires_settlement(
            "bash_kill",
            r#"{"id":"writer_1"}"#
        ));
        assert!(workspace_operation_requires_settlement(
            "write",
            r#"{"path":"next.txt","content":"next"}"#
        ));
    }

    fn background_entry(id: &str, state: hi_tools::BackgroundState) -> crate::ToolCallEntry {
        crate::ToolCallEntry {
            tool: "bash_output".into(),
            path: String::new(),
            duration_ms: 0,
            queue_delay_ms: 0,
            completion_index: 1,
            status: hi_tools::ToolStatus::Succeeded,
            background: Some(hi_tools::BackgroundOutcome {
                id: id.into(),
                state,
                exit_code: Some(0),
            }),
            process: None,
            effects: hi_tools::ToolEffects::default(),
            truncation: hi_tools::TruncationState::Complete,
            error: false,
            progress_kind: "none".into(),
            progress_reason: String::new(),
            normalized_signature: None,
            command: None,
            arg_chars: 0,
            result_chars: 0,
            truncated: false,
            kind: "process".into(),
        }
    }

    #[test]
    fn only_a_terminal_pending_writer_poll_requires_reconciliation() {
        let pending = vec![hi_tools::BackgroundJobId {
            source_id: "registry".into(),
            handle: "writer_1".into(),
        }];
        for state in [
            hi_tools::BackgroundState::Exited,
            hi_tools::BackgroundState::Killed,
            hi_tools::BackgroundState::Failed,
        ] {
            assert!(terminal_background_requires_reconciliation(
                &[background_entry("writer_1", state)],
                &pending
            ));
        }
        assert!(!terminal_background_requires_reconciliation(
            &[background_entry(
                "writer_1",
                hi_tools::BackgroundState::Running
            )],
            &pending
        ));
        assert!(!terminal_background_requires_reconciliation(
            &[background_entry(
                "read_only_1",
                hi_tools::BackgroundState::Exited
            )],
            &pending
        ));
    }

    #[test]
    fn program_report_never_rewrites_failure_as_success() {
        let intent = workspace_program_intent("run_program", "{}", None);
        let outcome = hi_workflow::ProgramOutcome::Failed {
            error: "nested command exited 1".into(),
            calls: vec![hi_workflow::ProgramToolResult {
                index: 0,
                name: "bash".into(),
                status: "failed".into(),
                output: "partial output".into(),
            }],
        };
        let report = workspace_program_execution_report(&intent, &outcome, true);

        assert_eq!(
            report.disposition,
            hi_workspace::ExecutionDisposition::Failed
        );
        assert!(report.workspace_may_have_changed);
        assert!(report.external_effect_may_have_occurred);
        assert_eq!(report.detail.as_deref(), Some("nested command exited 1"));
    }

    #[test]
    fn program_report_marks_cancelled_effect_without_claiming_success() {
        let intent = workspace_program_intent("run_program", "{}", None);
        let report = workspace_program_execution_report(
            &intent,
            &hi_workflow::ProgramOutcome::Cancelled { calls: Vec::new() },
            true,
        );

        assert_eq!(
            report.disposition,
            hi_workspace::ExecutionDisposition::Cancelled
        );
        assert!(report.workspace_may_have_changed);
        assert_eq!(report.detail.as_deref(), Some("program cancelled"));
    }
}

#[cfg(test)]
mod wait_flavored_tests {
    use super::*;

    fn outcome(mutation_applied: bool) -> hi_tools::ToolOutcome {
        let mut output = synthetic_tool_outcome("ok".into(), hi_tools::ToolStatus::Succeeded);
        output.effects.mutation_applied = mutation_applied;
        output
    }

    #[test]
    fn polls_bookkeeping_and_status_probes_are_wait_flavored_but_real_work_is_not() {
        assert!(wait_flavored_call(
            "bash_output",
            r#"{"id":"sh_1"}"#,
            &outcome(false)
        ));
        assert!(wait_flavored_call("update_plan", "{}", &outcome(false)));
        assert!(wait_flavored_call(
            "bash",
            r#"{"command":"tail -c 200 /tmp/download.log"}"#,
            &outcome(false)
        ));
        // Real work resets the waiting streak: mutations, validation runs,
        // and file reads are progress, not babysitting.
        assert!(!wait_flavored_call(
            "bash",
            r#"{"command":"echo hi > f.txt"}"#,
            &outcome(true)
        ));
        assert!(!wait_flavored_call(
            "bash",
            r#"{"command":"cargo test"}"#,
            &outcome(false)
        ));
        assert!(!wait_flavored_call(
            "read",
            r#"{"path":"src/main.rs"}"#,
            &outcome(false)
        ));
        assert!(!wait_flavored_call("edit", "{}", &outcome(true)));
    }
}

pub(super) fn parked_or_denied_shell(
    decision: &ConfirmationResult,
) -> (String, hi_tools::ToolStatus) {
    match decision {
        ConfirmationResult::Parked => (
            PARKED_TOOL_RESULT.to_string(),
            hi_tools::ToolStatus::Cancelled,
        ),
        ConfirmationResult::Unavailable => (
            "Shell mutation skipped: confirmation required, but this frontend cannot answer it; rerun interactively or disable --confirm-edits.".into(),
            hi_tools::ToolStatus::Denied,
        ),
        _ => (
            "Shell mutation skipped by user (not run).".into(),
            hi_tools::ToolStatus::Denied,
        ),
    }
}

pub(super) fn parked_or_denied_delegate(
    decision: &ConfirmationResult,
) -> (String, hi_tools::ToolStatus) {
    match decision {
        ConfirmationResult::Parked => (
            PARKED_TOOL_RESULT.to_string(),
            hi_tools::ToolStatus::Cancelled,
        ),
        ConfirmationResult::Unavailable => (
            "Delegate skipped: confirmation required, but this frontend cannot answer it.".into(),
            hi_tools::ToolStatus::Denied,
        ),
        _ => (
            "Delegate skipped by user (no changes applied).".into(),
            hi_tools::ToolStatus::Denied,
        ),
    }
}
