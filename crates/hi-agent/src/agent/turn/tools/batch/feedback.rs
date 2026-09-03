//! Post-edit proactive checks and affected-package fast feedback.

use std::collections::BTreeSet;

use crate::agent::turn::fast_feedback::{
    FastFeedbackOptions, FastFeedbackState, run_fast_feedback, signature_impact_notes,
};
use crate::steering::ImplementationTracker;
use crate::transcript::NudgeKind;
use crate::{TaskContract, Ui};

/// Proactive checks run concurrently with the rest of a tool batch, but they
/// remain owned by that turn. A plain `JoinHandle` detaches on drop; wrapping it
/// makes cancellation abort the task, which drops the process future and lets
/// `ProcessRunner` kill the check's complete process group.
pub(super) type PendingCheck = (
    String,
    String,
    tokio_util::task::AbortOnDropHandle<(bool, String)>,
);

#[allow(clippy::too_many_arguments)]
pub(super) async fn append_fast_feedback(
    agent: &mut crate::Agent,
    calls: &[(String, String, String)],
    pending_checks: Vec<PendingCheck>,
    batch_mutated_paths: BTreeSet<String>,
    task_contract: &TaskContract,
    fast_feedback: &mut FastFeedbackState,
    implementation_tracker: &mut ImplementationTracker,
    results: &mut [(String, String)],
    ui: &mut dyn Ui,
) {
    // Await the proactive per-edit checks kicked off during the
    // batch. A syntax/lint error appears here, during the turn, before
    // turn-end verify. Keep successful checks in the next model-facing
    // tool result too: otherwise reasoning models often run a duplicate
    // shell validation even though the result is already available.
    let mut proactive_failures = Vec::new();
    let mut proactive_passes = Vec::new();
    for (path, check, handle) in pending_checks {
        if let Ok((passed, output)) = handle.await {
            if passed {
                proactive_passes.push(format!("✓ fast check passed for {path} ({check})"));
                continue;
            }
            let msg = format!("⚠ proactive check failed for {path}:\n{output}");
            ui.status(&msg);
            proactive_failures.push(msg);
        }
    }
    // Mid-turn Rust fast path: LSP → affected cargo check → (if
    // test-gated) affected cargo test. Failures append to tool results.
    let mut fast_failures = Vec::new();
    if !batch_mutated_paths.is_empty() {
        let paths = batch_mutated_paths.into_iter().collect::<Vec<_>>();
        let run_tests = task_contract.wants_tests
            || agent
                .task
                .last_task_contract
                .as_ref()
                .is_some_and(|c| c.wants_tests);
        let report = run_fast_feedback(
            &agent.runtime,
            &paths,
            fast_feedback,
            FastFeedbackOptions { run_tests },
            ui,
        )
        .await;
        if report.tests_ran && !report.tests_failed && !report.tests_timed_out {
            implementation_tracker.record_validation_success();
        }
        if let Some(text) = report.combined_feedback() {
            fast_failures.push(text);
        }
        // Edits that landed on a definition line get a reverse-reference
        // note: the model updates the callers before the compiler starts
        // reporting them one at a time.
        let edited_regions: Vec<(String, String)> = calls
            .iter()
            .filter_map(|(_, name, arguments)| {
                let args: serde_json::Value = serde_json::from_str(arguments).ok()?;
                let path = args.get("path")?.as_str()?.to_string();
                let mut region = String::new();
                match name.as_str() {
                    "edit" => {
                        region.push_str(args.get("old_string")?.as_str()?);
                    }
                    "multi_edit" => {
                        for edit in args.get("edits")?.as_array()? {
                            if let Some(old) = edit.get("old_string").and_then(|v| v.as_str()) {
                                region.push_str(old);
                                region.push('\n');
                            }
                        }
                    }
                    _ => return None,
                }
                (!region.is_empty()).then_some((path, region))
            })
            .collect();
        if !edited_regions.is_empty() {
            fast_failures.extend(signature_impact_notes(&agent.runtime, &edited_regions).await);
        }
    }
    // Append failures onto the last mutating tool result so the model
    // sees them in the transcript before the next reasoning step.
    let mut feedback_blocks = proactive_passes;
    feedback_blocks.extend(proactive_failures);
    feedback_blocks.extend(fast_failures);
    if !feedback_blocks.is_empty() {
        let block = feedback_blocks.join("\n\n");
        if let Some((_, content)) = results.iter_mut().rev().find(|(id, _)| {
            // Prefer a result that came from a filesystem mutation if we can
            // spot one by matching call ids in this batch.
            calls.iter().any(|(call_id, name, _)| {
                call_id == id && (hi_tools::is_filesystem_mutating(name) || name == "bash")
            })
        }) {
            if !content.ends_with('\n') {
                content.push('\n');
            }
            content.push('\n');
            content.push_str(&block);
        } else if let Some((_, content)) = results.last_mut() {
            if !content.ends_with('\n') {
                content.push('\n');
            }
            content.push('\n');
            content.push_str(&block);
        } else {
            // No tool results (shouldn't happen for a mutation batch) —
            // still push a nudge so the model is not blind.
            agent.messages.push_nudge(
        NudgeKind::Continue,
        format!(
            "Fast check found problems after your last edits — fix these before continuing:\n{block}"
        ),
    );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PendingCheck;

    #[tokio::test]
    async fn dropping_pending_check_aborts_its_owned_task() {
        struct Dropped(std::sync::Arc<std::sync::atomic::AtomicBool>);
        impl Drop for Dropped {
            fn drop(&mut self) {
                self.0.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }

        let dropped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let task_dropped = dropped.clone();
        let handle = tokio::spawn(async move {
            let _guard = Dropped(task_dropped);
            std::future::pending::<()>().await;
            (true, String::new())
        });
        tokio::task::yield_now().await;
        let pending: PendingCheck = (
            "src/lib.rs".into(),
            "check".into(),
            tokio_util::task::AbortOnDropHandle::new(handle),
        );

        drop(pending);
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while !dropped.load(std::sync::atomic::Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("aborting the pending check should promptly drop its task");
    }
}
