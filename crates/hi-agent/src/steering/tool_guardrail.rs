use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};

mod test_credit;
mod validation;
pub(crate) use test_credit::tool_result_shows_passing_tests;
pub(crate) use validation::{command_runs_tests, is_validation_command};

use validation::bash_validation_scope;
pub(crate) use validation::{tool_result_hash_guard_applies, validation_exit_status_is_reliable};

/// How many consecutive idle `bash_output` polls (running, no new output) for
/// the same handle are allowed before the result-hash guard treats further
/// polls as no progress. Two free polls keep legitimate progress-watching
/// working; a third identical idle status is a tight loop.
const IDLE_BG_POLL_FREE_STRIKES: u32 = 2;
const IDEMPOTENT_RESULT_HASH_LIMIT: usize = 4_096;
const IDLE_BG_HANDLE_LIMIT: usize = 1_024;

#[derive(Clone, Debug, Default)]
pub(crate) struct ToolLoopGuardrail {
    seen_idempotent_result_hashes: HashSet<String>,
    seen_idempotent_result_order: VecDeque<String>,
    #[cfg_attr(not(test), allow(dead_code))]
    evicted_idempotent_result_hashes: u64,
    /// Consecutive idle `bash_output` polls per background handle id.
    idle_bg_poll_strikes: HashMap<String, u32>,
    /// A validation observation is reusable only until the next landed
    /// workspace mutation. Including this epoch in its key re-admits the same
    /// validator after real source changes without letting presentation-only
    /// argument churn manufacture progress.
    workspace_mutation_epoch: u64,
    /// The authoritative local ledger revision catches workspace changes that
    /// arrive through candidate publication or background reconciliation and
    /// therefore do not necessarily carry `mutation_applied` on this result.
    workspace_revision: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ToolResultProgress {
    pub(crate) hashable_idempotent: bool,
    pub(crate) repeated_idempotent_result: bool,
    /// True when this result was an idle background poll (running, no new
    /// output). Used to pick a dedicated nudge instead of the wait-poll or
    /// re-read copy.
    pub(crate) idle_background_poll: bool,
    /// True when this result was a `bash_output` poll of a process that is
    /// still running — with or without new output. A live progress bar makes
    /// every poll look like fresh output, so waiting-detection must key on the
    /// process lifecycle, not output novelty.
    pub(crate) running_background_poll: bool,
    /// True when a running-process poll delivered failure diagnostics
    /// (compiler errors, test failures, panics) in its fresh output. That is
    /// new work, not waiting: the wait-streak resets so the model may act on
    /// the evidence — a live turn was forced tool-free one round after its
    /// poll finally surfaced the compile error it needed to fix.
    pub(crate) actionable_background_output: bool,
}

impl ToolLoopGuardrail {
    pub(crate) fn observe_workspace_revision(&mut self, revision: u64) {
        self.workspace_revision = Some(revision);
    }

    #[cfg(test)]
    pub(crate) fn record_tool_result(
        &mut self,
        name: &str,
        arguments: &str,
        output: &str,
    ) -> ToolResultProgress {
        self.record_tool_result_with_effects(name, arguments, output, false)
    }

    pub(crate) fn record_tool_result_with_effects(
        &mut self,
        name: &str,
        arguments: &str,
        output: &str,
        mutation_applied: bool,
    ) -> ToolResultProgress {
        if mutation_applied {
            self.workspace_mutation_epoch = self.workspace_mutation_epoch.saturating_add(1);
        }
        // Wait-polls ("sleep 300 && du -sh …") are exempt from the
        // signature-based repeat guards, so their loop bound lives here: the
        // same poll returning byte-identical output means the awaited state
        // stopped changing.
        let wait_poll = name == "bash" && super::implementation::bash_call_waits(arguments);
        let bounded_probe = (name == "bash" && !mutation_applied)
            .then(|| super::implementation::bash_bounded_execution_probe(arguments))
            .flatten();
        let inspection = (name == "bash" && !mutation_applied)
            .then(|| super::implementation::bash_inspection_signature(arguments))
            .flatten();
        let validation = (name == "bash" && !mutation_applied)
            .then(|| bash_validation_scope(arguments))
            .flatten();
        let running_bg = name == "bash_output" && bash_output_is_running(output);
        let idle_bg = name == "bash_output" && bash_output_is_idle(output);
        if idle_bg {
            return self.record_idle_bg_poll(arguments);
        }
        if name == "bash_output" || name == "bash_kill" {
            // Any non-idle background handle result resets the idle streak so
            // a later quiet stretch starts fresh.
            if let Some(id) = background_handle_id(arguments) {
                self.idle_bg_poll_strikes.remove(&id);
            }
        }
        let actionable_bg = running_bg && output_has_failure_diagnostics(output);
        if !(is_hashable_idempotent_tool(name)
            || wait_poll
            || bounded_probe.is_some()
            || inspection.is_some()
            || validation.is_some())
            || (output.starts_with("Error:") && validation.is_none())
        {
            return ToolResultProgress {
                running_background_poll: running_bg && !output.starts_with("Error:"),
                actionable_background_output: actionable_bg && !output.starts_with("Error:"),
                ..ToolResultProgress::default()
            };
        }
        // Inspections dedup on output alone: the same content reached through
        // different arguments (another path to the same file, a wider grep) is
        // still no new evidence. A wait-poll's key must ALSO cover its
        // arguments: two different polls that happen to print the same bytes —
        // health checks of two different servers both saying "ready: True" —
        // are distinct events, not a static state.
        let key = if let Some(validation) = validation {
            format!(
                "bash-validation:{}:{}",
                stable_result_hash(&validation),
                stable_result_hash(output)
            )
        } else if let Some(probe) = bounded_probe {
            format!("bash-probe:{probe}:{}", stable_result_hash(output))
        } else if wait_poll {
            format!(
                "{name}:{}:{}",
                stable_result_hash(arguments),
                stable_result_hash(output)
            )
        } else if let Some(inspection) = inspection {
            format!(
                "bash-inspection:{}:{}",
                stable_result_hash(&inspection),
                stable_result_hash(output)
            )
        } else {
            format!("{name}:{}", stable_result_hash(output))
        };
        // Identical context may be needed again after an edit elsewhere in
        // the project. Both inspection and validation repeats are local to the
        // current revision, including effects without a ledger revision yet.
        let key = format!(
            "{:?}:{}:{key}",
            self.workspace_revision, self.workspace_mutation_epoch
        );
        let repeated = self.seen_idempotent_result_hashes.contains(&key);
        if !repeated {
            self.seen_idempotent_result_hashes.insert(key.clone());
            self.seen_idempotent_result_order.push_back(key);
            if self.seen_idempotent_result_order.len() > IDEMPOTENT_RESULT_HASH_LIMIT
                && let Some(evicted) = self.seen_idempotent_result_order.pop_front()
            {
                self.seen_idempotent_result_hashes.remove(&evicted);
                self.evicted_idempotent_result_hashes =
                    self.evicted_idempotent_result_hashes.saturating_add(1);
            }
        }
        ToolResultProgress {
            hashable_idempotent: true,
            repeated_idempotent_result: repeated,
            idle_background_poll: false,
            running_background_poll: running_bg,
            actionable_background_output: actionable_bg,
        }
    }

    fn record_idle_bg_poll(&mut self, arguments: &str) -> ToolResultProgress {
        let Some(id) = background_handle_id(arguments) else {
            return ToolResultProgress {
                hashable_idempotent: true,
                repeated_idempotent_result: false,
                idle_background_poll: true,
                running_background_poll: true,
                actionable_background_output: false,
            };
        };
        if !self.idle_bg_poll_strikes.contains_key(&id)
            && self.idle_bg_poll_strikes.len() >= IDLE_BG_HANDLE_LIMIT
            && let Some(evicted) = self.idle_bg_poll_strikes.keys().next().cloned()
        {
            self.idle_bg_poll_strikes.remove(&evicted);
        }
        let strikes = self.idle_bg_poll_strikes.entry(id).or_insert(0);
        *strikes = strikes.saturating_add(1);
        ToolResultProgress {
            hashable_idempotent: true,
            // First `IDLE_BG_POLL_FREE_STRIKES` idle polls are allowed; further
            // ones are the tight-loop case the UI used to render as hung.
            repeated_idempotent_result: *strikes > IDLE_BG_POLL_FREE_STRIKES,
            idle_background_poll: true,
            running_background_poll: true,
            // An idle poll has no fresh output, so nothing actionable in it.
            actionable_background_output: false,
        }
    }
}

/// Failure-diagnostic markers in a poll's output body (everything after the
/// status line). Deliberately failure-shaped only: progress bars and chatty
/// warning-heavy builds must not match, or the wait-streak would never end.
/// A process that emits fresh errors on every poll re-earns the round each
/// time — that is the model reading real evidence, bounded by the turn's
/// other budgets.
fn output_has_failure_diagnostics(output: &str) -> bool {
    let body = output.split_once('\n').map_or("", |(_, rest)| rest);
    [
        "error[",
        "error:",
        "panicked at",
        "FAILED",
        "fatal:",
        "Traceback (most recent call last)",
    ]
    .iter()
    .any(|marker| body.contains(marker))
}

fn bash_output_is_idle(output: &str) -> bool {
    output.lines().next().is_some_and(|status| {
        status.contains("still running — no new output")
            || status.contains("running — no new output")
    })
}

/// The poll's status line says the process is still running, whether or not
/// it delivered fresh output (`[sh_1 · cargo test: still running]` or the idle
/// form with "no new output").
fn bash_output_is_running(output: &str) -> bool {
    output.lines().next().is_some_and(|status| {
        status.starts_with('[')
            && (status.contains("still running")
                || status.contains(": running")
                || status.contains(": running —"))
    })
}

fn background_handle_id(arguments: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(arguments).ok()?;
    let id = value.get("id")?.as_str()?;
    if id.is_empty() {
        None
    } else {
        Some(id.to_string())
    }
}

fn is_hashable_idempotent_tool(name: &str) -> bool {
    matches!(name, "read" | "list" | "grep" | "glob")
}

fn stable_result_hash(output: &str) -> u64 {
    let normalized = serde_json::from_str::<serde_json::Value>(output)
        .map(|value| value.to_string())
        .unwrap_or_else(|_| output.replace("\r\n", "\n"));
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    normalized.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests;
