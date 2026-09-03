use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};

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
            || inspection.is_some())
            || output.starts_with("Error:")
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
        let key = if let Some(probe) = bounded_probe {
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
mod tests {
    use super::*;

    #[test]
    fn distinct_idempotent_results_use_bounded_fifo_repeat_memory() {
        let mut guard = ToolLoopGuardrail::default();
        for index in 0..5_000 {
            let result = guard.record_tool_result(
                "read",
                r#"{"path":"src/lib.rs"}"#,
                &format!("unique output {index}"),
            );
            assert!(!result.repeated_idempotent_result);
        }

        assert_eq!(
            guard.seen_idempotent_result_hashes.len(),
            IDEMPOTENT_RESULT_HASH_LIMIT
        );
        assert_eq!(
            guard.seen_idempotent_result_order.len(),
            IDEMPOTENT_RESULT_HASH_LIMIT
        );
        assert_eq!(guard.evicted_idempotent_result_hashes, 904);
        assert!(
            guard
                .record_tool_result("read", r#"{"path":"src/lib.rs"}"#, "unique output 4999",)
                .repeated_idempotent_result
        );
        assert!(
            !guard
                .record_tool_result("read", r#"{"path":"src/lib.rs"}"#, "unique output 0",)
                .repeated_idempotent_result,
            "evicting an ancient repeat only weakens the loop heuristic; it must not stop work"
        );
    }

    #[test]
    fn repeated_read_result_is_no_progress_even_with_different_args() {
        let mut guard = ToolLoopGuardrail::default();

        let first = guard.record_tool_result("read", r#"{"path":"a.rs"}"#, "same output");
        let second = guard.record_tool_result("read", r#"{"path":"b.rs"}"#, "same output");

        assert!(first.hashable_idempotent);
        assert!(!first.repeated_idempotent_result);
        assert!(second.hashable_idempotent);
        assert!(second.repeated_idempotent_result);
    }

    #[test]
    fn wait_poll_bash_is_hash_guarded_but_plain_bash_is_not() {
        let mut guard = ToolLoopGuardrail::default();
        let wait_args = r#"{"command":"sleep 300 && du -sh models/"}"#;

        let first = guard.record_tool_result("bash", wait_args, "90G\t18 shards");
        assert!(first.hashable_idempotent);
        assert!(!first.repeated_idempotent_result);

        let progressed = guard.record_tool_result("bash", wait_args, "124G\t27 shards");
        assert!(progressed.hashable_idempotent);
        assert!(
            !progressed.repeated_idempotent_result,
            "changing output is progress"
        );

        let static_poll = guard.record_tool_result("bash", wait_args, "124G\t27 shards");
        assert!(
            static_poll.repeated_idempotent_result,
            "identical output means the awaited state stopped changing"
        );

        let plain = guard.record_tool_result("bash", r#"{"command":"cargo test"}"#, "ok");
        assert!(!plain.hashable_idempotent, "plain bash is not hash guarded");
    }

    #[test]
    fn varied_bounded_launches_with_the_same_result_are_deduplicated() {
        let mut guard = ToolLoopGuardrail::default();
        let head = r#"{"command":"timeout 10 ./target/debug/app 2>&1 | head -30; echo exit=$?"}"#;
        let tail = r#"{"command":"timeout 20 ./target/debug/app 2>&1 | tail -30; echo exit=$?"}"#;
        let output = "Seeded 20 agents (day 1).\nexit=0";

        let first = guard.record_tool_result_with_effects("bash", head, output, false);
        let repeated = guard.record_tool_result_with_effects("bash", tail, output, false);
        assert!(first.hashable_idempotent);
        assert!(!first.repeated_idempotent_result);
        assert!(repeated.repeated_idempotent_result);

        let mut mutation_guard = ToolLoopGuardrail::default();
        let mutation = mutation_guard.record_tool_result_with_effects("bash", head, output, true);
        assert!(!mutation.hashable_idempotent);
    }

    #[test]
    fn alternating_shell_inspections_cannot_evade_result_deduplication() {
        let mut guard = ToolLoopGuardrail::default();
        let page_a =
            r#"{"command":"for f in blog_posts/txt/*.txt; do sed -n '1,100p' \"$f\"; done"}"#;
        let page_b =
            r#"{"command":"for f in blog_posts/txt/*.txt; do sed -n '100,150p' \"$f\"; done"}"#;

        let first_a = guard.record_tool_result_with_effects("bash", page_a, "page A", false);
        let first_b = guard.record_tool_result_with_effects("bash", page_b, "page B", false);
        let repeated_a = guard.record_tool_result_with_effects("bash", page_a, "page A", false);

        assert!(first_a.hashable_idempotent);
        assert!(!first_a.repeated_idempotent_result);
        assert!(first_b.hashable_idempotent);
        assert!(!first_b.repeated_idempotent_result);
        assert!(repeated_a.repeated_idempotent_result);

        let same_output_different_page =
            guard.record_tool_result_with_effects("bash", page_b, "page A", false);
        assert!(
            !same_output_different_page.repeated_idempotent_result,
            "a distinct inspection page is new evidence even when its text matches"
        );

        let mutation = guard.record_tool_result_with_effects("bash", page_a, "page A", true);
        assert!(!mutation.hashable_idempotent);
    }

    #[test]
    fn different_wait_polls_with_identical_output_are_distinct_events() {
        // Health checks of two different servers both printing "ready: True"
        // must not read as a static state — the key covers the arguments.
        let mut guard = ToolLoopGuardrail::default();
        let first = guard.record_tool_result(
            "bash",
            r#"{"command":"sleep 30 && curl -fsS http://127.0.0.1:18101/health"}"#,
            "ready: True",
        );
        let second = guard.record_tool_result(
            "bash",
            r#"{"command":"sleep 30 && curl -fsS http://127.0.0.1:18102/health"}"#,
            "ready: True",
        );
        assert!(!first.repeated_idempotent_result);
        assert!(
            !second.repeated_idempotent_result,
            "a different poll is a different event even with identical output"
        );

        let same_again = guard.record_tool_result(
            "bash",
            r#"{"command":"sleep 30 && curl -fsS http://127.0.0.1:18102/health"}"#,
            "ready: True",
        );
        assert!(
            same_again.repeated_idempotent_result,
            "the same poll repeating its own output is static"
        );
    }

    #[test]
    fn idle_bash_output_allows_two_polls_then_flags_tight_loop() {
        let mut guard = ToolLoopGuardrail::default();
        let args = r#"{"id":"sh_1"}"#;
        let idle = "[sh_1: still running — no new output]";

        let first = guard.record_tool_result("bash_output", args, idle);
        let second = guard.record_tool_result("bash_output", args, idle);
        let third = guard.record_tool_result("bash_output", args, idle);

        assert!(first.idle_background_poll && !first.repeated_idempotent_result);
        assert!(second.idle_background_poll && !second.repeated_idempotent_result);
        assert!(
            third.idle_background_poll && third.repeated_idempotent_result,
            "third consecutive idle poll is a tight loop"
        );

        let other = guard.record_tool_result(
            "bash_output",
            r#"{"id":"sh_2"}"#,
            "[sh_2: still running — no new output]",
        );
        assert!(
            !other.repeated_idempotent_result,
            "a different handle starts a fresh idle streak"
        );
    }

    #[test]
    fn fresh_bash_output_resets_idle_streak() {
        let mut guard = ToolLoopGuardrail::default();
        let args = r#"{"id":"sh_1"}"#;
        let idle = "[sh_1: still running — no new output]";

        assert!(
            !guard
                .record_tool_result("bash_output", args, idle)
                .repeated_idempotent_result
        );
        assert!(
            !guard
                .record_tool_result("bash_output", args, idle)
                .repeated_idempotent_result
        );

        let progressed =
            guard.record_tool_result("bash_output", args, "[sh_1: still running]\n== hi-ai ==\n");
        assert!(!progressed.idle_background_poll);
        assert!(!progressed.repeated_idempotent_result);

        // After progress, two more idle polls are allowed again.
        assert!(
            !guard
                .record_tool_result("bash_output", args, idle)
                .repeated_idempotent_result
        );
        assert!(
            !guard
                .record_tool_result("bash_output", args, idle)
                .repeated_idempotent_result
        );
        assert!(
            guard
                .record_tool_result("bash_output", args, idle)
                .repeated_idempotent_result
        );
    }

    #[test]
    fn running_polls_are_flagged_regardless_of_output_novelty() {
        let mut guard = ToolLoopGuardrail::default();
        let args = r#"{"id":"sh_1"}"#;

        // A progress bar delivers fresh bytes on every poll: not idle, but
        // still a poll of a running process — the waiting classifier keys on
        // this, not on output novelty.
        let progressing = guard.record_tool_result(
            "bash_output",
            args,
            "[sh_1: still running]\n42.1 GiB / 767.7 GiB",
        );
        assert!(progressing.running_background_poll);
        assert!(!progressing.idle_background_poll);

        let idle =
            guard.record_tool_result("bash_output", args, "[sh_1: still running — no new output]");
        assert!(idle.running_background_poll && idle.idle_background_poll);

        let exited =
            guard.record_tool_result("bash_output", args, "[sh_1: exited with code 0]\ndone");
        assert!(!exited.running_background_poll);

        let errored = guard.record_tool_result("bash_output", args, "Error: no background process");
        assert!(!errored.running_background_poll);
    }

    #[test]
    fn mutating_tools_are_not_hash_guarded() {
        let mut guard = ToolLoopGuardrail::default();

        let first = guard.record_tool_result("write", r#"{"path":"a.rs"}"#, "Wrote a.rs");
        let second = guard.record_tool_result("write", r#"{"path":"b.rs"}"#, "Wrote a.rs");

        assert!(!first.hashable_idempotent);
        assert!(!second.repeated_idempotent_result);
    }

    #[test]
    fn error_bearing_running_poll_is_actionable_but_progress_noise_is_not() {
        // The incident this pins: a 600s poll finally surfaced a compile
        // error, and the wait-streak escalation forced a tool-free final
        // answer anyway. Diagnostics in fresh output are work, not waiting.
        let mut guard = ToolLoopGuardrail::default();
        let args = r#"{"id":"cargo-check_1"}"#;
        let noise = guard.record_tool_result(
            "bash_output",
            args,
            "[cargo-check_1 \u{b7} cargo check: still running]\n42.1 GiB / 767.7 GiB",
        );
        assert!(noise.running_background_poll);
        assert!(
            !noise.actionable_background_output,
            "progress noise is not actionable"
        );
        let diag = guard.record_tool_result(
            "bash_output",
            args,
            "[cargo-check_1 \u{b7} cargo check: still running]\nerror[E0107]: enum takes 2 \
             generic arguments but 1 generic argument was supplied",
        );
        assert!(diag.running_background_poll);
        assert!(
            diag.actionable_background_output,
            "compiler errors are actionable"
        );
        // A terminal poll is not a running poll, so the flag stays off.
        let exited = guard.record_tool_result(
            "bash_output",
            args,
            "[cargo-check_1: exited code 101]\nerror: could not compile",
        );
        assert!(!exited.running_background_poll);
        assert!(!exited.actionable_background_output);
    }
}
