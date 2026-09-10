//! Grok-build identical-tool stationarity: nudge once, then hard-stop.
//! A hard stop after retained mutations is leftover, not `NoProgress`.

pub(crate) const NUDGE_AFTER_IDENTICAL_PROBLEMATIC_TOOL_CALLS: u32 = 4;
pub(crate) const NUDGE_AFTER_IDENTICAL_TOOL_CALLS: u32 = 8;
pub(crate) const MAX_CONSECUTIVE_IDENTICAL_PROBLEMATIC_TOOL_CALLS: u32 = 8;
pub(crate) const MAX_CONSECUTIVE_IDENTICAL_TOOL_CALLS: u32 = 12;

pub(crate) const STATIONARITY_NUDGE: &str = "You have repeated the same tool call without new evidence. Do not issue that identical call again. Take a different action: edit a new path, run a different check, or finish with the evidence already in the conversation.";

#[derive(Clone, Debug, Default)]
pub(crate) struct IdenticalToolCallRun {
    last_signature: Option<String>,
    pub(crate) tool_name: String,
    problematically_repeating_step: bool,
    pub(crate) run_len: u32,
    pub(crate) nudged: bool,
}

impl IdenticalToolCallRun {
    pub(crate) fn observe(
        &mut self,
        signature: &str,
        tool_name: &str,
        problematically_repeating_step: bool,
    ) -> u32 {
        if self.last_signature.as_deref() == Some(signature) {
            self.run_len = self.run_len.saturating_add(1);
        } else {
            self.run_len = 1;
            self.last_signature = Some(signature.to_owned());
            self.nudged = false;
        }
        self.tool_name = tool_name.to_owned();
        self.problematically_repeating_step = problematically_repeating_step;
        self.run_len
    }

    pub(crate) fn observe_calls(&mut self, calls: &[(String, String, String)]) -> u32 {
        let signature = step_signature(calls);
        let tool_name = calls
            .first()
            .map(|(_, name, _)| name.as_str())
            .unwrap_or("tool");
        let problematic = step_is_problematically_repeating(calls);
        self.observe(&signature, tool_name, problematic)
    }

    fn is_problematically_repeating(&self) -> bool {
        self.problematically_repeating_step
    }

    fn nudge_threshold(&self) -> u32 {
        if self.is_problematically_repeating() {
            NUDGE_AFTER_IDENTICAL_PROBLEMATIC_TOOL_CALLS
        } else {
            NUDGE_AFTER_IDENTICAL_TOOL_CALLS
        }
    }

    pub(crate) fn take_nudge(&mut self) -> bool {
        let fire = self.run_len >= self.nudge_threshold() && !self.nudged;
        self.nudged |= fire;
        fire
    }

    pub(crate) fn hard_stop(&self) -> bool {
        self.run_len >= self.hard_stop_threshold()
    }

    pub(crate) fn hard_stop_threshold(&self) -> u32 {
        if self.is_problematically_repeating() {
            MAX_CONSECUTIVE_IDENTICAL_PROBLEMATIC_TOOL_CALLS
        } else {
            MAX_CONSECUTIVE_IDENTICAL_TOOL_CALLS
        }
    }
}

pub(crate) fn step_signature(calls: &[(String, String, String)]) -> String {
    let mut parts: Vec<String> = calls
        .iter()
        .map(|(_, name, args)| format!("{name}\u{1f}{args}"))
        .collect();
    parts.sort();
    parts.join("\u{1e}")
}

pub(crate) fn step_is_problematically_repeating(calls: &[(String, String, String)]) -> bool {
    !calls.is_empty()
        && calls.iter().all(|(_, name, _)| {
            matches!(
                name.as_str(),
                "read" | "grep" | "glob" | "update_plan" | "bash_output"
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, args: &str) -> (String, String, String) {
        ("id".into(), name.into(), args.into())
    }

    #[test]
    fn identical_reads_nudge_then_hard_stop() {
        let mut run = IdenticalToolCallRun::default();
        let calls = [call("read", r#"{"path":"src/lib.rs"}"#)];
        for i in 1..NUDGE_AFTER_IDENTICAL_PROBLEMATIC_TOOL_CALLS {
            assert_eq!(run.observe_calls(&calls), i);
            assert!(!run.take_nudge(), "no nudge at {i}");
            assert!(!run.hard_stop());
        }
        assert_eq!(
            run.observe_calls(&calls),
            NUDGE_AFTER_IDENTICAL_PROBLEMATIC_TOOL_CALLS
        );
        assert!(run.take_nudge());
        assert!(!run.take_nudge());
        while run.run_len < MAX_CONSECUTIVE_IDENTICAL_PROBLEMATIC_TOOL_CALLS {
            run.observe_calls(&calls);
        }
        assert!(run.hard_stop());
        assert_eq!(
            run.hard_stop_threshold(),
            MAX_CONSECUTIVE_IDENTICAL_PROBLEMATIC_TOOL_CALLS
        );
    }

    #[test]
    fn different_signature_resets_the_run() {
        let mut run = IdenticalToolCallRun::default();
        run.observe_calls(&[call("read", r#"{"path":"a"}"#)]);
        run.observe_calls(&[call("read", r#"{"path":"a"}"#)]);
        assert_eq!(run.run_len, 2);
        run.observe_calls(&[call("read", r#"{"path":"b"}"#)]);
        assert_eq!(run.run_len, 1);
        assert!(!run.hard_stop());
    }

    #[test]
    fn execute_tools_use_the_looser_tier() {
        let mut run = IdenticalToolCallRun::default();
        let calls = [call("bash", r#"{"command":"true"}"#)];
        assert!(!step_is_problematically_repeating(&calls));
        run.observe_calls(&calls);
        assert_eq!(
            run.hard_stop_threshold(),
            MAX_CONSECUTIVE_IDENTICAL_TOOL_CALLS
        );
        assert_eq!(run.nudge_threshold(), NUDGE_AFTER_IDENTICAL_TOOL_CALLS);
    }
}
