//! Model-visible context-window budget and no-summary fresh-window reset.
//!
//! Remaining-token notices live on the latest user/volatile block (never
//! `message[0]`) and only fire at window start or when occupancy crosses 25 /
//! 50 / 75 percent. `/window` and the inject-only `new_context` tool drop
//! conversation history while keeping session identity and the current task.

use anyhow::Result;

use crate::Ui;
use crate::compaction;
use crate::transcript::{CONTEXT_BLOCK_END, CONTEXT_BLOCK_START};
use crate::{AUTO_KEEP_RECENT, FALLBACK_CONTEXT_WINDOW};

/// Occupancy at or above this percent may advertise and honor `new_context`.
pub(crate) const NEW_CONTEXT_MIN_OCCUPANCY_PERCENT: u8 = 50;

/// Appended to a goal/plan drive prompt after an automatic fresh window so the
/// model does not assume unread files are still in context.
pub(crate) const FRESH_WINDOW_REORIENT: &str = "This is a new context window — earlier conversation was dropped. Re-orient from the workspace and the goal/plan; do not assume prior reads are still in context.";

const THRESHOLDS: [u8; 3] = [25, 50, 75];

#[derive(Clone, Debug, Default)]
pub(crate) struct TokenBudgetState {
    /// Increments on each fresh window. Shown in the startup fragment.
    pub(crate) window_id: u64,
    /// Highest 25/50/75 band already announced this window (`0` = none).
    announced_threshold: u8,
    startup_announced: bool,
    /// Once `new_context` is advertised, keep it for the rest of the session
    /// so the tool catalog does not churn the prefix cache.
    advertised: bool,
    used_this_turn: bool,
    pending_fresh_window: bool,
    /// Fragment computed at turn start; attached to the volatile block.
    fragment: Option<String>,
}

impl TokenBudgetState {
    pub(crate) fn begin_turn(&mut self, context_used: u64, window: Option<u32>) {
        self.used_this_turn = false;
        self.fragment = self.take_notice(context_used, window);
    }

    pub(crate) fn fragment(&self) -> Option<&str> {
        self.fragment.as_deref()
    }

    pub(crate) fn should_advertise(
        &self,
        occupancy_percent: Option<u8>,
        is_subagent: bool,
        has_window: bool,
    ) -> bool {
        if is_subagent || !has_window {
            return false;
        }
        if self.advertised {
            return true;
        }
        occupancy_percent.is_some_and(|pct| pct >= NEW_CONTEXT_MIN_OCCUPANCY_PERCENT)
    }

    pub(crate) fn note_advertised(&mut self) {
        self.advertised = true;
    }

    pub(crate) fn request_fresh_window(
        &mut self,
        occupancy_percent: Option<u8>,
    ) -> Result<(), String> {
        if self.used_this_turn {
            return Err(
                "new_context already used this turn — finish the current task in this window"
                    .into(),
            );
        }
        match occupancy_percent {
            Some(pct) if pct >= NEW_CONTEXT_MIN_OCCUPANCY_PERCENT => {}
            Some(pct) => {
                return Err(format!(
                    "new_context is for a full or poisoned window (≥{NEW_CONTEXT_MIN_OCCUPANCY_PERCENT}% occupancy); currently ~{pct}%. Keep working, or use smaller reads."
                ));
            }
            None => {
                return Err(
                    "new_context needs a known context window; occupancy is unavailable".into(),
                );
            }
        }
        self.used_this_turn = true;
        self.pending_fresh_window = true;
        Ok(())
    }

    pub(crate) fn take_pending_fresh_window(&mut self) -> bool {
        let pending = self.pending_fresh_window;
        self.pending_fresh_window = false;
        pending
    }

    fn advance_window(&mut self) {
        self.window_id = self.window_id.saturating_add(1);
        self.announced_threshold = 0;
        self.startup_announced = false;
        self.pending_fresh_window = false;
        self.fragment = None;
    }

    fn take_notice(&mut self, context_used: u64, window: Option<u32>) -> Option<String> {
        let window = window.filter(|w| *w > 0)?;
        let pct = occupancy_percent(context_used, window);
        let remaining = u64::from(window).saturating_sub(context_used);
        if !self.startup_announced {
            self.startup_announced = true;
            self.announced_threshold = band(pct);
            return Some(startup_notice(self.window_id, remaining, window, pct));
        }
        let next = band(pct);
        if next > self.announced_threshold {
            self.announced_threshold = next;
            return Some(threshold_notice(next, remaining, window, pct));
        }
        None
    }
}

pub(crate) fn occupancy_percent(context_used: u64, window: u32) -> u8 {
    if window == 0 {
        return 0;
    }
    (context_used.saturating_mul(100) / u64::from(window)).min(100) as u8
}

fn band(percent: u8) -> u8 {
    THRESHOLDS
        .iter()
        .copied()
        .rev()
        .find(|&t| percent >= t)
        .unwrap_or(0)
}

fn startup_notice(window_id: u64, remaining: u64, window: u32, pct: u8) -> String {
    let mut text = format!(
        "[token_budget]\nContext window {window_id}. About {remaining} tokens remain of {window} (~{pct}% used)."
    );
    if pct >= NEW_CONTEXT_MIN_OCCUPANCY_PERCENT {
        text.push_str(
            "\nThe window is getting full — prefer smaller reads. Call `new_context` only if this conversation is no longer useful (failed approach, topic change, poisoned context); that drops history without summarizing. `/window` does the same. `/compact` keeps a summary.",
        );
    }
    text
}

fn threshold_notice(threshold: u8, remaining: u64, window: u32, pct: u8) -> String {
    let mut text = format!(
        "[token_budget]\nOccupancy crossed {threshold}% (~{pct}% of {window}; about {remaining} tokens remain). Prefer smaller reads; do not dump files."
    );
    if threshold >= NEW_CONTEXT_MIN_OCCUPANCY_PERCENT {
        text.push_str(
            " Call `new_context` only if this window is no longer useful — it drops conversation without a summary and keeps the current task, goal, and decisions.",
        );
    }
    text
}

pub(crate) fn inner_user_text(text: &str) -> String {
    if text.starts_with(CONTEXT_BLOCK_START)
        && let Some(end) = text.find(CONTEXT_BLOCK_END)
    {
        return text[end + CONTEXT_BLOCK_END.len()..]
            .trim_start_matches('\n')
            .to_string();
    }
    text.to_string()
}

fn wrap_current_task(volatile: Option<&str>, task: &str) -> String {
    match volatile.filter(|block| !block.trim().is_empty()) {
        Some(block) => format!("{CONTEXT_BLOCK_START}\n{block}\n{CONTEXT_BLOCK_END}\n\n{task}"),
        None => task.to_string(),
    }
}

fn new_context_outcome(
    content: impl Into<String>,
    status: hi_tools::ToolStatus,
) -> hi_tools::ToolOutcome {
    let (content, truncation) = hi_tools::bound_tool_content(content.into());
    hi_tools::ToolOutcome {
        content,
        display: None,
        plan: None,
        status,
        process: None,
        background: None,
        effects: hi_tools::ToolEffects::default(),
        truncation,
        images: Vec::new(),
    }
}

impl crate::Agent {
    pub(crate) fn handle_new_context(&mut self) -> hi_tools::ToolOutcome {
        let occupancy = self
            .config
            .routing
            .context_window
            .filter(|w| *w > 0)
            .map(|w| occupancy_percent(self.report.context_used, w));
        match self.token_budget.request_fresh_window(occupancy) {
            Ok(()) => new_context_outcome(
                "A new context window will start without summarizing conversation history. Goal, decisions, and the current task are kept.",
                hi_tools::ToolStatus::Succeeded,
            ),
            Err(message) => {
                new_context_outcome(format!("Error: {message}"), hi_tools::ToolStatus::Failed)
            }
        }
    }

    /// Drop conversation history; keep system identity, goal/decisions/memory,
    /// and the current user task. No summary call.
    pub(crate) fn apply_fresh_window(
        &mut self,
        ui: &mut dyn Ui,
        current_task: Option<&str>,
    ) -> Result<()> {
        let task = current_task
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_string)
            .or_else(|| self.current_window_task());
        self.wipe_conversation_keep_identity()?;
        if let Some(task) = task {
            let wrapped = wrap_current_task(self.volatile_context_block().as_deref(), &task);
            self.messages.push_user(wrapped);
        }
        ui.status("fresh context window — conversation dropped, goal/decisions kept");
        Ok(())
    }

    /// Replace the transcript with the stable system message. Goal, decisions,
    /// memory, and stall counters live on `Agent` and survive. A new window
    /// epoch resets drive stalls so a poisoned thread can keep running.
    fn wipe_conversation_keep_identity(&mut self) -> Result<()> {
        self.token_budget.advance_window();
        self.replace_history_with_compaction(vec![self.system_message()])?;
        self.runtime.invalidate_context_after_compaction();
        self.report.context_used = 0;
        self.token_budget
            .begin_turn(0, self.config.routing.context_window);
        self.reset_goal_drive_stall();
        self.reset_plan_drive_stall();
        Ok(())
    }

    /// If occupancy is past the auto-compact threshold, reclaim room before
    /// this turn's user message. Goal/plan drive uses a no-summary fresh
    /// window (the job already lives outside the transcript). Interactive
    /// user turns still elide, then summarize.
    ///
    /// Returns `true` when a drive turn started a fresh window (caller should
    /// append [`FRESH_WINDOW_REORIENT`]).
    pub(crate) async fn maybe_reclaim_context(
        &mut self,
        ui: &mut dyn Ui,
        continual_drive: bool,
    ) -> Result<bool> {
        if !self.config.memory.auto_compact || self.report.context_used == 0 {
            return Ok(false);
        }
        let real_window = self
            .config
            .routing
            .context_window
            .filter(|window| *window > 0);
        let occupancy_window = real_window.unwrap_or(FALLBACK_CONTEXT_WINDOW);
        if occupancy_window == 0
            || self.report.context_used * 100
                < u64::from(occupancy_window) * self.config.memory.auto_compact_percent
        {
            return Ok(false);
        }
        let pct = self.report.context_used * 100 / u64::from(occupancy_window);
        if continual_drive {
            self.wipe_conversation_keep_identity()?;
            ui.status(&format!(
                "context ~{pct}% full — fresh window so the goal can keep running (conversation dropped, goal/decisions kept)"
            ));
            return Ok(true);
        }
        ui.status(&format!("context ~{pct}% full — compacting to free room"));
        if let Some(split) = compaction::recent_split(self.messages.as_slice(), AUTO_KEEP_RECENT)
            && compaction::elide_tool_outputs(self.messages.mutate_slice(), split) > 0
        {
            self.runtime.invalidate_context_after_compaction();
        }
        if let Some(window) = real_window {
            let target = u64::from(window) * self.config.memory.compact_target_percent / 100;
            if compaction::estimate_tokens(self.messages.as_slice()) > target {
                let _ = self.compact(ui).await;
            }
        }
        self.report.context_used = 0;
        Ok(false)
    }

    pub(crate) fn compact_fresh_window(&mut self, ui: &mut dyn Ui) -> Result<()> {
        self.apply_fresh_window(ui, None)
    }

    fn current_window_task(&self) -> Option<String> {
        if let Some(prompt) = self
            .task
            .last_task_prompt
            .as_deref()
            .map(str::trim)
            .filter(|text| !text.is_empty())
        {
            return Some(prompt.to_string());
        }
        let inner = self.messages.as_slice().iter().rev().find_map(|message| {
            (message.role == hi_ai::Role::User).then(|| inner_user_text(&message.text()))
        })?;
        let trimmed = inner.trim();
        if trimmed.is_empty()
            || trimmed.contains("[CONTEXT COMPACTION")
            || trimmed.contains("Earlier conversation context was omitted")
        {
            return None;
        }
        Some(trimmed.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn occupancy_and_bands() {
        assert_eq!(occupancy_percent(0, 100), 0);
        assert_eq!(occupancy_percent(25, 100), 25);
        assert_eq!(occupancy_percent(50, 100), 50);
        assert_eq!(band(10), 0);
        assert_eq!(band(25), 25);
        assert_eq!(band(49), 25);
        assert_eq!(band(50), 50);
        assert_eq!(band(90), 75);
    }

    #[test]
    fn startup_then_threshold_notices_are_once_each() {
        let mut state = TokenBudgetState::default();
        state.begin_turn(10_000, Some(100_000));
        let first = state.fragment().expect("startup");
        assert!(first.contains("Context window 0"));
        assert!(first.contains("100000"));
        assert!(!first.contains("crossed"));

        state.begin_turn(12_000, Some(100_000));
        assert!(state.fragment().is_none(), "no re-announce below 25%");

        state.begin_turn(26_000, Some(100_000));
        let crossed = state.fragment().expect("25%");
        assert!(crossed.contains("crossed 25%"));
        assert!(!crossed.contains("new_context"));

        state.begin_turn(30_000, Some(100_000));
        assert!(state.fragment().is_none());

        state.begin_turn(51_000, Some(100_000));
        let half = state.fragment().expect("50%");
        assert!(half.contains("crossed 50%"));
        assert!(half.contains("new_context"));
    }

    #[test]
    fn advertise_is_sticky_after_occupancy_gate() {
        let mut state = TokenBudgetState::default();
        assert!(!state.should_advertise(Some(10), false, true));
        assert!(state.should_advertise(Some(50), false, true));
        state.note_advertised();
        assert!(
            state.should_advertise(Some(10), false, true),
            "sticky after first ad"
        );
        assert!(!state.should_advertise(Some(90), true, true));
        assert!(!state.should_advertise(Some(90), false, false));
    }

    #[test]
    fn new_context_rejects_low_occupancy_and_second_call() {
        let mut state = TokenBudgetState::default();
        assert!(state.request_fresh_window(Some(20)).is_err());
        assert!(state.request_fresh_window(None).is_err());
        assert!(state.request_fresh_window(Some(50)).is_ok());
        assert!(state.take_pending_fresh_window());
        assert!(!state.take_pending_fresh_window());
        assert!(state.request_fresh_window(Some(80)).is_err());
    }

    #[test]
    fn inner_user_text_strips_volatile_block() {
        let wrapped = format!("{CONTEXT_BLOCK_START}\ngoal\n{CONTEXT_BLOCK_END}\n\nfix the parser");
        assert_eq!(inner_user_text(&wrapped), "fix the parser");
        assert_eq!(inner_user_text("plain"), "plain");
    }
}
