//! `App` methods: lifecycle.

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

use crate::input::InputLine;
use crate::util::notify_done;
use crate::{NOTIFY_THRESHOLD, TurnState};

impl crate::App {
    pub(crate) fn new(provider: &str, model: &str) -> Self {
        Self {
            provider: provider.to_string(),
            model: model.to_string(),
            reasoning_effort: None,
            workspace_root: std::path::PathBuf::new(),
            input_history_path: std::path::PathBuf::new(),
            interrupt: None,
            active_profile: None,
            profiles: Vec::new(),
            resolver: Box::new(|_| anyhow::bail!("not available in the Pipe Network session")),
            saver: Box::new(|_| anyhow::bail!("not available in the Pipe Network session")),
            loader: Box::new(|_| anyhow::bail!("not available in the Pipe Network session")),
            remover: Box::new(|_| anyhow::bail!("not available in the Pipe Network session")),
            reasoning_effort_saver: None,
            session_remember: None,
            local_runtime: None,
            mcp_url: None,
            api_key: String::new(),
            event_sink: None,
            approval_store: None,
            transcript: Vec::new(),
            session_projection: Default::default(),
            workflow_revisions: std::collections::HashMap::new(),
            workflow_completion_handoffs: std::collections::HashMap::new(),
            pending: None,
            reasoning_buffer: String::new(),
            reasoning_started: None,
            show_reasoning: false,
            show_tool_output: false,
            density: crate::Density::Comfortable,
            mode: crate::mode::UiMode::Insert,
            last_search: None,
            block_cursor: 0,
            transcript_gen: 0,
            view_cache: crate::view_cache::TranscriptViewCache::default(),
            view_geometry_key: None,
            code_lang: None,
            last_code_block: None,
            table_buf: Vec::new(),
            input: InputLine::default(),
            voice: Default::default(),
            voice_model: Default::default(),
            voice_config: Default::default(),
            following: true,
            scroll: 0,
            view_max_scroll: 0,
            view_total: 0,
            view_inner: ratatui::layout::Rect::default(),
            view_scroll: 0,
            block_row_spans: Vec::new(),
            view_prefix: Vec::new(),
            view_line_texts: Vec::new(),
            select_anchor: None,
            select_cursor: None,
            select_dragged: false,
            copy_toast: None,
            mouse_capture: true,
            minimal_screen: false,
            vim_mode: true,
            multiline_mode: true,
            timeline_enabled: true,
            timestamps_enabled: true,
            page_flip_on_send: false,
            total_when_unpinned: 0,
            working: false,
            spinner: 0,
            started: None,
            last_turn_latency: None,
            finished_at: None,
            current_tool: None,
            current_tool_started: None,
            queue: VecDeque::new(),
            queue_paused: false,
            mid_turn_offered: VecDeque::new(),
            queue_selected: None,
            last_prompt: None,
            last_turn_start: 0,
            picker: None,
            session_picker: false,
            session_picker_searching: false,
            session_catalog_flags: HashMap::new(),
            session_delete_pending: None,
            provider_form: None,
            provider_picker: None,
            pending_login: None,
            pending_auth: None,
            x402_broker: None,
            fetching: None,
            planning: None,
            status: String::new(),
            plan: Vec::new(),
            confirmation: None,
            confirmation_scroll: 0,
            confirmation_selected: 0,
            confirm_focus: crate::confirm_overlay::ConfirmFocus::Options,
            confirmation_waiting: 0,
            pending_resume: None,
            resume_incomplete_requested: false,
            mouse_col: 0,
            mouse_row: 0,
            ctx_chip_rect: ratatui::layout::Rect::default(),
            turn_status_rect: ratatui::layout::Rect::default(),
            git_branch: None,
            plan_pane_expanded: true,
            plan_mode: false,
            permission_mode: hi_harness::PermissionMode::Ask,
            session_face_dirty: false,
            plan_drive_paused: false,
            plan_drive_pause_dirty: false,
            ask_user_draft: String::new(),
            block_viewer: None,
            timeline_hits: Vec::new(),
            timeline_rect: ratatui::layout::Rect::default(),
            changed_files_rect: ratatui::layout::Rect::default(),
            composer_rect: ratatui::layout::Rect::default(),
            frame_width: 0,
            plan_workflow_child: None,
            usage: (0, 0),
            usage_estimated: false,
            session_totals: hi_ai::Usage::default(),
            usage_pricing: None,
            new_model_ids: HashSet::new(),
            context_used: 0,
            context_window: None,
            rate_limits: None,
            served: HashMap::new(),
            model_ids: Vec::new(),
            trimmed: 0,
            current_assistant: String::new(),
            current_assistant_streamed_bytes: 0,
            assistant_message_open: false,
            show_btw: false,
            btw_scroll: 0,
            last_btw_area: ratatui::layout::Rect::default(),
            last_btw_close: ratatui::layout::Rect::default(),
            btw_thread: Vec::new(),
            last_assistant: String::new(),
            last_turn_event: None,
            last_turn_had_file_edits: false,
            turn_steering_seen: HashSet::new(),
            turn_status_seen: HashSet::new(),
            last_changed_files: Vec::new(),
            session_changed_files: Vec::new(),
            suggested_prompt: None,
            suggested_prompt_dismissed: false,
            review: crate::review::ReviewState::default(),
            auto_approve_session: false,
            auto_approve_paths: Vec::new(),
            auto_approve_mcp: Vec::new(),
            show_debug: false,
            show_help: false,
            palette: None,
            tutorial: None,
            usage_overlay: None,
            dashboard: None,
            pipe_base_url: String::new(),
            openai_api_key: None,
            openai_base_url: None,
            last_turn_phase: None,
            turn_tool_calls: 0,
            turn_rounds: 0,
            run_streamed_this_call: false,
            waiting_for: None,
            provider_activity: Default::default(),
            last_turn_state: TurnState::Idle,
            last_error: None,
            event_log: Vec::new(),
            model_issues: HashMap::new(),
            top_notice: None,
            working_status: None,
            startup_notice: None,
            checkpoint_warning: None,
            quit_notice: None,
            turn_stop_requested: false,
            exit_requested: false,
            completion: None,
            path_completion_cache: Vec::new(),
            focused: true,
            focus_known: false,
            sync_config: None,
            sync_active: false,
            sync_session_id: None,
            sync_http: None,
            session_lister: None,
            session_completion_cache: Vec::new(),
            session_renamer: None,
            session_host: None,
            pending_host_enable: None,
            team_picker_role: None,
            team_role_menu: false,
            sync_control: None,
            remote_event_tap: None,
            base_event_tap: None,
            remote_flush_callback: None,
            remote_input_rx: None,
            remote_input_poller: None,
            hosting_remote_input: false,
        }
    }

    /// Record a focus-change report from the terminal (and that it reports them).
    pub(crate) fn set_focus(&mut self, focused: bool) {
        self.focused = focused;
        self.focus_known = true;
    }

    /// Ping the terminal when a turn finishes and you're likely away: when the
    /// terminal reports it's unfocused, or — on terminals that don't report
    /// focus — when the turn ran long enough that you probably stepped away.
    pub(crate) fn maybe_notify_done(&self) {
        let elapsed = self.started.map(|t| t.elapsed()).unwrap_or_default();
        let away = if self.focus_known {
            !self.focused
        } else {
            elapsed >= NOTIFY_THRESHOLD
        };
        if away {
            notify_done();
        }
    }

    pub(crate) fn drain_loops(&mut self) {}

    /// Hide last turn's finished checklist when a new prompt starts running.
    pub(crate) fn dismiss_completed_plan(&mut self) {
        if hi_tools::PlanStep::all_complete(&self.plan) {
            self.plan.clear();
        }
    }

    /// Mark the turn as running (or done), stamping the start time so the
    /// prompt bar can show elapsed seconds.
    pub(crate) fn set_working(&mut self, working: bool) {
        let was_working = self.working;
        if was_working && !working {
            self.last_turn_latency = self.started.map(|started| started.elapsed());
            if let Some(elapsed) = self.last_turn_latency {
                let marker = match &self.last_turn_state {
                    TurnState::Done(_) | TurnState::Warning(_) => {
                        Some(format!("Worked for {}", crate::util::fmt_worked(elapsed)))
                    }
                    TurnState::Cancelled => Some(format!(
                        "Turn cancelled by user in {}.",
                        crate::util::fmt_worked(elapsed)
                    )),
                    TurnState::Failed(_) => Some(format!(
                        "Turn failed in {}.",
                        crate::util::fmt_worked(elapsed)
                    )),
                    _ => None,
                };
                if let Some(marker) = marker {
                    let blank = self
                        .transcript
                        .last()
                        .is_none_or(|entry| entry.text().trim().is_empty());
                    if !blank {
                        self.push(ratatui::text::Line::raw(""));
                    }
                    self.push(ratatui::text::Line::styled(marker, crate::render::dim()));
                }
            }
        }
        self.working = working;
        self.started = working.then(Instant::now);
        self.current_tool = None;
        self.current_tool_started = None;
        self.run_streamed_this_call = false;
        self.working_status = None;
        self.changed_files_rect = ratatui::layout::Rect::default();
        if !working {
            self.freeze_verb_group();
            self.turn_steering_seen.clear();
            self.turn_status_seen.clear();
        }
        if working {
            self.turn_stop_requested = false;
            self.checkpoint_warning = None;
            self.top_notice = None;
            self.last_turn_event = None;
            self.last_turn_had_file_edits = false;
            self.turn_steering_seen.clear();
            self.turn_status_seen.clear();
            self.waiting_for = Some(Duration::ZERO);
            self.provider_activity = Default::default();
            self.last_turn_state = TurnState::Running;
            // Ghost-text suggestions are for the idle composer only.
            self.suggested_prompt = None;
            self.suggested_prompt_dismissed = false;
            // A new turn's output would shift block ordinals and line indices;
            // leave block-nav and drop any stale text selection.
            if self.mode.is_block_nav() {
                self.mode.to_insert();
            }
            self.clear_selection();
        } else if matches!(self.last_turn_state, TurnState::Running) {
            self.last_turn_state = TurnState::Idle;
            self.waiting_for = None;
        }
        // Stamp the completion so the status line can flash briefly as it settles.
        if was_working && !working {
            self.finished_at = Some(Instant::now());
        }
    }

    /// Drop any idle ghost-text suggestion (typing, Esc, toggle off, etc.).
    pub(crate) fn clear_suggested_prompt(&mut self) {
        self.suggested_prompt = None;
        self.suggested_prompt_dismissed = false;
    }

    /// Hide the current suggestion until a new one loads (Esc on empty).
    pub(crate) fn dismiss_suggested_prompt(&mut self) {
        self.suggested_prompt_dismissed = true;
    }

    /// Remaining suffix of the suggestion when `text` is a proper prefix.
    pub(crate) fn ghost_suffix(&self) -> Option<&str> {
        if self.suggested_prompt_dismissed {
            return None;
        }
        let suggestion = self.suggested_prompt.as_deref()?;
        let text = self.input.text();
        let rest = suggestion.strip_prefix(&text)?;
        if rest.is_empty() { None } else { Some(rest) }
    }

    /// Accept the ghost-text suggestion into the input buffer (Tab/Right).
    /// Inserts the remaining suffix when the draft is a matching prefix.
    pub(crate) fn accept_suggested_prompt(&mut self) -> bool {
        let Some(rest) = self.ghost_suffix().map(str::to_owned) else {
            return false;
        };
        self.input.insert_str(&rest);
        self.suggested_prompt = None;
        self.suggested_prompt_dismissed = false;
        true
    }

    pub(crate) fn record_model_issue(&mut self) {
        let _count = {
            let entry = self.model_issues.entry(self.model.clone()).or_insert(0);
            *entry += 1;
            *entry
        };
        // Note: don't touch `last_error` here — it holds the actual failure
        // reason set by the caller. The per-model count remains internal.
    }

    /// Invalidate the transcript view cache (structural change).
    pub(crate) fn bump_transcript(&mut self) {
        self.transcript_gen = self.transcript_gen.wrapping_add(1);
    }

    /// Persist the current provider/model (and profile, when set) so the next
    /// bare `hi` in this workspace restores the same routing.
    pub(crate) fn remember_session_routing(&self) {
        let Some(cb) = &self.session_remember else {
            return;
        };
        let profile = self
            .active_profile
            .as_deref()
            .filter(|name| self.profiles.iter().any(|p| p.name == *name));
        cb(profile, &self.provider, &self.model);
    }

    /// Whether a confirmation should be skipped because of session-wide or
    /// path-scoped auto-approve.
    pub(crate) fn should_auto_approve(&self, request: &hi_harness::ConfirmationRequest) -> bool {
        if self.permission_mode == hi_harness::PermissionMode::Always {
            return true;
        }
        if self.permission_mode == hi_harness::PermissionMode::Auto && request.safe_for_auto() {
            return true;
        }
        match request {
            hi_harness::ConfirmationRequest::FileEdit { path, .. } => {
                self.auto_approve_session || self.path_auto_approved(path)
            }
            hi_harness::ConfirmationRequest::ShellMutation { .. } => false,
        }
    }

    pub(crate) fn mcp_auto_approved(&self, server: &str, tool: &str) -> bool {
        self.auto_approve_mcp
            .iter()
            .any(|(s, t)| s == server && t == tool)
    }

    pub(crate) fn add_auto_approve_mcp(&mut self, server: String, tool: String) {
        if !self.mcp_auto_approved(&server, &tool) {
            self.auto_approve_mcp.push((server, tool));
        }
    }

    pub(crate) fn path_auto_approved(&self, path: &str) -> bool {
        if self.auto_approve_paths.is_empty() || !Self::can_scope_auto_approve_path(path) {
            return false;
        }
        let path = path.replace('\\', "/");
        self.auto_approve_paths.iter().any(|prefix| {
            let p = prefix.replace('\\', "/");
            Self::can_scope_auto_approve_path(&p)
                && (path == p || path.starts_with(&format!("{p}/")))
        })
    }

    /// Path grants require a concrete normalized target. The agent supplies
    /// prepared canonical paths; legacy or unknown paths must not broaden a
    /// prefix grant through traversal or a shared multi-file placeholder.
    pub(crate) fn can_scope_auto_approve_path(path: &str) -> bool {
        let normalized = path.replace('\\', "/");
        !matches!(
            normalized.as_str(),
            "" | "/" | "(unknown)" | "(multiple files)"
        ) && !normalized
            .split('/')
            .any(|component| matches!(component, "." | ".."))
    }

    /// Remember a path prefix for session-scoped auto-approve (`p` on confirm).
    /// Uses the parent directory of a file path, or the path itself if it looks
    /// like a directory (no extension / trailing slash).
    pub(crate) fn add_auto_approve_path(&mut self, path: &str) {
        if !Self::can_scope_auto_approve_path(path) {
            return;
        }
        let normalized = path.replace('\\', "/");
        let prefix = {
            let trimmed = normalized.trim_end_matches('/');
            // Prefer the parent directory so "always allow src/" covers the file.
            match trimmed.rsplit_once('/') {
                Some((parent, _)) if !parent.is_empty() => parent.to_string(),
                _ => trimmed.to_string(),
            }
        };
        if prefix.is_empty() {
            return;
        }
        if !self.auto_approve_paths.iter().any(|p| p == &prefix) {
            self.auto_approve_paths.push(prefix);
        }
    }

    /// Path prefix label shown after `p` on a confirmation (for the status line).
    pub(crate) fn auto_approve_prefix_for(path: &str) -> String {
        let normalized = path.replace('\\', "/");
        let trimmed = normalized.trim_end_matches('/');
        match trimmed.rsplit_once('/') {
            Some((parent, _)) if !parent.is_empty() => parent.to_string(),
            _ => trimmed.to_string(),
        }
    }

    pub(crate) fn clamp_queue_selection(&mut self) {
        if self.queue.is_empty() {
            self.queue_selected = None;
            self.queue_paused = false;
            return;
        }
        if let Some(i) = self.queue_selected {
            self.queue_selected = Some(i.min(self.queue.len() - 1));
        }
    }

    pub(crate) fn resume_queue(&mut self) -> usize {
        let queued = self.queue.len();
        self.queue_paused = false;
        queued
    }

    /// Push a next-turn prompt. Empty / whitespace-only strings are ignored.
    /// The queue deliberately grows with accepted work so a long-running turn
    /// cannot silently discard distinct user or synthetic follow-ups.
    pub(crate) fn try_enqueue_prompt(&mut self, prompt: impl Into<String>) -> bool {
        let prompt = prompt.into();
        if prompt.trim().is_empty() {
            return false;
        }
        self.queue.push_back(prompt);
        self.clamp_queue_selection();
        if let Some(prompt) = self.queue.back() {
            self.trace_prompt_queued(prompt);
        }
        true
    }

    /// Interactive alias for [`Self::try_enqueue_prompt`].
    pub(crate) fn enqueue_prompt(&mut self, prompt: impl Into<String>) -> bool {
        let prompt = prompt.into();
        if prompt.trim().is_empty() {
            return false;
        }
        self.try_enqueue_prompt(prompt)
    }

    /// Insert at the front (next to run) without displacing queued work.
    pub(crate) fn enqueue_prompt_front(&mut self, prompt: impl Into<String>) -> bool {
        let prompt = prompt.into();
        if prompt.trim().is_empty() {
            return false;
        }
        self.queue.push_front(prompt);
        self.clamp_queue_selection();
        if let Some(prompt) = self.queue.front() {
            self.trace_prompt_queued(prompt);
        }
        true
    }

    pub(crate) fn queue_select_next(&mut self) {
        if self.queue.is_empty() {
            self.queue_selected = None;
            return;
        }
        let n = self.queue.len();
        self.queue_selected = Some(match self.queue_selected {
            Some(i) => (i + 1).min(n - 1),
            None => 0,
        });
    }

    pub(crate) fn queue_select_prev(&mut self) {
        if self.queue.is_empty() {
            self.queue_selected = None;
            return;
        }
        self.queue_selected = Some(match self.queue_selected {
            Some(0) | None => 0,
            Some(i) => i - 1,
        });
    }

    pub(crate) fn queue_remove_selected(&mut self) -> Option<String> {
        let i = self.queue_selected?;
        if i >= self.queue.len() {
            self.queue_selected = None;
            return None;
        }
        let removed = self.queue.remove(i);
        self.clamp_queue_selection();
        if let Some(prompt) = removed.as_deref() {
            self.trace_prompt_removed(prompt);
        }
        removed
    }

    pub(crate) fn queue_move_selected(&mut self, delta: i32) {
        let Some(i) = self.queue_selected else { return };
        if self.queue.len() < 2 {
            return;
        }
        let j = if delta < 0 {
            i.saturating_sub(1)
        } else {
            (i + 1).min(self.queue.len() - 1)
        };
        if i != j {
            self.queue.swap(i, j);
            self.queue_selected = Some(j);
        }
    }

    /// Grok `page_flip_on_send`: keep the just-sent prompt at the top of the
    /// transcript until the response fills the page, then follow the tail.
    pub(crate) fn apply_page_flip(&mut self, inner_h: u16, total: u16) {
        if !self.page_flip_on_send {
            return;
        }
        let Some(&idx) = self.view_cache.prompt_line_starts.last() else {
            return;
        };
        let prompt_row = self.view_cache.prefix.get(idx).copied().unwrap_or(0);
        let after = u32::from(total).saturating_sub(prompt_row);
        if inner_h > 0 && after >= u32::from(inner_h) {
            self.following = true;
            self.page_flip_on_send = false;
        } else {
            self.following = false;
            self.scroll = prompt_row.min(u32::from(u16::MAX)) as u16;
        }
    }

    pub(crate) fn context_pct(&self) -> Option<u64> {
        let window = self.context_window? as u64;
        if window == 0 {
            return None;
        }
        Some((self.context_used.saturating_mul(100) / window).min(100))
    }

    pub(crate) fn session_cost_chip(&self) -> Option<String> {
        let (input_rate, output_rate) = self.usage_pricing?;
        let cost = (self.session_totals.input_tokens as f64) * input_rate / 1_000_000.0
            + (self.session_totals.output_tokens as f64) * output_rate / 1_000_000.0;
        if cost <= 0.0 {
            return None;
        }
        Some(format!("${cost:.2}"))
    }

    pub(crate) fn composer_flags(&self) -> Vec<&'static str> {
        let mut flags = Vec::new();
        if self.plan_mode {
            flags.push("plan");
        }
        match self.permission_mode {
            hi_harness::PermissionMode::Ask => {}
            mode => flags.push(mode.label()),
        }
        flags
    }

    pub(crate) fn scroll_to_user_prompt(&mut self, index: usize) -> bool {
        let mut n = 0usize;
        for (i, entry) in self.transcript.iter().enumerate() {
            if matches!(entry, crate::TranscriptEntry::UserPrompt { .. }) {
                if n == index {
                    self.scroll_to(i as u16);
                    return true;
                }
                n += 1;
            }
        }
        false
    }

    fn trace_prompt_queued(&self, _prompt: &str) {}
    fn trace_prompt_removed(&self, _prompt: &str) {}

    /// Apply a completed `/login` poll. Returns the provider name when the
    /// credential landed.
    pub(crate) async fn poll_pending_login(&mut self) -> Option<String> {
        let finished = self
            .pending_login
            .as_ref()
            .is_some_and(|(_, task)| task.is_finished());
        if !finished {
            return None;
        }
        let (provider, task) = self.pending_login.take()?;
        match task.await {
            Ok(Ok(())) => {
                self.push(ratatui::text::Line::styled(
                    format!(
                        "signed in to {provider} — credential stored (run /logout {provider} to pair a different account)"
                    ),
                    crate::render::dim(),
                ));
                self.follow();
                Some(provider)
            }
            Ok(Err(error)) => {
                self.push(ratatui::text::Line::styled(
                    format!("/login {provider} failed: {error:#}"),
                    ratatui::style::Style::default().fg(crate::theme::theme().warning),
                ));
                self.follow();
                None
            }
            Err(error) if error.is_cancelled() => None,
            Err(error) => {
                self.push(ratatui::text::Line::styled(
                    format!("/login {provider} failed: {error}"),
                    ratatui::style::Style::default().fg(crate::theme::theme().warning),
                ));
                self.follow();
                None
            }
        }
    }
}

#[cfg(test)]
mod page_flip_tests {
    use crate::tests::test_app;
    use ratatui::text::Line;

    #[test]
    fn sending_a_prompt_pins_it_at_the_top() {
        let mut app = test_app("pipe", "m");
        app.following = true;
        app.push_user_prompt(Line::raw("❯ hello"));
        assert!(
            app.page_flip_on_send,
            "grok page_flip_on_send after a live send"
        );
        app.view_cache.prompt_line_starts = vec![0];
        app.view_cache.prefix = vec![0, 1];
        app.apply_page_flip(20, 1);
        assert!(
            !app.following,
            "short page stays top-aligned, not stuck to the composer"
        );
        assert_eq!(app.scroll, 0);
        assert!(app.page_flip_on_send);
    }

    #[test]
    fn sending_a_new_prompt_clears_a_finished_plan() {
        let mut app = test_app("pipe", "m");
        app.plan = vec![
            hi_tools::PlanStep {
                title: "read server.rs".into(),
                status: hi_tools::PlanStatus::Done,
            },
            hi_tools::PlanStep {
                title: "fix the loop".into(),
                status: hi_tools::PlanStatus::Done,
            },
        ];
        app.dismiss_completed_plan();
        assert!(app.plan.is_empty());
    }

    #[test]
    fn sending_a_new_prompt_keeps_an_open_plan() {
        let mut app = test_app("pipe", "m");
        app.plan = vec![hi_tools::PlanStep {
            title: "fix the loop".into(),
            status: hi_tools::PlanStatus::Active,
        }];
        app.dismiss_completed_plan();
        assert_eq!(app.plan.len(), 1);
    }

    #[test]
    fn page_flip_follows_once_the_response_fills_the_viewport() {
        let mut app = test_app("pipe", "m");
        app.push_user_prompt(Line::raw("❯ hello"));
        app.view_cache.prompt_line_starts = vec![0];
        app.view_cache.prefix = vec![0, 1];
        app.apply_page_flip(10, 30);
        assert!(app.following);
        assert!(!app.page_flip_on_send);
    }

    #[test]
    fn resume_follow_cancels_page_flip() {
        let mut app = test_app("pipe", "m");
        app.push_user_prompt(Line::raw("❯ old"));
        assert!(app.page_flip_on_send);
        app.follow();
        assert!(!app.page_flip_on_send);
        assert!(app.following);
    }
}
