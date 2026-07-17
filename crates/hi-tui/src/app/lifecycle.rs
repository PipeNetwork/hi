//! `App` methods: lifecycle.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use crate::input::InputLine;
use crate::util::notify_done;
use crate::{
    MlxProfileSwitcher, NOTIFY_THRESHOLD, ProfileInfo, ProfileLoader, ProfileRemover,
    ProfileResolver, ProfileSaver, TurnState,
};

impl crate::App {
    pub(crate) fn resume_goal_drive(&mut self, agent: &hi_agent::Agent) {
        self.refresh_goal(agent);
        self.maybe_queue_goal_drive(agent);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        provider: &str,
        model: &str,
        profiles: Vec<ProfileInfo>,
        active_profile: Option<String>,
        resolver: ProfileResolver,
        saver: ProfileSaver,
        loader: ProfileLoader,
        remover: ProfileRemover,
        mlx_switcher: MlxProfileSwitcher,
        mcp_url: Option<String>,
        api_key: String,
    ) -> Self {
        Self {
            provider: provider.to_string(),
            model: model.to_string(),
            workspace_root: std::path::PathBuf::new(),
            interrupt: None,
            active_profile,
            profiles,
            resolver,
            saver,
            loader,
            remover,
            mlx_switcher,
            mcp_url,
            api_key,
            transcript: Vec::new(),
            pending: None,
            reasoning_buffer: String::new(),
            reasoning_started: None,
            show_reasoning: false,
            show_tool_output: false,
            nav_mode: false,
            block_cursor: 0,
            code_lang: None,
            last_code_block: None,
            table_buf: Vec::new(),
            input: InputLine::default(),
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
            total_when_unpinned: 0,
            working: false,
            spinner: 0,
            started: None,
            finished_at: None,
            current_tool: None,
            current_tool_started: None,
            pending_explore_label: None,
            explore_run: None,
            queue: VecDeque::new(),
            last_prompt: None,
            last_turn_start: 0,
            last_turn_snapshot: None,
            picker: None,
            session_picker: false,
            session_picker_searching: false,
            session_catalog_flags: HashMap::new(),
            session_delete_pending: None,
            provider_form: None,
            history_search: None,
            fetching: None,
            planning: None,
            status: String::new(),
            plan: Vec::new(),
            confirmation: None,
            confirmation_scroll: 0,
            goal: None,
            goal_drive_stall: 0,
            fleet: Vec::new(),
            fleet_next_id: 0,
            loops: None,
            usage: (0, 0),
            usage_estimated: false,
            context_used: 0,
            context_window: None,
            rate_limits: None,
            served: HashMap::new(),
            model_ids: Vec::new(),
            trimmed: 0,
            current_assistant: String::new(),
            last_assistant: String::new(),
            last_turn_event: None,
            last_turn_had_file_edits: false,
            last_changed_files: Vec::new(),
            show_diff: false,
            diff_text: None,
            show_review: false,
            review_scroll: 0,
            auto_approve_session: false,
            show_debug: false,
            show_help: false,
            last_telemetry: None,
            turn_tool_calls: 0,
            turn_rounds: 0,
            tool_stream_tail: Vec::new(),
            waiting_for: None,
            last_turn_state: TurnState::Idle,
            last_error: None,
            event_log: Vec::new(),
            model_issues: HashMap::new(),
            startup_notice: None,
            checkpoint_warning: None,
            quit_notice: None,
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
            session_switcher: None,
            session_renamer: None,
            sync_control: None,
            remote_event_tap: None,
            sync_remote_ui: None,
            remote_flush_callback: None,
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

    /// Surface any completed `/loop` firings: quiet checks land dim, changes
    /// land loud (cyan) and ping the terminal when you're unfocused. Called
    /// from the UI tick arms so results appear even while idle.
    pub(crate) fn drain_loops(&mut self) {
        let Some(loops) = &self.loops else { return };
        let lines = loops.drain();
        if lines.is_empty() {
            return;
        }
        let away = self.focus_known && !self.focused;
        for (text, loud) in lines {
            let style = if loud {
                ratatui::style::Style::default().fg(ratatui::style::Color::Cyan)
            } else {
                crate::render::dim()
            };
            self.push(ratatui::text::Line::styled(text, style));
            if loud && away {
                crate::util::notify_done();
            }
        }
    }

    /// Mark the turn as running (or done), stamping the start time so the
    /// prompt bar can show elapsed seconds.
    pub(crate) fn set_working(&mut self, working: bool) {
        let was_working = self.working;
        self.working = working;
        self.started = working.then(Instant::now);
        self.current_tool = None;
        self.current_tool_started = None;
        self.pending_explore_label = None;
        self.explore_run = None;
        if working {
            self.checkpoint_warning = None;
            self.last_turn_event = None;
            self.last_turn_had_file_edits = false;
            self.waiting_for = Some(Duration::ZERO);
            self.last_turn_state = TurnState::Running;
            // A new turn's output would shift block ordinals and line indices;
            // leave nav mode and drop any stale text selection.
            self.nav_mode = false;
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

    pub(crate) fn record_model_issue(&mut self) {
        let _count = {
            let entry = self.model_issues.entry(self.model.clone()).or_insert(0);
            *entry += 1;
            *entry
        };
        // Note: don't touch `last_error` here — it holds the actual failure
        // reason set by the caller. The per-model count remains internal.
    }
}
