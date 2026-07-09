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
            code_lang: None,
            input: InputLine::default(),
            following: true,
            scroll: 0,
            view_max_scroll: 0,
            view_total: 0,
            total_when_unpinned: 0,
            working: false,
            spinner: 0,
            started: None,
            current_tool: None,
            current_tool_started: None,
            pending_explore_label: None,
            explore_run: None,
            queue: VecDeque::new(),
            last_prompt: None,
            last_turn_start: 0,
            last_turn_snapshot: None,
            picker: None,
            provider_form: None,
            history_search: None,
            fetching: None,
            planning: None,
            status: String::new(),
            plan: Vec::new(),
            goal: None,
            goal_drive_stall: 0,
            usage: (0, 0),
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
            quit_notice: None,
            completion: None,
            focused: true,
            focus_known: false,
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

    /// Mark the turn as running (or done), stamping the start time so the
    /// prompt bar can show elapsed seconds.
    pub(crate) fn set_working(&mut self, working: bool) {
        self.working = working;
        self.started = working.then(Instant::now);
        self.current_tool = None;
        self.current_tool_started = None;
        self.pending_explore_label = None;
        self.explore_run = None;
        if working {
            self.last_turn_event = None;
            self.last_turn_had_file_edits = false;
            self.waiting_for = Some(Duration::ZERO);
            self.last_turn_state = TurnState::Running;
        } else if matches!(self.last_turn_state, TurnState::Running) {
            self.last_turn_state = TurnState::Idle;
            self.waiting_for = None;
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
