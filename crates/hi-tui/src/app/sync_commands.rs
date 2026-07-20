//! `App` methods: sync-related slash commands (`/sync`, `/sessions`, `/attach`,
//! `/daemon`).

use ratatui::style::{Style};
use ratatui::text::Line;

use crate::model_picker::ModelPicker;
use crate::render::dim;

#[derive(Clone, Debug)]
struct SyncedSessionInfo {
    id: String,
    title: String,
    status: String,
    records: u64,
    project: String,
    favorite: bool,
    archived: bool,
}

fn valid_session_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && !matches!(id, "." | "..")
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

impl crate::App {
    /// `/sync on|off|status` — toggle or query session sync to ipop.
    pub(crate) async fn handle_sync_command(&mut self, arg: &str) {
        match arg.trim() {
            "on" => {
                if let Some(control) = &self.sync_control
                    && let Err(error) = (control.set_mode)("on")
                {
                    self.push(Line::styled(format!("sync mode update failed: {error:#}"), dim()));
                    self.follow();
                    return;
                }
                if self.sync_config.is_some() {
                    // Need a session ID to stream events to. If sync was enabled
                    // at startup, this is already set. If not, we can't stream
                    // mid-session without one — the user needs to restart with
                    // --sync-session-id or --sync.
                    let session_id = match &self.sync_session_id {
                        Some(id) if !id.is_empty() => id.clone(),
                        _ => {
                            self.push(Line::styled(
                                "✗ no sync session ID — start hi with --sync to enable \
                                 mid-session streaming",
                                Style::default().fg(crate::theme::theme().warning),
                            ));
                            self.follow();
                            return;
                        }
                    };
                    self.sync_active = true;
                    // Create the remote event tap so live events are forwarded to ipop.
                    // The tap calls push_event on a RemoteUi, which buffers events
                    // for the next flush. We use the sync_config to construct it.
                    // Note: the actual RemoteSessionSink for durable record sync
                    // is created in main.rs at startup; this tap handles live events.
                    // If sync wasn't enabled at startup, we can only stream events
                    // (not durable records) from this point — a full sync requires
                    // restarting with --sync.
                    if self.remote_flush_callback.is_none() {
                        let config = self.sync_config.clone().unwrap();
                        let rui = std::sync::Arc::new(crate::sync_tui::RemoteUi::new(
                            crate::sync_tui::SyncConfig {
                                base_url: config.base_url,
                                api_key: config.api_key,
                            },
                            session_id,
                        ));
                        let rui_clone = rui.clone();
                        let tap: std::sync::Arc<dyn Fn(&crate::event::UiEvent) + Send + Sync> =
                            std::sync::Arc::new(move |event: &crate::event::UiEvent| {
                                rui_clone.push_event(event.clone());
                            });
                        self.remote_event_tap = Some(tap);
                        self.sync_remote_ui = Some(rui);
                    }
                    self.push(Line::styled(
                        "✓ sync on — retained records/events and future portal data will upload",
                        Style::default().fg(crate::theme::theme().accent_success),
                    ));
                } else {
                    self.push(Line::styled(
                        "✗ sync not configured — set HI_SYNC_BASE_URL and HI_SYNC_API_KEY, \
                         or add a [sync] section to hi.toml",
                        Style::default().fg(crate::theme::theme().warning),
                    ));
                }
            }
            "paused" => {
                if let Some(control) = &self.sync_control {
                    let _ = (control.set_mode)("paused");
                }
                self.sync_active = false;
                if self.remote_flush_callback.is_none()
                    && self.sync_remote_ui.is_none()
                    && let (Some(config), Some(session_id)) =
                        (self.sync_config.clone(), self.sync_session_id.clone())
                {
                    let remote = std::sync::Arc::new(crate::sync_tui::RemoteUi::new(
                        crate::sync_tui::SyncConfig {
                            base_url: config.base_url,
                            api_key: config.api_key,
                        },
                        session_id,
                    ));
                    let tap_remote = remote.clone();
                    self.remote_event_tap = Some(std::sync::Arc::new(move |event| {
                        tap_remote.push_event(event.clone());
                    }));
                    self.sync_remote_ui = Some(remote);
                }
                self.push(Line::styled(
                    "sync paused — records and bounded live events remain queued; network activity stopped",
                    dim(),
                ));
            }
            "off" => {
                if let Some(control) = &self.sync_control {
                    let _ = (control.set_mode)("off");
                }
                self.sync_active = false;
                self.sync_remote_ui = None;
                self.push(Line::styled(
                    "sync off — no portal data will be enqueued or sent; the existing queue is retained",
                    dim(),
                ));
            }
            "" | "status" => {
                if let Some(control) = &self.sync_control {
                    match (control.status)(self.sync_session_id.as_deref()) {
                        Ok(status) => self.push(Line::styled(format!("sync: {status}"), dim())),
                        Err(error) => self.push(Line::styled(
                            format!("sync status unavailable: {error:#}"),
                            Style::default().fg(crate::theme::theme().warning),
                        )),
                    }
                    self.follow();
                    return;
                }
                if self.sync_config.is_some() {
                    let status = if self.sync_active {
                        "active"
                    } else if self.remote_flush_callback.is_some() {
                        "records active, live events paused"
                    } else {
                        "paused"
                    };
                    self.push(Line::styled(
                        format!(
                            "sync: {status} · endpoint: {} · session: {}",
                            self.sync_config.as_ref().unwrap().base_url,
                            self.sync_session_id.as_deref().unwrap_or("(not set)"),
                        ),
                        dim(),
                    ));
                } else {
                    self.push(Line::styled(
                        "sync: not configured (set HI_SYNC_BASE_URL and HI_SYNC_API_KEY)",
                        dim(),
                    ));
                }
            }
            "purge" => self.push(Line::styled(
                "purge permanently removes the retained portal queue; run `/sessions sync purge confirm`",
                Style::default().fg(crate::theme::theme().warning),
            )),
            "purge confirm" => {
                match &self.sync_control {
                    Some(control) => match (control.purge)() {
                        Ok(()) => self.push(Line::styled("✓ portal sync queue purged", dim())),
                        Err(error) => self.push(Line::styled(
                            format!("sync purge failed: {error:#}"),
                            Style::default().fg(crate::theme::theme().warning),
                        )),
                    },
                    None => self.push(Line::styled("sync persistence is unavailable", dim())),
                }
            }
            other => {
                self.push(Line::styled(
                    format!("usage: /sync on|paused|off|status|purge (got '{other}')"),
                    dim(),
                ));
            }
        }
        self.follow();
    }

    /// `/sessions` owns the complete session-management surface: list, switch,
    /// and rename.
    pub(crate) async fn handle_sessions_command(&mut self, agent: &mut hi_agent::Agent, arg: &str) {
        match arg.trim() {
            "" => self.list_sessions().await,
            value if value == "sync" || value.starts_with("sync ") => {
                let sync_arg = value.strip_prefix("sync").unwrap_or("").trim();
                self.handle_sync_command(sync_arg).await;
            }
            value if value == "attach" || value.starts_with("attach ") => {
                let session_id = value.strip_prefix("attach").unwrap_or("").trim();
                self.handle_attach_command(agent, session_id).await;
            }
            value if value == "host" || value.starts_with("host ") => {
                let host_arg = value.strip_prefix("host").unwrap_or("").trim();
                self.handle_daemon_command(host_arg).await;
            }
            value if value == "switch" || value.starts_with("switch ") => {
                let session_id = value.strip_prefix("switch").unwrap_or("").trim();
                self.switch_session(agent, session_id).await;
            }
            value if value == "rename" || value.starts_with("rename ") => {
                let rest = value.strip_prefix("rename").unwrap_or("").trim();
                let Some((session_id, name)) = rest.split_once(char::is_whitespace) else {
                    self.push(Line::styled(
                        "usage: /sessions rename <session-id> <name>",
                        dim(),
                    ));
                    self.follow();
                    return;
                };
                self.rename_session(session_id, name.trim()).await;
            }
            value if value.starts_with("favorite ") => {
                self.patch_session(
                    value.trim_start_matches("favorite ").trim(),
                    serde_json::json!({"favorite": true}),
                )
                .await;
            }
            value if value.starts_with("archive ") => {
                self.patch_session(
                    value.trim_start_matches("archive ").trim(),
                    serde_json::json!({"archived": true}),
                )
                .await;
            }
            value if value.starts_with("restore ") => {
                self.patch_session(
                    value.trim_start_matches("restore ").trim(),
                    serde_json::json!({"archived": false}),
                )
                .await;
            }
            value if value.starts_with("delete ") => {
                let rest = value.trim_start_matches("delete ").trim();
                let Some(id) = rest.strip_suffix(" confirm").map(str::trim) else {
                    self.push(Line::styled(
                        format!("permanent deletion requires `/sessions delete {rest} confirm`"),
                        Style::default().fg(crate::theme::theme().warning),
                    ));
                    self.follow();
                    return;
                };
                self.delete_session(id).await;
            }
            other => {
                self.push(Line::styled(
                    format!(
                        "usage: /sessions [switch <id>|rename <id> <name>|favorite <id>|archive <id>|restore <id>|delete <id> confirm|attach <id>|host [on|off|status]|sync on|paused|off|status|purge] (got '{other}')"
                    ),
                    dim(),
                ));
            }
        }
        self.follow();
    }

    pub(crate) async fn switch_session(&mut self, agent: &mut hi_agent::Agent, session_id: &str) {
        if session_id.is_empty() {
            self.push(Line::styled("usage: /sessions switch <session-id>", dim()));
            self.follow();
            return;
        }
        if !valid_session_id(session_id) {
            self.push(Line::styled(
                "invalid session id",
                Style::default().fg(crate::theme::theme().warning),
            ));
            self.follow();
            return;
        }
        if self.sync_session_id.as_deref() == Some(session_id) {
            self.push(Line::styled(
                format!("session {session_id} is already active"),
                dim(),
            ));
            self.follow();
            return;
        }

        // Temporarily take the callback to avoid borrowing `self` immutably
        // while resetting the UI after it mutates the agent.
        let Some(switcher) = self.session_switcher.take() else {
            self.push(Line::styled(
                "session switching is unavailable in this mode",
                Style::default().fg(crate::theme::theme().warning),
            ));
            self.follow();
            return;
        };
        let result = switcher(session_id, agent).await;
        self.session_switcher = Some(switcher);

        match result {
            Ok(switched) => {
                self.transcript.clear();
                self.event_log.clear();
                self.pending = None;
                self.code_lang = None;
                self.current_assistant.clear();
                self.last_assistant.clear();
                self.status.clear();
                self.last_turn_state = crate::TurnState::Idle;
                self.last_prompt = None;
                self.last_turn_snapshot = None;
                self.last_turn_start = agent.messages().len();
                self.queue.clear();
                self.mid_turn_offered.clear();
                self.plan = agent.current_plan().to_vec();
                self.goal = agent.structured_goal().cloned();
                self.goal_drive_stall = 0;
                self.usage = (0, 0);
                self.usage_estimated = false;
                self.context_used = 0;
                // Switching sessions drops host-mode for the previous id; the
                // new session must opt in again with `/sessions host on`.
                self.stop_host_mode();
                self.sync_session_id = Some(switched.id.clone());
                // `/sync off` followed by `/sync on` owns a TUI-local event
                // streamer. Rebind it when the session changes so live events
                // cannot continue landing under the previous session id.
                if self.sync_remote_ui.is_some()
                    && let Some(config) = self.sync_config.clone()
                {
                    if let Some(previous) = self.sync_remote_ui.take() {
                        tokio::spawn(async move {
                            let _ = previous.flush().await;
                        });
                    }
                    let remote = std::sync::Arc::new(crate::sync_tui::RemoteUi::new(
                        crate::sync_tui::SyncConfig {
                            base_url: config.base_url,
                            api_key: config.api_key,
                        },
                        switched.id.clone(),
                    ));
                    let tap_remote = remote.clone();
                    self.remote_event_tap = Some(std::sync::Arc::new(move |event| {
                        tap_remote.push_event(event.clone());
                    }));
                    self.sync_remote_ui = Some(remote);
                }
                // Replay the adopted history into the transcript so the user
                // sees the remote conversation instead of a blank pane.
                self.replay_agent_history(agent);
                self.push(Line::styled(
                    format!("✓ switched to session {}", switched.id),
                    Style::default().fg(crate::theme::theme().accent_success),
                ));
                self.push(Line::styled(switched.summary, dim()));
                self.push(Line::styled(
                    "  remote resume ready — type to continue, or `/sessions host on` to accept remote prompts",
                    dim(),
                ));
            }
            Err(err) => self.push(Line::styled(
                format!("session switch failed: {err:#}"),
                Style::default().fg(crate::theme::theme().warning),
            )),
        }
        self.follow();
    }

    async fn rename_session(&mut self, session_id: &str, name: &str) {
        if session_id.is_empty() || name.is_empty() {
            self.push(Line::styled(
                "usage: /sessions rename <session-id> <name>",
                dim(),
            ));
            self.follow();
            return;
        }
        if !valid_session_id(session_id) {
            self.push(Line::styled(
                "invalid session id",
                Style::default().fg(crate::theme::theme().warning),
            ));
            self.follow();
            return;
        }
        if name.chars().count() > 120 {
            self.push(Line::styled(
                "session name must be at most 120 characters",
                Style::default().fg(crate::theme::theme().warning),
            ));
            self.follow();
            return;
        }

        let cached = self
            .session_lister
            .as_ref()
            .is_some_and(|lister| lister().iter().any(|session| session.id == session_id));
        if cached {
            let Some(renamer) = self.session_renamer.take() else {
                self.push(Line::styled(
                    "session renaming is unavailable in this mode",
                    Style::default().fg(crate::theme::theme().warning),
                ));
                self.follow();
                return;
            };
            let result = renamer(session_id, name);
            self.session_renamer = Some(renamer);
            if let Err(err) = result {
                self.push(Line::styled(
                    format!("session rename failed: {err:#}"),
                    Style::default().fg(crate::theme::theme().warning),
                ));
                self.follow();
                return;
            }
        }

        let mut synced = false;
        if let (Some(config), Some(client)) = (&self.sync_config, &self.sync_http) {
            match client
                .post(format!(
                    "{}/hi/sessions/{session_id}/rename",
                    config.base_url
                ))
                .header("x-api-key", &config.api_key)
                .json(&serde_json::json!({ "title": name }))
                .send()
                .await
            {
                Ok(response) if response.status().is_success() => synced = true,
                Ok(response) if response.status() == reqwest::StatusCode::NOT_FOUND && cached => {}
                Ok(response) => {
                    if cached {
                        self.push(Line::styled(
                            format!("✓ renamed session {session_id} → {name}"),
                            Style::default().fg(crate::theme::theme().accent_success),
                        ));
                    }
                    self.push(Line::styled(
                        format!("session sync update failed with HTTP {}", response.status()),
                        Style::default().fg(crate::theme::theme().warning),
                    ));
                    self.follow();
                    return;
                }
                Err(err) if cached => self.push(Line::styled(
                    format!("session renamed; sync update failed: {err}"),
                    Style::default().fg(crate::theme::theme().warning),
                )),
                Err(err) => {
                    self.push(Line::styled(
                        format!("session rename failed: {err}"),
                        Style::default().fg(crate::theme::theme().warning),
                    ));
                    self.follow();
                    return;
                }
            }
        }
        if !cached && !synced {
            self.push(Line::styled(
                format!("session '{session_id}' was not found"),
                Style::default().fg(crate::theme::theme().warning),
            ));
        } else {
            self.push(Line::styled(
                format!("✓ renamed session {session_id} → {name}"),
                Style::default().fg(crate::theme::theme().accent_success),
            ));
        }
        self.follow();
    }

    pub(crate) async fn patch_session(&mut self, session_id: &str, body: serde_json::Value) {
        if !valid_session_id(session_id) {
            self.push(Line::styled(
                "invalid session id",
                Style::default().fg(crate::theme::theme().warning),
            ));
            return;
        }
        let (Some(config), Some(client)) = (&self.sync_config, &self.sync_http) else {
            self.push(Line::styled("session catalog is unavailable", dim()));
            return;
        };
        match client
            .patch(format!("{}/hi/sessions/{session_id}", config.base_url))
            .header("x-api-key", &config.api_key)
            .json(&body)
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => self.push(Line::styled(
                format!("✓ updated session {session_id}"),
                Style::default().fg(crate::theme::theme().accent_success),
            )),
            Ok(response) => self.push(Line::styled(
                format!("session update failed with HTTP {}", response.status()),
                Style::default().fg(crate::theme::theme().warning),
            )),
            Err(error) => self.push(Line::styled(
                format!("session update failed: {error}"),
                Style::default().fg(crate::theme::theme().warning),
            )),
        }
    }

    pub(crate) async fn delete_session(&mut self, session_id: &str) {
        if !valid_session_id(session_id) {
            self.push(Line::styled(
                "invalid session id",
                Style::default().fg(crate::theme::theme().warning),
            ));
            return;
        }
        let (Some(config), Some(client)) = (&self.sync_config, &self.sync_http) else {
            self.push(Line::styled("session catalog is unavailable", dim()));
            return;
        };
        match client
            .delete(format!("{}/hi/sessions/{session_id}", config.base_url))
            .header("x-api-key", &config.api_key)
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => self.push(Line::styled(
                format!("✓ permanently deleted session {session_id}"),
                Style::default().fg(crate::theme::theme().accent_success),
            )),
            Ok(response) => self.push(Line::styled(
                format!("session delete failed with HTTP {}", response.status()),
                Style::default().fg(crate::theme::theme().warning),
            )),
            Err(error) => self.push(Line::styled(
                format!("session delete failed: {error}"),
                Style::default().fg(crate::theme::theme().warning),
            )),
        }
    }

    /// `/attach <session-id>` / `/sessions attach <id>` — resume a remote
    /// session inside this TUI (fetch history if needed, claim writer lease,
    /// continue with the live agent). Same path as `/sessions switch`.
    pub(crate) async fn handle_attach_command(
        &mut self,
        agent: &mut hi_agent::Agent,
        arg: &str,
    ) {
        let session_id = arg.trim();
        if session_id.is_empty() {
            self.push(Line::styled(
                "usage: /attach <session-id>  (or /sessions attach <id> / /sessions switch <id>)",
                dim(),
            ));
            self.follow();
            return;
        }
        if self.sync_config.is_none() {
            self.push(Line::styled(
                "sync is not configured — set [sync] / HI_SYNC_* or run with --sync",
                Style::default().fg(crate::theme::theme().warning),
            ));
            self.follow();
            return;
        }
        self.push(Line::styled(
            format!("attaching to session {session_id}…"),
            dim(),
        ));
        self.follow();
        self.switch_session(agent, session_id).await;
    }

    /// `/sessions host [on|off|status]` — advertise remote-input acceptance and
    /// long-poll attach prompts into the local turn queue. Replaces the old
    /// "exit and run hi --daemon" hand-off for interactive use.
    pub(crate) async fn handle_daemon_command(&mut self, arg: &str) {
        let action = match arg.trim() {
            "" | "on" | "start" | "enable" => "on",
            "off" | "stop" | "disable" => "off",
            "status" => "status",
            other => {
                self.push(Line::styled(
                    format!("usage: /sessions host [on|off|status] (got '{other}')"),
                    dim(),
                ));
                self.follow();
                return;
            }
        };

        if action == "status" {
            let state = if self.hosting_remote_input {
                "on — accepting remote prompts for this session"
            } else {
                "off"
            };
            self.push(Line::styled(
                format!(
                    "host: {state}{}",
                    self.sync_session_id
                        .as_deref()
                        .map(|id| format!(" · session {id}"))
                        .unwrap_or_default()
                ),
                dim(),
            ));
            self.follow();
            return;
        }

        let enable = action == "on";
        if enable && self.hosting_remote_input {
            self.push(Line::styled(
                "already hosting remote input for this session",
                dim(),
            ));
            self.follow();
            return;
        }
        if !enable && !self.hosting_remote_input {
            self.push(Line::styled("host mode is already off", dim()));
            self.follow();
            return;
        }

        let Some(controller) = self.session_host.take() else {
            self.push(Line::styled(
                "host mode unavailable — enable sync first (`/sessions sync on`)",
                Style::default().fg(crate::theme::theme().warning),
            ));
            self.follow();
            return;
        };
        let result = controller(enable).await;
        self.session_host = Some(controller);

        match result {
            Ok(enabled) => {
                self.stop_host_mode();
                if let Some((rx, abort)) = enabled {
                    self.remote_input_rx = Some(rx);
                    self.remote_input_poller = Some(abort);
                    self.hosting_remote_input = true;
                    self.push(Line::styled(
                        "✓ host on — remote attach clients can send prompts into this session",
                        Style::default().fg(crate::theme::theme().accent_success),
                    ));
                    self.push(Line::styled(
                        "  other machines: /sessions attach <id>  (or hi --attach <id>)",
                        dim(),
                    ));
                } else {
                    self.push(Line::styled(
                        "host off — no longer accepting remote prompts",
                        dim(),
                    ));
                }
            }
            Err(err) => self.push(Line::styled(
                format!("host mode failed: {err:#}"),
                Style::default().fg(crate::theme::theme().warning),
            )),
        }
        self.follow();
    }

    fn stop_host_mode(&mut self) {
        if let Some(abort) = self.remote_input_poller.take() {
            abort.abort();
        }
        self.remote_input_rx = None;
        self.hosting_remote_input = false;
    }

    /// Drain any remote attach prompts into the local turn queue. Returns true
    /// when at least one prompt was enqueued (caller should leave the idle
    /// input wait and run the queue).
    pub(crate) fn drain_remote_input(&mut self) -> bool {
        let Some(rx) = self.remote_input_rx.as_mut() else {
            return false;
        };
        let mut queued = 0usize;
        loop {
            match rx.try_recv() {
                Ok(prompt) => {
                    let prompt = prompt.trim().to_string();
                    if prompt.is_empty() {
                        continue;
                    }
                    self.queue.push_back(prompt);
                    queued += 1;
                }
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                    self.stop_host_mode();
                    break;
                }
            }
        }
        if queued > 0 {
            self.push(Line::styled(
                format!(
                    "← {queued} remote prompt{} queued",
                    if queued == 1 { "" } else { "s" }
                ),
                dim(),
            ));
            self.follow();
            true
        } else {
            false
        }
    }

    /// Push a compact transcript of the agent's loaded history after a session
    /// switch/attach so the pane isn't blank.
    fn replay_agent_history(&mut self, agent: &hi_agent::Agent) {
        let mut replayed = 0usize;
        for message in agent.messages() {
            match message.role {
                hi_ai::Role::User => {
                    let text = message.text();
                    if text.trim().is_empty() {
                        continue;
                    }
                    self.push(Line::styled(
                        format!("you: {text}"),
                        Style::default().fg(crate::theme::theme().accent_user),
                    ));
                    replayed += 1;
                }
                hi_ai::Role::Assistant => {
                    let text = message.text();
                    if text.trim().is_empty() {
                        continue;
                    }
                    // Keep replay compact — show the first ~12 lines of long answers.
                    let mut lines = text.lines();
                    if let Some(first) = lines.next() {
                        self.push(Line::styled(
                            format!("hi: {first}"),
                            Style::default().fg(crate::theme::theme().accent_assistant),
                        ));
                        let mut extra = 0usize;
                        for line in lines.by_ref().take(11) {
                            self.push(Line::styled(
                                format!("    {line}"),
                                Style::default().fg(crate::theme::theme().accent_assistant),
                            ));
                            extra += 1;
                        }
                        if lines.next().is_some() {
                            self.push(Line::styled("    …", dim()));
                        }
                        let _ = extra;
                    }
                    self.last_assistant = text;
                    replayed += 1;
                }
                hi_ai::Role::System | hi_ai::Role::Tool => {}
            }
        }
        if replayed > 0 {
            self.push(Line::styled(
                format!("— resumed {replayed} prior messages —"),
                dim(),
            ));
        }
        self.bump_transcript();
    }

    async fn list_sessions(&mut self) {
        let cached = self
            .session_lister
            .as_ref()
            .map(|lister| lister())
            .unwrap_or_default();
        let synced_result = self.fetch_synced_sessions().await;
        let synced = synced_result
            .as_ref()
            .map(Vec::as_slice)
            .unwrap_or_default();
        let mut seen = std::collections::HashSet::new();
        let mut completion = Vec::new();
        let total = cached
            .iter()
            .map(|session| session.id.as_str())
            .chain(synced.iter().map(|session| session.id.as_str()))
            .collect::<std::collections::HashSet<_>>()
            .len();

        if total == 0 {
            self.push(Line::styled("sessions: (none)", dim()));
        } else {
            self.push(Line::styled(format!("sessions ({total}):"), dim()));
        }

        for session in cached {
            seen.insert(session.id.clone());
            let synced_match = synced.iter().find(|item| item.id == session.id);
            let title = synced_match
                .filter(|item| !item.title.is_empty())
                .map(|item| item.title.clone())
                .unwrap_or_else(|| session.title.clone());
            let marker = if self.sync_session_id.as_deref() == Some(session.id.as_str()) {
                "●"
            } else {
                "○"
            };
            self.push(Line::styled(
                format!(
                    "  {marker} {}{}  · Enter / /sessions attach {}",
                    session.id,
                    if title.is_empty() {
                        String::new()
                    } else {
                        format!(": {title}")
                    },
                    session.id
                ),
                dim(),
            ));
            completion.push(crate::LocalSessionInfo { title, ..session });
        }
        let mut last_project = None::<&str>;
        for session in synced.iter().filter(|session| !seen.contains(&session.id)) {
            if last_project != Some(session.project.as_str()) {
                self.push(Line::styled(
                    format!("  project {}", session.project),
                    dim(),
                ));
                last_project = Some(&session.project);
            }
            let marker = if self.sync_session_id.as_deref() == Some(session.id.as_str()) {
                "●"
            } else {
                "○"
            };
            self.push(Line::styled(
                format!(
                    "  {marker} {}{}{}{}  · Enter / /sessions attach {}",
                    session.id,
                    if session.title.is_empty() {
                        String::new()
                    } else {
                        format!(": {}", session.title)
                    },
                    if session.favorite { " ★" } else { "" },
                    if session.archived { " [archived]" } else { "" },
                    session.id
                ),
                dim(),
            ));
            completion.push(crate::LocalSessionInfo {
                id: session.id.clone(),
                title: session.title.clone(),
                age: session.status.clone(),
                lines: session.records as usize,
            });
        }
        self.session_catalog_flags = synced
            .iter()
            .map(|session| (session.id.clone(), (session.favorite, session.archived)))
            .collect();
        let ids = completion
            .iter()
            .map(|session| session.id.clone())
            .collect::<Vec<_>>();
        self.picker = Some(ModelPicker::new(
            ids,
            self.sync_session_id.as_deref().unwrap_or_default(),
            std::collections::HashMap::new(),
            &self.served,
        ));
        self.session_picker = true;
        self.session_picker_searching = false;
        self.session_delete_pending = None;
        self.session_completion_cache = completion;

        if let Err(err) = synced_result {
            self.push(Line::styled(
                format!("session sync unavailable: {err}"),
                Style::default().fg(crate::theme::theme().warning),
            ));
        }
    }

    /// Fetch synced session metadata for merging into the one session view.
    async fn fetch_synced_sessions(&self) -> anyhow::Result<Vec<SyncedSessionInfo>> {
        let Some(config) = &self.sync_config else {
            return Ok(Vec::new());
        };
        let Some(client) = &self.sync_http else {
            return Ok(Vec::new());
        };

        let url = format!("{}/hi/sessions", config.base_url);
        let mut cursor: Option<String> = None;
        let mut sessions = Vec::new();
        loop {
            let mut request = client
                .get(&url)
                .header("x-api-key", &config.api_key)
                .query(&[("limit", "100")]);
            if let Some(value) = &cursor {
                request = request.query(&[("cursor", value)]);
            }
            let response = request
                .send()
                .await
                .map_err(|e| anyhow::anyhow!("fetch failed: {e}"))?;
            if !response.status().is_success() {
                anyhow::bail!("HTTP {}", response.status());
            }
            let body: serde_json::Value = response.json().await?;
            sessions.extend(
                body["sessions"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|session| {
                        Some(SyncedSessionInfo {
                            id: session["session_id"].as_str()?.to_string(),
                            title: session["title"].as_str().unwrap_or("").to_string(),
                            status: session["status"].as_str().unwrap_or("saved").to_string(),
                            records: session["record_count"].as_u64().unwrap_or(0),
                            project: session["project_fingerprint"]
                                .as_str()
                                .map(|value| value.chars().take(8).collect())
                                .unwrap_or_else(|| "local".to_string()),
                            favorite: session["favorite"].as_bool().unwrap_or(false),
                            archived: !session["archived_at_unix"].is_null(),
                        })
                    }),
            );
            if !body["has_more"].as_bool().unwrap_or(false) {
                break;
            }
            cursor = body["next_cursor"].as_str().map(str::to_string);
            if cursor.is_none() {
                break;
            }
        }
        sessions.sort_by(|a, b| {
            a.project
                .cmp(&b.project)
                .then_with(|| b.favorite.cmp(&a.favorite))
        });
        Ok(sessions)
    }
}
