//! `App` methods: sync-related slash commands (`/sync`, `/sessions`, `/attach`,
//! `/daemon`).

use ratatui::style::{Color, Style};
use ratatui::text::Line;

use crate::render::dim;

#[derive(Clone, Debug)]
struct SyncedSessionInfo {
    id: String,
    title: String,
    status: String,
    records: u64,
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
                                Style::default().fg(Color::Yellow),
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
                    // Store the RemoteUi on the app so it can be flushed after turns.
                    self.sync_remote_ui = Some(rui);
                    self.push(Line::styled(
                        "✓ sync active — live events are streaming to ipop\n  \
                         (durable record sync requires restart with --sync for full history)",
                        Style::default().fg(Color::Green),
                    ));
                } else {
                    self.push(Line::styled(
                        "✗ sync not configured — set HI_SYNC_BASE_URL and HI_SYNC_API_KEY, \
                         or add a [sync] section to hi.toml",
                        Style::default().fg(Color::Yellow),
                    ));
                }
            }
            "off" => {
                self.sync_active = false;
                self.remote_event_tap = None;
                // Flush any remaining events before stopping. Spawn as a
                // background task so a slow ipop doesn't block the TUI.
                if let Some(rui) = self.sync_remote_ui.take() {
                    tokio::spawn(async move {
                        let _ = rui.flush().await;
                    });
                }
                let message = if self.remote_flush_callback.is_some() {
                    "live event streaming paused; durable record sync remains active until exit"
                } else {
                    "sync paused (session continues)"
                };
                self.push(Line::styled(message, dim()));
            }
            "" | "status" => {
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
            other => {
                self.push(Line::styled(
                    format!("usage: /sync on|off|status (got '{other}')"),
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
                self.handle_attach_command(session_id).await;
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
            other => {
                self.push(Line::styled(
                    format!(
                        "usage: /sessions [switch <id>|rename <id> <name>|attach <id>|host|sync on|off|status] (got '{other}')"
                    ),
                    dim(),
                ));
            }
        }
        self.follow();
    }

    async fn switch_session(&mut self, agent: &mut hi_agent::Agent, session_id: &str) {
        if session_id.is_empty() {
            self.push(Line::styled("usage: /sessions switch <session-id>", dim()));
            self.follow();
            return;
        }
        if !valid_session_id(session_id) {
            self.push(Line::styled(
                "invalid session id",
                Style::default().fg(Color::Yellow),
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
                Style::default().fg(Color::Yellow),
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
                self.plan.clear();
                self.goal = agent.structured_goal().cloned();
                self.goal_drive_stall = 0;
                self.usage = (0, 0);
                self.context_used = 0;
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
                self.push(Line::styled(
                    format!("✓ switched to session {}", switched.id),
                    Style::default().fg(Color::Green),
                ));
                self.push(Line::styled(switched.summary, dim()));
            }
            Err(err) => self.push(Line::styled(
                format!("session switch failed: {err:#}"),
                Style::default().fg(Color::Yellow),
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
                Style::default().fg(Color::Yellow),
            ));
            self.follow();
            return;
        }
        if name.chars().count() > 120 {
            self.push(Line::styled(
                "session name must be at most 120 characters",
                Style::default().fg(Color::Yellow),
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
                    Style::default().fg(Color::Yellow),
                ));
                self.follow();
                return;
            };
            let result = renamer(session_id, name);
            self.session_renamer = Some(renamer);
            if let Err(err) = result {
                self.push(Line::styled(
                    format!("session rename failed: {err:#}"),
                    Style::default().fg(Color::Yellow),
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
                            Style::default().fg(Color::Green),
                        ));
                    }
                    self.push(Line::styled(
                        format!("session sync update failed with HTTP {}", response.status()),
                        Style::default().fg(Color::Yellow),
                    ));
                    self.follow();
                    return;
                }
                Err(err) if cached => self.push(Line::styled(
                    format!("session renamed; sync update failed: {err}"),
                    Style::default().fg(Color::Yellow),
                )),
                Err(err) => {
                    self.push(Line::styled(
                        format!("session rename failed: {err}"),
                        Style::default().fg(Color::Yellow),
                    ));
                    self.follow();
                    return;
                }
            }
        }
        if !cached && !synced {
            self.push(Line::styled(
                format!("session '{session_id}' was not found"),
                Style::default().fg(Color::Yellow),
            ));
        } else {
            self.push(Line::styled(
                format!("✓ renamed session {session_id} → {name}"),
                Style::default().fg(Color::Green),
            ));
        }
        self.follow();
    }

    /// `/attach <session-id>` — attach to a running session as a viewer.
    ///
    /// In the TUI, this prints a notice that attach mode runs in a separate
    /// `hi --attach` process (the TUI can't both run a local agent and
    /// attach to a remote one simultaneously). The user should exit the TUI
    /// and run `hi --attach <id>` from the terminal.
    pub(crate) async fn handle_attach_command(&mut self, arg: &str) {
        let session_id = arg.trim();
        if session_id.is_empty() {
            self.push(Line::styled(
                "usage: /attach <session-id> — or run `hi --attach <id>` from the terminal",
                dim(),
            ));
            self.follow();
            return;
        }
        self.push(Line::styled(
            format!(
                "→ to attach to session {session_id}, exit the TUI and run:\n  \
                 hi --attach {session_id}"
            ),
            Style::default().fg(Color::Cyan),
        ));
        self.follow();
    }

    /// `/daemon` — start this session as a persistent daemon.
    ///
    /// In the TUI, this prints a notice that daemon mode runs in a separate
    /// `hi --daemon` process. The user should exit the TUI and run
    /// `hi --daemon --sync` from the terminal.
    pub(crate) async fn handle_daemon_command(&mut self, _arg: &str) {
        self.push(Line::styled(
            "→ to start a daemon for this session, exit the TUI and run:\n  \
             hi --daemon --sync",
            Style::default().fg(Color::Cyan),
        ));
        self.follow();
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
                    "  {marker} {}{}  · /sessions switch {}",
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
        for session in synced.iter().filter(|session| !seen.contains(&session.id)) {
            let marker = if self.sync_session_id.as_deref() == Some(session.id.as_str()) {
                "●"
            } else {
                "○"
            };
            self.push(Line::styled(
                format!(
                    "  {marker} {}{}  · /sessions switch {}",
                    session.id,
                    if session.title.is_empty() {
                        String::new()
                    } else {
                        format!(": {}", session.title)
                    },
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
        self.session_completion_cache = completion;

        if let Err(err) = synced_result {
            self.push(Line::styled(
                format!("session sync unavailable: {err}"),
                Style::default().fg(Color::Yellow),
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
        let response = client
            .get(&url)
            .header("x-api-key", &config.api_key)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("fetch failed: {e}"))?;

        if !response.status().is_success() {
            anyhow::bail!("HTTP {}", response.status());
        }

        let body: serde_json::Value = response.json().await?;
        Ok(body["sessions"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|session| {
                Some(SyncedSessionInfo {
                    id: session["session_id"].as_str()?.to_string(),
                    title: session["title"].as_str().unwrap_or("").to_string(),
                    status: session["status"].as_str().unwrap_or("saved").to_string(),
                    records: session["record_count"].as_u64().unwrap_or(0),
                })
            })
            .collect())
    }
}
