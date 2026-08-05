//! `App` methods: sync-related slash commands (`/sync`, `/sessions`, `/attach`,
//! `/daemon`).

use ratatui::style::Style;
use ratatui::text::Line;

use crate::model_picker::ModelPicker;
use crate::render::dim;

/// Active "steer remote host" bridge — typed prompts go to ipop, not local agent.
#[derive(Clone, Debug)]
pub struct SteeringRemote {
    pub session_id: String,
    pub base_url: String,
    pub api_key: String,
    pub http: reqwest::Client,
}

#[derive(Clone, Debug)]
struct SyncedSessionInfo {
    id: String,
    title: String,
    status: String,
    records: u64,
    project: String,
    favorite: bool,
    archived: bool,
    /// Host advertises remote input *and* still looks alive (API field).
    host_alive: bool,
    machine_id: String,
}

/// How Enter / `/sessions attach` should join a listed session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SessionJoinKind {
    /// Steer a live host over the ipop API (tmux-like).
    SteerHost,
    /// Continue the conversation with a local agent (portable).
    ContinueHere,
}

impl SyncedSessionInfo {
    fn join_kind(&self) -> SessionJoinKind {
        if self.host_alive {
            SessionJoinKind::SteerHost
        } else {
            SessionJoinKind::ContinueHere
        }
    }

    fn mode_label(&self) -> &'static str {
        match self.join_kind() {
            SessionJoinKind::SteerHost => "hosted · Enter steers host",
            SessionJoinKind::ContinueHere => "portable · Enter continues here",
        }
    }
}

/// Startup tap (local runtime publishing, startup RemoteUi slot) plus a
/// TUI-local streamer. Always built from `base_event_tap`, never from the
/// current tap, so repeated `/sync`/switch commands can't grow a chain of
/// orphaned RemoteUis.
fn compose_tap(
    base: Option<crate::RemoteEventTap>,
    rui: std::sync::Arc<crate::sync_tui::RemoteUi>,
) -> crate::RemoteEventTap {
    std::sync::Arc::new(move |event: &crate::event::UiEvent| {
        if let Some(base) = &base {
            base(event);
        }
        rui.push_event(event.clone());
    })
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
                    // `sync_remote_ui.is_none()` guards re-entry (e.g. `/sync
                    // paused` then `/sync on`) so a second RemoteUi is never
                    // composed on top of the first, duplicating every event.
                    if self.remote_flush_callback.is_none() && self.sync_remote_ui.is_none() {
                        let config = self.sync_config.clone().unwrap();
                        let rui = std::sync::Arc::new(crate::sync_tui::RemoteUi::new(
                            crate::sync_tui::SyncConfig {
                                base_url: config.base_url,
                                api_key: config.api_key,
                            },
                            session_id,
                        ));
                        // Compose onto the STARTUP tap, never the current one:
                        // the startup tap publishes to the local runtime for
                        // attach viewers and must keep running, while composing
                        // onto the current tap would grow the chain by one
                        // orphaned RemoteUi per on/off cycle.
                        self.remote_event_tap =
                            Some(compose_tap(self.base_event_tap.clone(), rui.clone()));
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
                    // Compose onto the startup tap — see the `/sync on` branch.
                    self.remote_event_tap =
                        Some(compose_tap(self.base_event_tap.clone(), remote.clone()));
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
                // Restore the startup tap: without this, the dropped RemoteUi
                // stays reachable from the composed tap and keeps serializing
                // and buffering every event despite sync being "off".
                self.remote_event_tap = self.base_event_tap.clone();
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
                // switch always continues here (portable), never steers.
                self.steering_remote_session = None;
                self.switch_session(agent, session_id).await;
            }
            value if value == "continue" || value.starts_with("continue ") => {
                let session_id = value.strip_prefix("continue").unwrap_or("").trim();
                self.steering_remote_session = None;
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
                // Switching sessions drops host-mode / steer-bridge for the
                // previous id; the new session must opt in again.
                self.stop_host_mode();
                self.steering_remote_session = None;
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
                    // Compose onto the startup tap: replacing it here cut off
                    // local-runtime attach viewers and the startup RemoteUi
                    // slot (which the switcher just repointed at this session).
                    self.remote_event_tap =
                        Some(compose_tap(self.base_event_tap.clone(), remote.clone()));
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

    /// `/attach <session-id>` / `/sessions attach <id>` — smart join over the
    /// ipop API (no SSH):
    ///   • host alive + accepting input → steer that runtime (A)
    ///   • otherwise → continue conversation on this machine (B)
    /// Force portable with `/sessions switch <id>` / `/sessions continue <id>`.
    pub(crate) async fn handle_attach_command(&mut self, agent: &mut hi_agent::Agent, arg: &str) {
        let mut parts = arg.split_whitespace();
        let session_id = parts.next().unwrap_or("").trim();
        let force = parts.next().unwrap_or("");
        if session_id.is_empty() {
            self.push(Line::styled(
                "usage: /attach <session-id> [continue|steer]",
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
        if !valid_session_id(session_id) {
            self.push(Line::styled(
                "invalid session id",
                Style::default().fg(crate::theme::theme().warning),
            ));
            self.follow();
            return;
        }

        // Optional override: `continue` forces portable; `steer` forces host.
        let forced = match force {
            "continue" | "here" | "local" | "portable" => Some(SessionJoinKind::ContinueHere),
            "steer" | "host" | "remote" => Some(SessionJoinKind::SteerHost),
            "" => None,
            other => {
                self.push(Line::styled(
                    format!("unknown attach mode '{other}' (use continue|steer)"),
                    Style::default().fg(crate::theme::theme().warning),
                ));
                self.follow();
                return;
            }
        };

        let detail = match self.fetch_session_detail(session_id).await {
            Ok(value) => value,
            Err(err) => {
                self.push(Line::styled(
                    format!("could not read session metadata: {err:#}"),
                    Style::default().fg(crate::theme::theme().warning),
                ));
                self.follow();
                return;
            }
        };
        let auto_kind = if detail
            .get("host_alive")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            SessionJoinKind::SteerHost
        } else {
            SessionJoinKind::ContinueHere
        };
        let kind = forced.unwrap_or(auto_kind);
        let host = detail
            .get("machine_id")
            .and_then(|v| v.as_str())
            .unwrap_or("remote-host");

        match kind {
            SessionJoinKind::SteerHost => {
                self.push(Line::styled(
                    format!(
                        "⟳ hosted session — steering {host} over API (no SSH). \
                         Type to send prompts; `/sessions attach {session_id} continue` to take over here."
                    ),
                    Style::default().fg(crate::theme::theme().accent_system),
                ));
                self.follow();
                // Viewer/steerer path stays in-process: open a background
                // prompt bridge into the local queue while showing live status.
                // Full SSE transcript lives in CLI attach; here we wire input
                // and keep the user in the TUI with a clear hosted banner.
                self.start_steer_bridge(session_id).await;
            }
            SessionJoinKind::ContinueHere => {
                self.push(Line::styled(
                    format!(
                        "⟳ portable session — continuing on this machine{}",
                        if forced.is_some() {
                            ""
                        } else {
                            " (host offline or not accepting input)"
                        }
                    ),
                    dim(),
                ));
                self.follow();
                self.switch_session(agent, session_id).await;
            }
        }
    }

    /// Fetch one session's metadata JSON.
    async fn fetch_session_detail(&self, session_id: &str) -> anyhow::Result<serde_json::Value> {
        let Some(config) = &self.sync_config else {
            anyhow::bail!("sync not configured");
        };
        let Some(client) = &self.sync_http else {
            anyhow::bail!("sync HTTP client unavailable");
        };
        let url = format!("{}/hi/sessions/{session_id}", config.base_url);
        let response = client
            .get(&url)
            .header("x-api-key", &config.api_key)
            .send()
            .await?;
        if !response.status().is_success() {
            anyhow::bail!("HTTP {}", response.status());
        }
        Ok(response.json().await?)
    }

    /// Steer a remote host: POST typed lines to its input queue. Local agent
    /// is left alone so we don't steal the writer lease from the host.
    async fn start_steer_bridge(&mut self, session_id: &str) {
        let Some(config) = self.sync_config.clone() else {
            return;
        };
        let Some(client) = self.sync_http.clone() else {
            self.push(Line::styled(
                "sync HTTP client unavailable",
                Style::default().fg(crate::theme::theme().warning),
            ));
            self.follow();
            return;
        };
        // Mark UI state so the user sees we're in hosted-steer mode.
        self.sync_session_id = Some(session_id.to_string());
        self.stop_host_mode();
        // Install a lightweight "forward next queue lines as remote prompts"
        // flag via a dedicated remote-input style channel that the idle loop
        // already drains — but populate it from *local* queue by swapping in
        // a steerer callback. Simpler: set a steerer mode that redirects
        // submitted lines in the run loop.
        self.steering_remote_session = Some(SteeringRemote {
            session_id: session_id.to_string(),
            base_url: config.base_url,
            api_key: config.api_key,
            http: client,
        });
        self.push(Line::styled(
            format!("✓ steering {session_id} — lines you type are sent to the host over the API"),
            Style::default().fg(crate::theme::theme().accent_success),
        ));
        self.push(Line::styled(
            "  /sessions attach <id> continue  · take over on this machine",
            dim(),
        ));
        self.push(Line::styled(
            "  /sessions host off              · stop if you were hosting",
            dim(),
        ));
        self.follow();
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
        self.apply_host_toggle_result(result, true);
    }

    /// Apply a host-mode toggle outcome to UI state — shared by the awaited
    /// `/sessions host` command and the startup background enable.
    ///
    /// Sync is best-effort decoration on a local-first app: a portal failure
    /// must never surface as an error or affect the coding workflow. An
    /// explicit `/sessions host` command gets a calm availability note
    /// (`announce_failure`); the automatic startup enable stays fully silent
    /// and the session simply runs unhosted.
    pub(crate) fn apply_host_toggle_result(
        &mut self,
        result: anyhow::Result<Option<crate::SessionHostEnable>>,
        announce_failure: bool,
    ) {
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
                    self.follow();
                } else {
                    self.push(Line::styled(
                        "host off — no longer accepting remote prompts",
                        dim(),
                    ));
                    self.follow();
                }
            }
            Err(_) if announce_failure => {
                self.push(Line::styled(
                    "host mode unavailable right now — the sync portal is unreachable; this session keeps working locally",
                    dim(),
                ));
                self.follow();
            }
            Err(_) => {}
        }
    }

    /// `/team [role] [model|local|off] [base-url] [api-key]` — the team-role
    /// table: the driver plans on the big model while explore/delegate
    /// executors can run elsewhere (typically a local model, so execution
    /// rounds cost nothing). Route changes apply to children started after
    /// the command.
    pub(crate) fn handle_team_command(&mut self, agent: &mut hi_agent::Agent, arg: &str) {
        let parts: Vec<&str> = arg.split_whitespace().collect();
        if parts.is_empty() {
            for row in agent.team_roles() {
                let suffix = if row.inherited { "  (driver)" } else { "" };
                self.push(Line::styled(
                    format!("  {:<9} {}  @ {}{}", row.role, row.model, row.route, suffix),
                    dim(),
                ));
            }
            if let Some(pending) = &self.pending_team_provision {
                let phase = pending.phase_rx.borrow().clone();
                self.push(Line::styled(
                    format!(
                        "  {:<9} setting up {} — {}",
                        pending.role,
                        pending.display,
                        provision_phase_line(&pending.display, &phase)
                            .trim_start_matches("⟳ ")
                            .trim_start_matches(&format!("{}: ", pending.display))
                    ),
                    dim(),
                ));
            }
            // The table above is context; the menu below is the interface.
            self.open_team_role_menu(agent);
            return;
        }
        if parts[0] == "auto" {
            if let Some(pending) = &self.pending_team_provision {
                self.push(Line::styled(
                    format!(
                        "already setting up {} for {} — wait for it to finish before starting another auto-setup",
                        pending.display, pending.role
                    ),
                    dim(),
                ));
                self.follow();
                return;
            }
            self.run_team_auto_setup(agent);
            self.follow();
            return;
        }
        let role = parts[0];
        let value = parts.get(1).copied();
        match (role, value) {
            ("driver", _) => {
                self.push(Line::styled(
                    "the driver is the session model — switch it with /model or /provider",
                    dim(),
                ));
            }
            ("skeptic", _) => {
                self.push(Line::styled(
                    "skeptic routing has dedicated commands: /config skeptic-local on|off (auto-managed) or HI_SKEPTIC_ENDPOINT",
                    dim(),
                ));
            }
            ("planner", Some("off")) => {
                agent.set_planner_model(None);
                self.push(Line::styled("planner → driver model", dim()));
            }
            ("planner", Some(model)) => {
                agent.set_planner_model(Some(model.to_string()));
                self.push(Line::styled(format!("planner → {model}"), dim()));
            }
            ("explore" | "delegate" | "editor", None) => {
                self.open_team_model_picker(role);
            }
            ("explore" | "delegate" | "editor", Some("off")) => {
                self.cancel_team_setup_for_role(role);
                agent.set_team_route(role, None, None, None);
                self.push(Line::styled(
                    format!("{role} → driver route (applies to new {role} runs)"),
                    dim(),
                ));
            }
            ("explore" | "delegate" | "editor", Some(model)) => {
                // Power users may still pass an explicit endpoint; everyone
                // else picks a name and hi does the rest.
                self.cancel_team_setup_for_role(role);
                let explicit_endpoint = parts
                    .get(2)
                    .filter(|value| value.starts_with("http"))
                    .map(|s| s.to_string());
                if let Some(endpoint) = explicit_endpoint {
                    let key = parts.get(3).map(|s| s.to_string());
                    agent.set_team_route(
                        role,
                        Some(model.to_string()),
                        Some(endpoint.clone()),
                        key,
                    );
                    self.push(Line::styled(
                        format!("{role} → {model} @ {endpoint} (applies to new {role} runs)"),
                        dim(),
                    ));
                } else if let Some(resolved) = hi_agent::local_skeptic::resolve_team_local_model(
                    model,
                    hi_agent::local_skeptic::system_ram_gb(),
                    hi_agent::local_skeptic::detect_backend_cached(),
                ) {
                    self.assign_supported_local_model(agent, role, resolved);
                } else {
                    // Not a supported local name → a model id on the driver's
                    // provider (e.g. a cheaper cloud model for recon).
                    agent.set_team_route(role, Some(model.to_string()), None, None);
                    self.push(Line::styled(
                        format!("{role} → {model} (driver route; applies to new {role} runs)"),
                        dim(),
                    ));
                }
            }
            (other, _) => {
                self.push(Line::styled(
                    format!("unknown role '{other}' — roles: driver, explore, delegate, editor, skeptic, planner"),
                    dim(),
                ));
            }
        }
        self.follow();
    }

    /// Mark an in-flight setup stale before applying a manual route change.
    /// Provisioning is not aborted: it may already have spawned a server, and
    /// dropping the task at that point would leak that process. The poller
    /// disposes of a successful stale server and never applies its route.
    pub(crate) fn cancel_team_setup_for_role(&mut self, role: &str) {
        let pending_matches = self
            .pending_team_provision
            .as_ref()
            .is_some_and(|pending| pending.role == role && !pending.cancelled);
        let queued_role = self
            .queued_team_assignments
            .iter()
            .any(|(queued_role, _)| queued_role == role);
        let auto_chain_active = !self.queued_team_assignments.is_empty() || self.auto_setup_skeptic;

        // A role can be waiting behind a different role's download. In that
        // case there is no matching pending task to mark stale, but leaving
        // the queue intact would let the later auto assignment overwrite the
        // user's explicit route. Any manual edit to an active auto chain must
        // therefore clear the queue; only a task currently provisioning this
        // exact role needs to be marked stale and stopped after it completes.
        if pending_matches || queued_role || auto_chain_active {
            if pending_matches {
                if let Some(pending) = &mut self.pending_team_provision {
                    pending.cancelled = true;
                }
            }
            self.queued_team_assignments.clear();
            self.auto_setup_skeptic = false;
            self.push(Line::styled(
                if pending_matches {
                    format!("cancelling local setup for {role} — the new route will be kept")
                } else {
                    format!("cancelling team auto-setup — the {role} route will be kept")
                },
                dim(),
            ));
        }
    }

    /// Wire a supported local model to a role: reuse a running managed server
    /// when one already serves it, otherwise provision (download + spawn) on
    /// a background task and wire the role when it completes.
    pub(crate) fn assign_supported_local_model(
        &mut self,
        agent: &mut hi_agent::Agent,
        role: &str,
        resolved: hi_agent::local_skeptic::ResolvedLocalModel,
    ) {
        let reuse = resolved
            .mlx
            .and_then(|quant| agent.running_local_model_server(quant.model_id))
            .or_else(|| {
                resolved
                    .entry
                    .cuda
                    .and_then(|cuda| agent.running_local_model_server(cuda.model_id))
            });
        if let Some((endpoint, model_id)) = reuse {
            agent.set_team_route(role, Some(model_id.clone()), Some(endpoint.clone()), None);
            self.push(Line::styled(
                format!("{role} → {model_id} @ local (reusing the running server; applies to new {role} runs)"),
                dim(),
            ));
            return;
        }
        if let Some(pending) = &self.pending_team_provision
            && !pending.cancelled
        {
            self.push(Line::styled(
                format!(
                    "already setting up {} for {} — one local setup at a time; retry when it finishes",
                    pending.display, pending.role
                ),
                dim(),
            ));
            return;
        }
        if self
            .pending_team_provision
            .as_ref()
            .is_some_and(|pending| pending.cancelled)
        {
            // The previous task cannot be aborted safely after it may have
            // spawned a server, so retain the replacement request until the
            // poller reaps and stops that stale task.
            self.queued_team_assignments = vec![(role.to_string(), resolved)];
            self.auto_setup_skeptic = false;
            self.push(Line::styled(
                format!(
                    "previous local setup is finishing — {role} will switch to the new local model when it is stopped"
                ),
                dim(),
            ));
            return;
        }
        let Some(backend) = hi_agent::local_skeptic::detect_backend_cached() else {
            self.push(Line::styled(
                "no local-inference backend on this machine (needs Apple Silicon or an NVIDIA runtime); the role stays on the driver",
                dim(),
            ));
            return;
        };
        if backend != hi_agent::local_skeptic::LocalBackend::Mlx {
            self.push(Line::styled(
                "the selected provider action is MLX-only and needs Apple Silicon",
                dim(),
            ));
            return;
        }
        let spec = match hi_agent::local_skeptic::team_model_spec(resolved, backend) {
            Ok(spec) => spec,
            Err(error) => {
                self.push(Line::styled(format!("{error:#}"), dim()));
                self.follow();
                return;
            }
        };
        let display = resolved.display();
        let model_dir = hi_tools::skeptic_model_dir(&spec.repo);
        let (phase_tx, phase_rx) =
            tokio::sync::watch::channel(hi_agent::local_skeptic::ProvisionPhase::Resolving);
        let task = tokio::spawn(async move {
            hi_agent::local_skeptic::provision_team_local_model(resolved, phase_tx).await
        });
        self.pending_team_provision = Some(crate::PendingTeamProvision {
            role: role.to_string(),
            display: display.clone(),
            cancelled: false,
            task,
            phase_rx,
            announced_phase: hi_agent::local_skeptic::ProvisionPhase::Resolving,
            phase_started: std::time::Instant::now(),
            model_dir,
            ticks_since_report: 0,
            last_reported_bytes: 0,
            progress_entry_index: None,
        });
        self.push(Line::styled(
            format!(
                "⟳ setting up {display} locally for {role} — the download and server start run in the background; the role wires itself when ready"
            ),
            dim(),
        ));
    }

    /// Non-blocking: apply a finished `/team` local-model provisioning, if
    /// any, and surface quiet download progress as an occasional dim line
    /// (roughly every 30s of ticker time) while one is running.
    pub(crate) async fn poll_pending_team_provision(&mut self, agent: &mut hi_agent::Agent) {
        let finished = self
            .pending_team_provision
            .as_ref()
            .is_some_and(|pending| pending.task.is_finished());
        if !finished {
            let mut transition = None;
            let mut heartbeat = None;
            let mut index = None;
            if let Some(pending) = &mut self.pending_team_provision {
                // Phase transition: announce immediately with a permanent line
                // and start a fresh in-place progress line for the new phase.
                let current = pending.phase_rx.borrow().clone();
                if current != pending.announced_phase {
                    pending.announced_phase = current.clone();
                    pending.phase_started = std::time::Instant::now();
                    pending.ticks_since_report = 0;
                    pending.progress_entry_index = None;
                    transition = Some(provision_phase_line(&pending.display, &current));
                }
                // Heartbeat within the phase: a single transcript line that
                // updates in place (bar/percent/elapsed), not a spam stream.
                pending.ticks_since_report = pending.ticks_since_report.saturating_add(1);
                let cadence = provision_heartbeat_ticks(&pending.announced_phase);
                if pending.ticks_since_report >= cadence {
                    pending.ticks_since_report = 0;
                    let bytes = dir_size_shallow(&pending.model_dir);
                    if let Some(line) = provision_heartbeat_line(
                        &pending.display,
                        &pending.announced_phase,
                        pending.phase_started.elapsed(),
                        bytes,
                        pending.last_reported_bytes,
                    ) {
                        pending.last_reported_bytes = bytes;
                        heartbeat = Some(line);
                        index = Some(pending.progress_entry_index);
                    }
                }
            }
            let mut redraw = false;
            if let Some(line) = transition {
                self.push(Line::styled(line, dim()));
                redraw = true;
            }
            if let Some(line) = heartbeat {
                let mut slot = index.flatten();
                self.push_or_replace_progress(&mut slot, "⟳", Line::styled(line, dim()));
                if let Some(pending) = &mut self.pending_team_provision {
                    pending.progress_entry_index = slot;
                }
                redraw = true;
            }
            if redraw {
                self.follow();
            }
            return;
        }
        let Some(pending) = self.pending_team_provision.take() else {
            return;
        };
        let result = match pending.task.await {
            Ok(result) => result,
            Err(join_error) => Err(anyhow::anyhow!("local setup task failed: {join_error}")),
        };
        if pending.cancelled {
            if let Ok((_, _, process_id)) = result {
                if !process_id.is_empty() {
                    hi_tools::stop_local_server(&process_id);
                }
            }
            self.push(Line::styled(
                format!(
                    "local setup for {} cancelled; the newer route is unchanged",
                    pending.role
                ),
                dim(),
            ));
            // A manual local choice made while the stale task was winding
            // down is queued by `assign_supported_local_model`. Start it now
            // that the single in-flight provisioning slot is free; otherwise
            // the user's replacement choice would be silently lost.
            self.drain_team_assignment_queue(agent);
            self.follow();
            return;
        }
        let failed = result.is_err();
        self.apply_team_provision_result(agent, &pending.role, &pending.display, result);
        if failed {
            // Auto-setup must not cascade a broken model across more roles.
            self.queued_team_assignments.clear();
            self.auto_setup_skeptic = false;
            return;
        }
        // Auto-setup: wire the next queued role — usually an instant reuse of
        // the server that just came up.
        self.drain_team_assignment_queue(agent);
    }

    /// Apply a provisioning outcome: success wires the role and registers the
    /// server for reuse/teardown; failure leaves the role on the driver with
    /// a calm note (never a raw error dump, never a broken workflow).
    pub(crate) fn apply_team_provision_result(
        &mut self,
        agent: &mut hi_agent::Agent,
        role: &str,
        display: &str,
        result: anyhow::Result<(String, String, String)>,
    ) {
        match result {
            Ok((endpoint, model_id, process_id)) => {
                agent.register_team_local_server(endpoint.clone(), model_id.clone(), process_id);
                agent.set_team_route(role, Some(model_id.clone()), Some(endpoint), None);
                self.push(Line::styled(
                    format!("✓ {role} → {model_id} @ local (ready — applies to new {role} runs)"),
                    Style::default().fg(crate::theme::theme().accent_success),
                ));
            }
            Err(error) => {
                let reason: String = format!("{error:#}").chars().take(140).collect();
                self.push(Line::styled(
                    format!(
                        "couldn't set up {display} locally ({reason}); {role} stays on the driver"
                    ),
                    dim(),
                ));
            }
        }
        self.follow();
    }

    /// Kick off hosted-mode enablement without blocking the UI. The
    /// controller's network work (portal registration) runs on a background
    /// task; [`Self::poll_pending_host_enable`] applies the outcome from the
    /// event loop. Used at startup, where awaiting an unreachable portal
    /// delayed first paint by tens of seconds.
    pub(crate) fn start_host_enable_in_background(&mut self) {
        if self.hosting_remote_input || self.pending_host_enable.is_some() {
            return;
        }
        let Some(controller) = self.session_host.take() else {
            return;
        };
        let enable_future = controller(true);
        self.session_host = Some(controller);
        self.pending_host_enable = Some(tokio::spawn(enable_future));
    }

    /// Non-blocking: if the background host-enable finished, apply it.
    pub(crate) async fn poll_pending_host_enable(&mut self) {
        let finished = self
            .pending_host_enable
            .as_ref()
            .is_some_and(|task| task.is_finished());
        if !finished {
            return;
        }
        let Some(task) = self.pending_host_enable.take() else {
            return;
        };
        let result = match task.await {
            Ok(result) => result,
            Err(join_error) => Err(anyhow::anyhow!("host enable task failed: {join_error}")),
        };
        // Automatic startup enable: silent on failure by design.
        self.apply_host_toggle_result(result, false);
    }

    fn stop_host_mode(&mut self) {
        if let Some(abort) = self.remote_input_poller.take() {
            abort.abort();
        }
        self.remote_input_rx = None;
        self.hosting_remote_input = false;
    }

    /// If we're in hosted-steer mode, POST `prompt` to the remote host's input
    /// queue over the ipop API. Returns true when the line was handled (caller
    /// must not run it as a local agent turn).
    pub(crate) async fn maybe_forward_steered_prompt(&mut self, prompt: &str) -> bool {
        let Some(steering) = self.steering_remote_session.clone() else {
            return false;
        };
        let trimmed = prompt.trim();
        if trimmed.is_empty() {
            return true;
        }
        // Escape hatch while steering.
        if trimmed == "/sessions detach"
            || trimmed == "/detach"
            || trimmed.starts_with("/sessions attach ")
            || trimmed.starts_with("/attach ")
            || trimmed.starts_with("/sessions switch ")
            || trimmed.starts_with("/sessions continue ")
            || trimmed.starts_with("/sessions host")
        {
            if trimmed == "/sessions detach" || trimmed == "/detach" {
                self.steering_remote_session = None;
                self.push(Line::styled("detached from remote host", dim()));
                self.follow();
                return true;
            }
            return false;
        }
        let url = format!(
            "{}/hi/sessions/{}/input",
            steering.base_url, steering.session_id
        );
        let result = steering
            .http
            .post(&url)
            .header("x-api-key", &steering.api_key)
            .json(&serde_json::json!({ "prompt": trimmed }))
            .send()
            .await;
        match result {
            Ok(response) if response.status().is_success() => {
                self.push(Line::styled(
                    format!("→ sent to host {}", steering.session_id),
                    dim(),
                ));
            }
            Ok(response) => {
                self.push(Line::styled(
                    format!(
                        "→ host rejected prompt (HTTP {}) — try `/sessions attach {} continue`",
                        response.status(),
                        steering.session_id
                    ),
                    Style::default().fg(crate::theme::theme().warning),
                ));
            }
            Err(err) => {
                self.push(Line::styled(
                    format!("→ failed to reach host: {err:#}"),
                    Style::default().fg(crate::theme::theme().warning),
                ));
            }
        }
        self.follow();
        true
    }

    /// Drain any remote attach prompts into the local turn queue. Returns true
    /// when at least one prompt was enqueued (caller should leave the idle
    /// input wait and run the queue).
    pub(crate) fn drain_remote_input(&mut self) -> bool {
        if self.remote_input_rx.is_none() {
            return false;
        }
        // Collect first so enqueue can borrow `self` without overlapping the
        // channel receiver borrow.
        let mut incoming = Vec::new();
        let mut disconnected = false;
        if let Some(rx) = self.remote_input_rx.as_mut() {
            loop {
                match rx.try_recv() {
                    Ok(prompt) => {
                        let prompt = prompt.trim().to_string();
                        if !prompt.is_empty() {
                            incoming.push(prompt);
                        }
                    }
                    Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                    Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }
        }
        if disconnected {
            self.stop_host_mode();
        }
        let mut queued = 0usize;
        let mut rejected = 0usize;
        for prompt in incoming {
            if self.try_enqueue_prompt(prompt) {
                queued += 1;
            } else {
                rejected += 1;
            }
        }
        if rejected > 0 {
            self.push(Line::styled(
                format!(
                    "← dropped {rejected} remote prompt{} — queue full ({}/{})",
                    if rejected == 1 { "" } else { "s" },
                    self.queue.len(),
                    crate::MAX_PROMPT_QUEUE
                ),
                Style::default().fg(crate::theme::theme().warning),
            ));
            self.follow();
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
            // Rejected-only drains must not kick the session loop into a
            // no-op queue run — status was already written above.
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
            let mode = synced_match
                .map(SyncedSessionInfo::mode_label)
                .unwrap_or("local");
            self.push(Line::styled(
                format!(
                    "  {marker} {}{}  · {mode}",
                    session.id,
                    if title.is_empty() {
                        String::new()
                    } else {
                        format!(": {title}")
                    },
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
            let host = if session.machine_id.is_empty() {
                String::new()
            } else {
                format!(
                    " @{}",
                    session.machine_id.chars().take(12).collect::<String>()
                )
            };
            self.push(Line::styled(
                format!(
                    "  {marker} {}{}{}{}{host}  · {}",
                    session.id,
                    if session.title.is_empty() {
                        String::new()
                    } else {
                        format!(": {}", session.title)
                    },
                    if session.favorite { " ★" } else { "" },
                    if session.archived { " [archived]" } else { "" },
                    session.mode_label(),
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
                            host_alive: session["host_alive"].as_bool().unwrap_or(false),
                            machine_id: session["machine_id"].as_str().unwrap_or("").to_string(),
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

/// Total size of the files directly inside `dir` (model repos download flat).
/// Best-effort: unreadable entries count as zero.
pub(crate) fn dir_size_shallow(dir: &std::path::Path) -> u64 {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .filter_map(|entry| entry.metadata().ok())
                .filter(|meta| meta.is_file())
                .map(|meta| meta.len())
                .sum()
        })
        .unwrap_or(0)
}

/// The transcript line announcing a provisioning phase transition.
pub(crate) fn provision_phase_line(
    display: &str,
    phase: &hi_agent::local_skeptic::ProvisionPhase,
) -> String {
    use hi_agent::local_skeptic::ProvisionPhase;
    match phase {
        ProvisionPhase::Resolving => format!("⟳ {display}: checking hardware and cached weights…"),
        ProvisionPhase::Downloading => {
            format!("⟳ {display}: downloading weights (quiet, in the background)…")
        }
        ProvisionPhase::BuildingServer => {
            format!("⟳ {display}: compiling the hi-local serving binary (first run)…")
        }
        ProvisionPhase::LoadingModel { deadline_secs, .. } => format!(
            "⟳ {display}: server started — loading weights into memory (can take up to {})…",
            format_secs(*deadline_secs)
        ),
    }
}

pub(crate) fn local_runtime_phase_line(
    display: &str,
    phase: &hi_agent::local_skeptic::LocalRuntimePhase,
) -> String {
    use hi_agent::local_skeptic::LocalRuntimePhase;
    match phase {
        LocalRuntimePhase::Resolving => {
            format!("⟳ {display}: checking hardware and cached weights…")
        }
        LocalRuntimePhase::Downloading => {
            format!("⟳ {display}: downloading/resuming model weights…")
        }
        LocalRuntimePhase::PreparingRuntime => {
            format!("⟳ {display}: preparing the bundled MLX runtime…")
        }
        LocalRuntimePhase::StartingServer => format!("⟳ {display}: starting hi-local…"),
        LocalRuntimePhase::LoadingModel { deadline_secs, .. } => {
            format!(
                "⟳ {display}: loading weights into unified memory (up to {})…",
                format_secs(*deadline_secs)
            )
        }
        LocalRuntimePhase::Verifying => {
            format!("⟳ {display}: verifying /v1/models and chat compatibility…")
        }
        LocalRuntimePhase::Ready => format!("✓ {display}: local runtime ready"),
    }
}

/// Heartbeat cadence for the provider-picker runtime. The loading phase gets
/// the fastest refresh because it is the phase most likely to look stalled;
/// compilation and download use a gentler cadence to avoid needless redraws.
pub(crate) fn local_runtime_heartbeat_ticks(
    phase: &hi_agent::local_skeptic::LocalRuntimePhase,
) -> u32 {
    use hi_agent::local_skeptic::LocalRuntimePhase;
    match phase {
        LocalRuntimePhase::Downloading => 16,
        LocalRuntimePhase::LoadingModel { .. } => 8,
        LocalRuntimePhase::PreparingRuntime | LocalRuntimePhase::StartingServer => 20,
        LocalRuntimePhase::Resolving | LocalRuntimePhase::Verifying | LocalRuntimePhase::Ready => {
            40
        }
    }
}

/// Produce one in-place progress line for the provider-picker runtime. Model
/// loading uses resident memory as an honest approximation of work completed;
/// the other slow phases show elapsed time instead of a fake percentage.
pub(crate) fn local_runtime_heartbeat_line(
    display: &str,
    phase: &hi_agent::local_skeptic::LocalRuntimePhase,
    in_phase: std::time::Duration,
    bytes_on_disk: u64,
    last_reported_bytes: u64,
) -> Option<String> {
    use hi_agent::local_skeptic::LocalRuntimePhase;
    match phase {
        LocalRuntimePhase::Downloading => {
            if bytes_on_disk > last_reported_bytes {
                Some(format!(
                    "⟳ {display}: downloading — {:.1} GiB on disk…",
                    bytes_on_disk as f64 / (1024.0 * 1024.0 * 1024.0)
                ))
            } else {
                Some(format!(
                    "⟳ {display}: still downloading ({} in)…",
                    format_secs(in_phase.as_secs())
                ))
            }
        }
        LocalRuntimePhase::PreparingRuntime => Some(format!(
            "⟳ {display}: preparing MLX runtime ({} elapsed)…",
            format_secs(in_phase.as_secs())
        )),
        LocalRuntimePhase::StartingServer => Some(format!(
            "⟳ {display}: starting hi-local ({} elapsed)…",
            format_secs(in_phase.as_secs())
        )),
        LocalRuntimePhase::LoadingModel {
            deadline_secs,
            server_handle,
            expected_bytes,
        } => {
            let rss = hi_tools::local_server_os_pid(server_handle).and_then(rss_bytes);
            Some(loading_bar_line(
                display,
                rss,
                *expected_bytes,
                in_phase,
                *deadline_secs,
            ))
        }
        LocalRuntimePhase::Resolving | LocalRuntimePhase::Verifying | LocalRuntimePhase::Ready => {
            None
        }
    }
}

/// Heartbeat cadence per phase, in ~120ms ticker calls. Heartbeats update a
/// single transcript line IN PLACE, so the slow phases can refresh every
/// second or two without spamming: a live bar while weights load, a growing
/// GiB counter while downloading.
pub(crate) fn provision_heartbeat_ticks(phase: &hi_agent::local_skeptic::ProvisionPhase) -> u32 {
    use hi_agent::local_skeptic::ProvisionPhase;
    match phase {
        ProvisionPhase::Downloading => 16,
        ProvisionPhase::LoadingModel { .. } => 8,
        ProvisionPhase::Resolving | ProvisionPhase::BuildingServer => 40,
    }
}

/// The within-phase heartbeat line, when the phase warrants one.
pub(crate) fn provision_heartbeat_line(
    display: &str,
    phase: &hi_agent::local_skeptic::ProvisionPhase,
    in_phase: std::time::Duration,
    bytes_on_disk: u64,
    last_reported_bytes: u64,
) -> Option<String> {
    use hi_agent::local_skeptic::ProvisionPhase;
    match phase {
        ProvisionPhase::Downloading => {
            if bytes_on_disk > last_reported_bytes {
                Some(format!(
                    "⟳ {display}: downloading — {:.1} GiB on disk…",
                    bytes_on_disk as f64 / (1024.0 * 1024.0 * 1024.0)
                ))
            } else {
                Some(format!(
                    "⟳ {display}: still downloading ({} in)…",
                    format_secs(in_phase.as_secs())
                ))
            }
        }
        ProvisionPhase::LoadingModel {
            deadline_secs,
            server_handle,
            expected_bytes,
        } => {
            let rss = hi_tools::local_server_os_pid(server_handle).and_then(rss_bytes);
            Some(loading_bar_line(
                display,
                rss,
                *expected_bytes,
                in_phase,
                *deadline_secs,
            ))
        }
        ProvisionPhase::BuildingServer => Some(format!(
            "⟳ {display}: still compiling hi-local ({} in)…",
            format_secs(in_phase.as_secs())
        )),
        ProvisionPhase::Resolving => None,
    }
}

/// `95` → `1m35s`, `40` → `40s`.
pub(crate) fn format_secs(total: u64) -> String {
    if total >= 60 {
        format!("{}m{:02}s", total / 60, total % 60)
    } else {
        format!("{total}s")
    }
}

/// Resident memory of a process in bytes (`ps -o rss=` reports KiB).
pub(crate) fn rss_bytes(pid: i32) -> Option<u64> {
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .ok()?;
    String::from_utf8(output.stdout)
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(|kib| kib * 1024)
}

/// The live weights-loading line: a bar of memory growth toward the model's
/// on-disk size. RSS slightly overshoots the weights, so the bar clamps at
/// 99% until the health check flips it to the ✓ line.
pub(crate) fn loading_bar_line(
    display: &str,
    rss_bytes: Option<u64>,
    expected_bytes: u64,
    in_phase: std::time::Duration,
    deadline_secs: u64,
) -> String {
    let gib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    match rss_bytes {
        Some(rss) if expected_bytes > 0 => {
            let frac = (rss as f64 / expected_bytes as f64).clamp(0.0, 0.99);
            format!(
                "⟳ {display}: loading weights {} {:>2.0}% · {:.1}/{:.1} GiB · {}",
                render_bar(frac, 18),
                frac * 100.0,
                gib(rss),
                gib(expected_bytes),
                format_secs(in_phase.as_secs()),
            )
        }
        _ => format!(
            "⟳ {display}: loading weights — {} elapsed (allow up to {})…",
            format_secs(in_phase.as_secs()),
            format_secs(deadline_secs)
        ),
    }
}

/// `render_bar(0.5, 10)` → `▰▰▰▰▰▱▱▱▱▱`.
pub(crate) fn render_bar(frac: f64, width: usize) -> String {
    let filled = ((frac.clamp(0.0, 1.0)) * width as f64).round() as usize;
    let mut bar = String::with_capacity(width * 3);
    for i in 0..width {
        bar.push(if i < filled { '▰' } else { '▱' });
    }
    bar
}

impl crate::App {
    /// Start a best-effort refresh of Pipe Network's live MLX catalog. The
    /// built-in catalog is already usable, so this is deliberately detached
    /// from opening the provider picker.
    pub(crate) fn start_local_catalog_refresh(&mut self) {
        if self
            .pending_local_catalog
            .as_ref()
            .is_some_and(|task| !task.is_finished())
        {
            return;
        }
        self.pending_local_catalog = Some(tokio::spawn(
            hi_agent::local_skeptic::refresh_pipenetwork_catalog(),
        ));
        self.status = "refreshing Pipe Network local model catalog…".to_string();
    }

    /// Apply a completed catalog refresh without interrupting a model picker
    /// or changing the active provider.
    pub(crate) async fn poll_pending_local_catalog(&mut self) {
        let finished = self
            .pending_local_catalog
            .as_ref()
            .is_some_and(tokio::task::JoinHandle::is_finished);
        if !finished {
            return;
        }
        let Some(task) = self.pending_local_catalog.take() else {
            return;
        };
        match task.await {
            Ok(Ok(catalog)) => {
                if let Some(picker) = self.provider_picker.as_mut() {
                    picker.replace_local_models(crate::provider_picker::local_model_rows());
                }
                self.status = format!(
                    "Pipe Network catalog refreshed — {} chat-capable MLX models discovered",
                    catalog.len()
                );
            }
            Ok(Err(error)) => {
                self.status = format!(
                    "Pipe Network catalog unavailable; using built-in local models ({error:#})"
                );
            }
            Err(error) => {
                self.status = format!(
                    "Pipe Network catalog refresh failed; using built-in local models ({error})"
                );
            }
        }
    }

    /// Bare `/team`: an interactive role menu — the same dropdown feel as
    /// `/model`. Enter on a role opens its model picker; the first row wires
    /// the whole team in one keystroke.
    pub(crate) fn open_team_role_menu(&mut self, agent: &mut hi_agent::Agent) {
        let roles = agent.team_roles();
        let describe = |name: &str| -> String {
            roles
                .iter()
                .find(|row| row.role == name)
                .map(|row| {
                    if row.inherited {
                        "driver route".to_string()
                    } else {
                        format!("{} @ {}", row.model, row.route)
                    }
                })
                .unwrap_or_else(|| "driver route".to_string())
        };
        let skeptic_state = if agent.managed_local_route().is_some() {
            "local (on)".to_string()
        } else {
            describe("skeptic")
        };
        let rows = vec![
            "auto-setup — wire delegate+editor+explore to recommended local models, skeptic included".to_string(),
            format!("delegate — {} · write-capable executor (pick a model)", describe("delegate")),
            format!("editor — {} · fast lane for mechanical edits (pick a model)", describe("editor")),
            format!("explore — {} · read-only recon (pick a model)", describe("explore")),
            format!("skeptic — {skeptic_state} · toggle free local review"),
            "planner — set with /team planner <model|off>".to_string(),
        ];
        let current = rows[0].clone();
        self.team_role_menu = true;
        self.team_picker_role = None;
        self.picker = Some(ModelPicker::new(
            rows,
            &current,
            std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
        ));
        self.push(Line::styled(
            "team roles — ↑↓ to choose, Enter to configure, Esc to close · or /team <role> <model|local|off>",
            dim(),
        ));
        self.follow();
    }

    /// One-keystroke team: delegate on the machine's best verified local
    /// model, editor/explore on the fast small one, skeptic riding the same
    /// server. Provisioning is chained through the single in-flight slot;
    /// later roles usually reuse the first server instantly.
    pub(crate) fn run_team_auto_setup(&mut self, agent: &mut hi_agent::Agent) {
        let ram = hi_agent::local_skeptic::system_ram_gb();
        let backend = hi_agent::local_skeptic::detect_backend_cached();
        let Some(delegate) =
            hi_agent::local_skeptic::resolve_team_local_model("auto", ram, backend)
        else {
            self.push(Line::styled(
                "no supported local model fits this machine; roles stay on the driver",
                dim(),
            ));
            return;
        };
        // Editor + explore: the fast small executor when it fits and differs
        // from the delegate pick; otherwise they share the delegate's server.
        let fast = hi_agent::local_skeptic::resolve_team_local_model("nemotron-4b", ram, backend)
            .filter(|fast| fast.entry.fits(ram, backend) && fast.entry.name != delegate.entry.name)
            .unwrap_or(delegate);
        self.push(Line::styled(
            format!(
                "auto-setup: delegate → {} · editor/explore → {} · skeptic → local (reuses the team server)",
                delegate.display(),
                fast.display()
            ),
            dim(),
        ));
        self.queued_team_assignments =
            vec![("editor".to_string(), fast), ("explore".to_string(), fast)];
        self.auto_setup_skeptic = true;
        self.assign_supported_local_model(agent, "delegate", delegate);
        // If delegate reused a running server (no pending provisioning), the
        // queue won't be drained by the poller — drain it now.
        self.drain_team_assignment_queue(agent);
    }

    /// Assign queued roles while no provisioning is in flight (reuse hits
    /// resolve instantly; a download/spawn re-enters via the poller).
    pub(crate) fn drain_team_assignment_queue(&mut self, agent: &mut hi_agent::Agent) {
        while self.pending_team_provision.is_none() {
            let Some((role, resolved)) = self.queued_team_assignments.first().cloned() else {
                if self.auto_setup_skeptic {
                    self.auto_setup_skeptic = false;
                    self.enable_team_skeptic_for_auto(agent);
                }
                return;
            };
            self.queued_team_assignments.remove(0);
            self.assign_supported_local_model(agent, &role, resolved);
        }
    }

    /// Point the skeptic gate at the running team server (or back at the
    /// driver). Instant when a team server is up — the reuse path never
    /// downloads or spawns anything.
    pub(crate) fn toggle_team_skeptic(&mut self, agent: &mut hi_agent::Agent) {
        if agent.managed_local_route().is_some() {
            agent.disable_local_skeptic();
            self.push(Line::styled("skeptic → driver model", dim()));
            return;
        }
        if agent.any_team_local_server().is_none() {
            self.push(Line::styled(
                "no team server running yet — set up delegate first, or use /config skeptic-local on to serve a dedicated review model",
                dim(),
            ));
            return;
        }
        // Reuse branch returns before any backend probe/download/spawn, so
        // blocking here is a few field writes, not I/O.
        let outcome = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(agent.enable_local_skeptic(false))
        });
        match outcome {
            Ok(hi_agent::LocalSkepticOutcome::Ready { model_id, .. }) => {
                self.push(Line::styled(
                    format!("skeptic → {model_id} @ local (free review on the team server)"),
                    dim(),
                ));
            }
            Ok(_) | Err(_) => {
                self.push(Line::styled(
                    "skeptic unchanged — use /config skeptic-local on for the dedicated flow",
                    dim(),
                ));
            }
        }
    }

    /// Complete the auto-setup flow without toggling off an explicitly
    /// configured local skeptic. Interactive `/team` uses toggle semantics;
    /// automatic setup should be idempotent and preserve the user's choice.
    fn enable_team_skeptic_for_auto(&mut self, agent: &mut hi_agent::Agent) {
        if agent.local_skeptic_endpoint().is_some() {
            self.push(Line::styled(
                "skeptic already uses a local server — preserving that route",
                dim(),
            ));
            return;
        }
        if agent.any_team_local_server().is_none() {
            self.push(Line::styled(
                "team setup finished without a local server for skeptic; skeptic stays on the driver",
                dim(),
            ));
            return;
        }
        let outcome = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(agent.enable_local_skeptic(false))
        });
        match outcome {
            Ok(hi_agent::LocalSkepticOutcome::Ready { model_id, .. }) => {
                self.push(Line::styled(
                    format!("skeptic → {model_id} @ local (free review on the team server)"),
                    dim(),
                ));
            }
            Ok(_) | Err(_) => self.push(Line::styled(
                "skeptic stays on the driver — local team setup was unavailable",
                dim(),
            )),
        }
    }

    /// `/team <role>` with no model: open the picker over the supported
    /// catalog, largest first, annotated with what fits this machine and
    /// what's already downloaded. Enter assigns the selection to the role.
    pub(crate) fn open_team_model_picker(&mut self, role: &str) {
        let ram = hi_agent::local_skeptic::system_ram_gb();
        let backend = hi_agent::local_skeptic::detect_backend_cached();
        let mut entries: Vec<&'static hi_agent::local_skeptic::SupportedLocalModel> =
            hi_agent::local_skeptic::SUPPORTED_LOCAL_MODELS
                .iter()
                .collect();
        // Largest first, sized by the quant this machine would actually get
        // (the ladder means a family's effective size is per-machine).
        entries.sort_by_key(|entry| {
            std::cmp::Reverse(
                entry
                    .pick_mlx(ram)
                    .or_else(|| entry.smallest_mlx())
                    .map(|quant| quant.min_ram_gb)
                    .unwrap_or_else(|| entry.min_ram_gb(backend)),
            )
        });
        let rows: Vec<String> = entries
            .iter()
            .map(|entry| team_picker_row(entry, ram, backend))
            .collect();
        let auto_name = hi_agent::local_skeptic::resolve_team_local_model("local", ram, backend)
            .map(|auto| auto.entry.name);
        let current = rows
            .iter()
            .find(|row| auto_name.is_some_and(|name| row.starts_with(name)))
            .cloned()
            .unwrap_or_default();
        self.team_picker_role = Some(role.to_string());
        self.picker = Some(ModelPicker::new(
            rows,
            &current,
            std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
        ));
        self.push(Line::styled(
            format!("pick a local model for {role} — ↑↓ to choose, Enter to set up, Esc to cancel"),
            dim(),
        ));
        self.follow();
    }

    /// Start provisioning a local model selected from the driver provider
    /// picker. The current provider remains active until the poller applies a
    /// verified runtime.
    pub(crate) async fn start_local_provider_provision(
        &mut self,
        agent: &mut hi_agent::Agent,
        model_name: &str,
    ) {
        if self
            .pending_local_provider
            .as_ref()
            .is_some_and(|pending| !pending.task.is_finished())
        {
            self.push(Line::styled(
                "a local model is already being prepared — wait for it to finish",
                dim(),
            ));
            return;
        }
        let Some(backend) = hi_agent::local_skeptic::detect_backend_cached() else {
            self.push(Line::styled(
                "local MLX is available only on Apple Silicon (or use a CUDA local model)",
                dim(),
            ));
            return;
        };
        let ram = hi_agent::local_skeptic::system_ram_gb();
        let catalog_model = hi_agent::local_skeptic::cached_pipenetwork_catalog()
            .and_then(|catalog| catalog.into_iter().find(|model| model.repo == model_name));
        if let Some(model) = catalog_model.as_ref() {
            let available =
                hi_tools::available_space_bytes(&hi_tools::skeptic_model_dir(&model.repo));
            if !model.fits_machine(ram, available) {
                self.push(Line::styled(
                    format!(
                        "{} no longer fits the available RAM or disk; local setup cancelled",
                        model.display_name
                    ),
                    dim(),
                ));
                return;
            }
        }
        let (display, runtime) = if let Some(model) = catalog_model {
            // Live Pipe Network rows use their repository id as the stable
            // picker action. They are intentionally not required to be in the
            // curated short-name catalog.
            let runtime =
                match hi_agent::local_skeptic::local_runtime_spec(model_name, ram, backend) {
                    Ok(runtime) => runtime,
                    Err(error) => {
                        self.push(Line::styled(
                            format!("local setup failed: {error:#}"),
                            dim(),
                        ));
                        return;
                    }
                };
            (model.display_name, runtime)
        } else {
            let Some(resolved) =
                hi_agent::local_skeptic::resolve_team_local_model(model_name, ram, Some(backend))
            else {
                self.push(Line::styled(
                    format!("unknown local model '{model_name}'"),
                    dim(),
                ));
                return;
            };
            if resolved.mlx.is_none_or(|quant| ram < quant.min_ram_gb) {
                let needed = resolved
                    .mlx
                    .map(|quant| quant.min_ram_gb)
                    .unwrap_or_default();
                self.push(Line::styled(
                    format!("{model_name} needs at least {needed}GB RAM; local setup cancelled"),
                    dim(),
                ));
                return;
            }
            let runtime =
                match hi_agent::local_skeptic::local_runtime_spec(model_name, ram, backend) {
                    Ok(runtime) => runtime,
                    Err(error) => {
                        self.push(Line::styled(
                            format!("local setup failed: {error:#}"),
                            dim(),
                        ));
                        return;
                    }
                };
            (resolved.display(), runtime)
        };
        if let Some((endpoint, model_id)) = agent.running_local_model_server(&runtime.model_id) {
            let process_id = agent
                .running_local_model_process(&runtime.model_id)
                .unwrap_or_default();
            let ready = hi_agent::local_skeptic::ManagedLocalRuntime {
                runtime_id: hi_agent::local_skeptic::local_runtime_id(&runtime),
                profile_name: runtime.profile_name,
                repo: runtime.repo,
                model_id,
                base_url: endpoint,
                process_id,
                model_dir: runtime.model_dir,
                backend: runtime.backend,
            };
            self.apply_local_provider_runtime(agent, ready).await;
            return;
        }
        let model_dir = runtime.model_dir.clone();
        let (phase_tx, phase_rx) =
            tokio::sync::watch::channel(hi_agent::local_skeptic::LocalRuntimePhase::Resolving);
        let task = tokio::spawn(async move {
            hi_agent::local_skeptic::provision_local_runtime(runtime, phase_tx).await
        });
        self.pending_local_provider = Some(crate::PendingLocalProviderProvision {
            display: display.clone(),
            task,
            phase_rx,
            announced_phase: hi_agent::local_skeptic::LocalRuntimePhase::Resolving,
            phase_started: std::time::Instant::now(),
            ticks_since_report: 0,
            model_dir,
            last_reported_bytes: 0,
            progress_entry_index: None,
        });
        self.push(Line::styled(
            format!(
                "⟳ preparing {display} locally — download, MLX startup, and verification run in the background"
            ),
            dim(),
        ));
    }

    /// Cancel an in-flight driver-local setup without changing the active
    /// provider. The provisioning task owns a cleanup guard for any server it
    /// has already spawned; aborting it therefore cannot leak a process.
    pub(crate) fn cancel_pending_local_provider(&mut self) -> bool {
        if self.cancel_pending_local_provider_if_active() {
            return true;
        }
        self.push(Line::styled("no local model setup is in progress", dim()));
        false
    }

    pub(crate) fn cancel_pending_local_provider_if_active(&mut self) -> bool {
        let Some(pending) = self.pending_local_provider.take() else {
            return false;
        };
        pending.task.abort();
        self.push(Line::styled(
            format!(
                "local MLX setup for {} cancelled; current provider remains active",
                pending.display
            ),
            dim(),
        ));
        true
    }

    /// Apply a finished local runtime. This is deliberately transactional:
    /// profile/provider state changes happen only after runtime verification.
    async fn apply_local_provider_runtime(
        &mut self,
        agent: &mut hi_agent::Agent,
        runtime: hi_agent::local_skeptic::ManagedLocalRuntime,
    ) {
        let switched = match (self.local_runtime_switcher)(&runtime) {
            Ok(switched) => switched,
            Err(error) => {
                hi_tools::stop_local_server(&runtime.process_id);
                self.push(Line::styled(
                    format!("local MLX profile switch failed: {error:#}"),
                    Style::default().fg(crate::theme::theme().warning),
                ));
                return;
            }
        };
        let label = switched.switched.label.clone();
        let model = switched.switched.model.clone();
        let profile = runtime.profile_name.clone();
        agent.register_driver_local_server(
            runtime.base_url.clone(),
            runtime.model_id.clone(),
            runtime.process_id.clone(),
        );
        agent.set_provider(
            switched.switched.provider.into(),
            model.clone(),
            None,
            switched.switched.max_tokens,
            switched.switched.max_tokens_explicit,
            None,
        );
        if let Ok(models) = agent.list_models().await {
            self.served = models
                .into_iter()
                .map(|model| (model.id.clone(), model))
                .collect();
        }
        self.provider = label.clone();
        self.model = model.clone();
        self.active_profile = Some(profile.clone());
        self.profiles = switched.profiles;
        self.apply_model(agent, &model);
        self.remember_session_routing();
        self.push(Line::styled(
            format!("using local MLX profile '{profile}' — model: {model}"),
            dim(),
        ));
    }

    /// Non-blocking poller for managed driver-local setup.
    pub(crate) async fn poll_pending_local_provider(&mut self, agent: &mut hi_agent::Agent) {
        let finished = self
            .pending_local_provider
            .as_ref()
            .is_some_and(|pending| pending.task.is_finished());
        if !finished {
            let mut transition = None;
            let mut heartbeat = None;
            let mut index = None;
            if let Some(pending) = &mut self.pending_local_provider {
                // Phase transitions are permanent transcript lines. Each new
                // phase starts a fresh in-place heartbeat line so the bar never
                // overwrites a completed phase announcement.
                let current = pending.phase_rx.borrow().clone();
                if current != pending.announced_phase {
                    pending.announced_phase = current.clone();
                    pending.phase_started = std::time::Instant::now();
                    pending.ticks_since_report = 0;
                    pending.progress_entry_index = None;
                    transition = Some(local_runtime_phase_line(&pending.display, &current));
                }

                pending.ticks_since_report = pending.ticks_since_report.saturating_add(1);
                if pending.ticks_since_report
                    >= local_runtime_heartbeat_ticks(&pending.announced_phase)
                {
                    pending.ticks_since_report = 0;
                    let bytes = dir_size_shallow(&pending.model_dir);
                    if let Some(line) = local_runtime_heartbeat_line(
                        &pending.display,
                        &pending.announced_phase,
                        pending.phase_started.elapsed(),
                        bytes,
                        pending.last_reported_bytes,
                    ) {
                        pending.last_reported_bytes = bytes;
                        heartbeat = Some(line);
                        index = Some(pending.progress_entry_index);
                    }
                }
            }

            let mut redraw = false;
            if let Some(line) = transition {
                self.push(Line::styled(line, dim()));
                redraw = true;
            }
            if let Some(line) = heartbeat {
                let mut slot = index.flatten();
                self.push_or_replace_progress(&mut slot, "⟳", Line::styled(line, dim()));
                if let Some(pending) = &mut self.pending_local_provider {
                    pending.progress_entry_index = slot;
                }
                redraw = true;
            }
            if redraw {
                self.follow();
            }
            return;
        }
        let pending = self.pending_local_provider.take().expect("pending checked");
        match pending.task.await {
            Ok(Ok(runtime)) => self.apply_local_provider_runtime(agent, runtime).await,
            Ok(Err(error)) => self.push(Line::styled(
                format!("local MLX setup failed; current provider remains active: {error:#}"),
                Style::default().fg(crate::theme::theme().warning),
            )),
            Err(error) => self.push(Line::styled(
                format!("local MLX setup task failed; current provider remains active: {error}"),
                Style::default().fg(crate::theme::theme().warning),
            )),
        }
    }
}

/// One picker row: `name — label · <quant/fit note> [· quants …] [· downloaded]`.
/// The fit note names the quant this machine would get, so a ladder like
/// Laguna's reads honestly: `3bit fits (needs 64GB RAM)` on a 64GB Mac.
pub(crate) fn team_picker_row(
    entry: &'static hi_agent::local_skeptic::SupportedLocalModel,
    ram_gb: u64,
    backend: Option<hi_agent::local_skeptic::LocalBackend>,
) -> String {
    let chosen = entry.pick_mlx(ram_gb);
    let fit = match (backend, chosen, entry.smallest_mlx()) {
        (Some(hi_agent::local_skeptic::LocalBackend::Cuda), _, _) => match entry.cuda {
            Some(cuda) if ram_gb >= cuda.min_ram_gb => {
                format!("needs {}GB RAM · fits", cuda.min_ram_gb)
            }
            Some(cuda) => {
                format!("needs {}GB RAM · too big for this machine", cuda.min_ram_gb)
            }
            None => "MLX-only — not packaged for CUDA yet".to_string(),
        },
        (_, Some(quant), _) if entry.mlx.len() > 1 => {
            format!("{} fits (needs {}GB RAM)", quant.quant, quant.min_ram_gb)
        }
        (_, Some(quant), _) => format!("needs {}GB RAM · fits", quant.min_ram_gb),
        (_, None, Some(smallest)) => {
            format!(
                "needs {}GB+ RAM · too big for this machine",
                smallest.min_ram_gb
            )
        }
        (_, None, None) => "unavailable".to_string(),
    };
    let downloaded = backend
        .and_then(|backend| {
            let resolved = hi_agent::local_skeptic::ResolvedLocalModel {
                entry,
                mlx: chosen.or_else(|| entry.smallest_mlx()),
            };
            hi_agent::local_skeptic::team_model_spec(resolved, backend).ok()
        })
        .map(|spec| {
            let dir = hi_tools::skeptic_model_dir(&spec.repo);
            hi_agent::local_skeptic::model_present(&dir, &spec)
        })
        .unwrap_or(false);
    let mut row = format!("{} — {} · {fit}", entry.name, entry.label);
    if entry.mlx.len() > 1 {
        row.push_str(&format!(" · quants {}", entry.quant_summary()));
    }
    if downloaded {
        row.push_str(" · downloaded");
    }
    row
}

#[cfg(test)]
mod provision_narration_tests {
    use super::*;
    use hi_agent::local_skeptic::{LocalRuntimePhase, ProvisionPhase};

    fn loading_phase() -> ProvisionPhase {
        ProvisionPhase::LoadingModel {
            deadline_secs: 345,
            server_handle: "bg_none".into(),
            expected_bytes: 19 * 1024 * 1024 * 1024,
        }
    }

    fn local_loading_phase() -> LocalRuntimePhase {
        LocalRuntimePhase::LoadingModel {
            deadline_secs: 345,
            server_handle: "bg_none".into(),
            expected_bytes: 9 * 1024 * 1024 * 1024,
        }
    }

    #[test]
    fn phase_lines_and_heartbeats_narrate_the_slow_parts() {
        assert!(provision_phase_line("coder-32b", &loading_phase()).contains("up to 5m45s"));
        // Unknown server pid → honest elapsed line, no fake bar.
        let hb = provision_heartbeat_line(
            "coder-32b",
            &loading_phase(),
            std::time::Duration::from_secs(95),
            0,
            0,
        )
        .unwrap();
        assert!(hb.contains("1m35s elapsed"), "{hb}");
        let dl = provision_heartbeat_line(
            "coder-32b",
            &ProvisionPhase::Downloading,
            std::time::Duration::from_secs(30),
            3 * 1024 * 1024 * 1024,
            1024,
        )
        .unwrap();
        assert!(dl.contains("3.0 GiB on disk"), "{dl}");
        assert!(
            provision_heartbeat_line("x", &ProvisionPhase::Resolving, Default::default(), 0, 0)
                .is_none(),
            "resolving is instant; no heartbeat spam"
        );
        assert!(
            provision_heartbeat_ticks(&loading_phase())
                < provision_heartbeat_ticks(&ProvisionPhase::Downloading),
            "loading refreshes fastest — that's the phase mistaken for a hang"
        );
    }

    #[test]
    fn loading_bar_reflects_memory_growth_and_clamps() {
        let gib = 1024u64 * 1024 * 1024;
        let half = loading_bar_line(
            "coder-32b",
            Some(9 * gib + gib / 2),
            19 * gib,
            std::time::Duration::from_secs(70),
            345,
        );
        assert!(half.contains("50%"), "{half}");
        assert!(half.contains("9.5/19.0 GiB"), "{half}");
        assert!(half.contains('▰') && half.contains('▱'), "{half}");
        let over = loading_bar_line(
            "coder-32b",
            Some(25 * gib),
            19 * gib,
            std::time::Duration::from_secs(200),
            345,
        );
        assert!(over.contains("99%"), "clamps below done: {over}");
        assert_eq!(render_bar(0.5, 10), "▰▰▰▰▰▱▱▱▱▱");
        assert_eq!(render_bar(2.0, 4), "▰▰▰▰");
    }

    #[test]
    fn provider_picker_loading_reports_elapsed_progress_when_pid_is_unavailable() {
        let phase = local_loading_phase();
        assert!(local_runtime_phase_line("deepseek", &phase).contains("up to 5m45s"));
        let line = local_runtime_heartbeat_line(
            "deepseek",
            &phase,
            std::time::Duration::from_secs(95),
            0,
            0,
        )
        .unwrap();
        assert!(line.contains("1m35s elapsed"), "{line}");
        assert_eq!(local_runtime_heartbeat_ticks(&phase), 8);
    }

    #[test]
    fn provider_picker_download_and_runtime_phases_have_live_heartbeats() {
        let download = LocalRuntimePhase::Downloading;
        let line = local_runtime_heartbeat_line(
            "deepseek",
            &download,
            std::time::Duration::from_secs(30),
            3 * 1024 * 1024 * 1024,
            1024,
        )
        .unwrap();
        assert!(line.contains("3.0 GiB on disk"), "{line}");
        let preparing = local_runtime_heartbeat_line(
            "deepseek",
            &LocalRuntimePhase::PreparingRuntime,
            std::time::Duration::from_secs(30),
            0,
            0,
        )
        .unwrap();
        assert!(preparing.contains("30s elapsed"), "{preparing}");
    }
}
