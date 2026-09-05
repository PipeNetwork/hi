//! Session-bound remote input hosting and cancellation of delayed enablement.

use crate::render::dim;
use ratatui::{style::Style, text::Line};

impl crate::App {
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
        if enable && self.pending_host_enable.is_some() {
            self.push(Line::styled("host mode is starting", dim()));
            self.follow();
            return;
        }
        if !enable {
            let active = self.hosting_remote_input || self.pending_host_enable.is_some();
            // Local input acceptance must stop even if publishing the portal
            // update fails. A pending startup enable is also an active request.
            self.stop_host_mode();
            if !active {
                self.push(Line::styled("host mode is already off", dim()));
                self.follow();
                return;
            }
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
    /// explicit `/sessions host` command gets a calm note (`announce`); the
    /// automatic startup enable stays fully silent so the empty-session
    /// wordmark is not buried under host-mode chatter.
    pub(crate) fn apply_host_toggle_result(
        &mut self,
        result: anyhow::Result<Option<crate::SessionHostEnable>>,
        announce: bool,
    ) {
        match result {
            Ok(enabled) => {
                self.stop_host_mode();
                if let Some((rx, abort)) = enabled {
                    self.remote_input_rx = Some(rx);
                    self.remote_input_poller = Some(abort);
                    self.hosting_remote_input = true;
                    if announce {
                        self.push(Line::styled(
                            "✓ host on — remote attach clients can send prompts into this session",
                            Style::default().fg(crate::theme::theme().accent_success),
                        ));
                        self.push(Line::styled(
                            "  other machines: /sessions attach <id>  (or hi --attach <id>)",
                            dim(),
                        ));
                        self.follow();
                    }
                } else if announce {
                    self.push(Line::styled(
                        "host off — no longer accepting remote prompts",
                        dim(),
                    ));
                    self.follow();
                }
            }
            Err(_) if announce => {
                self.push(Line::styled(
                    "host mode unavailable right now — the sync portal is unreachable; this session keeps working locally",
                    dim(),
                ));
                self.follow();
            }
            Err(_) => {}
        }
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

    pub(super) fn stop_host_mode(&mut self) {
        if let Some(task) = self.pending_host_enable.take() {
            task.abort();
            // An already-completed task can still own a newly spawned poller.
            // Reap its result so dropping the AbortHandle cannot detach that
            // old session's poller after a switch or an explicit host-off.
            tokio::spawn(async move {
                if let Ok(Ok(Some((_, abort)))) = task.await {
                    abort.abort();
                }
            });
        }
        if let Some(abort) = self.remote_input_poller.take() {
            abort.abort();
        }
        self.remote_input_rx = None;
        self.hosting_remote_input = false;
    }
}

#[cfg(test)]
#[path = "host_mode_tests.rs"]
mod tests;
