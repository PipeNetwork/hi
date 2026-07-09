//! Reach-you notifications for loud loop events (a change a firing found, a
//! landed fix, a budget pause) — so a headless daemon running while you're away
//! can actually reach you, not just log to a transcript you're not watching.
//!
//! Two best-effort, opt-in sinks, configured from the environment:
//! - **Desktop** (`HI_NOTIFY_DESKTOP=1`): `terminal-notifier` on macOS or
//!   `notify-send` on Linux — whichever is on PATH.
//! - **Webhook** (`HI_NOTIFY_WEBHOOK=<url>`): a JSON `{"text":"…"}` POST via
//!   `curl` (Slack/Mattermost-compatible; many receivers accept it).
//!
//! Everything here is fire-and-forget: a missing tool or a failed POST never
//! blocks or fails a firing.

/// Where loud events should be sent (read once from the environment).
#[derive(Clone, Default)]
pub(crate) struct NotifyConfig {
    /// A JSON webhook URL (`HI_NOTIFY_WEBHOOK`).
    webhook: Option<String>,
    /// Whether to post OS desktop notifications (`HI_NOTIFY_DESKTOP`).
    desktop: bool,
}

impl NotifyConfig {
    pub(crate) fn from_env() -> Self {
        NotifyConfig {
            webhook: std::env::var("HI_NOTIFY_WEBHOOK")
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty()),
            desktop: matches!(
                std::env::var("HI_NOTIFY_DESKTOP").ok().as_deref(),
                Some("1" | "on" | "true" | "yes")
            ),
        }
    }

    /// Whether any sink is configured (skip the work entirely otherwise).
    pub(crate) fn enabled(&self) -> bool {
        self.webhook.is_some() || self.desktop
    }

    /// A short description of the active sinks (for the daemon startup line).
    pub(crate) fn describe(&self) -> Option<String> {
        match (self.desktop, self.webhook.is_some()) {
            (true, true) => Some("desktop + webhook".into()),
            (true, false) => Some("desktop".into()),
            (false, true) => Some("webhook".into()),
            (false, false) => None,
        }
    }
}

/// Fire a notification for a loud event, if any sink is configured. Spawns a
/// detached task so it never blocks the caller (and reaps the child).
pub(crate) fn maybe_notify(cfg: &NotifyConfig, title: &str, body: &str) {
    if !cfg.enabled() {
        return;
    }
    let (cfg, title, body) = (cfg.clone(), title.to_string(), truncate(body, 240));
    tokio::spawn(send(cfg, title, body));
}

async fn send(cfg: NotifyConfig, title: String, body: String) {
    use std::process::Stdio;

    if cfg.desktop {
        // macOS `terminal-notifier`, else Linux `notify-send` — first on PATH wins.
        let candidates: [(&str, Vec<String>); 2] = [
            (
                "terminal-notifier",
                vec![
                    "-title".into(),
                    title.clone(),
                    "-message".into(),
                    body.clone(),
                ],
            ),
            ("notify-send", vec![title.clone(), body.clone()]),
        ];
        for (bin, args) in candidates {
            if let Ok(mut child) = tokio::process::Command::new(bin)
                .args(&args)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
            {
                let _ = child.wait().await;
                break; // one notifier exists; don't run the other
            }
        }
    }

    if let Some(url) = &cfg.webhook {
        let payload = webhook_payload(&title, &body);
        let _ = tokio::process::Command::new("curl")
            .args([
                "-s",
                "-m",
                "10",
                "-X",
                "POST",
                "-H",
                "Content-Type: application/json",
                "-d",
                &payload,
                url,
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map(|mut c| async move {
                let _ = c.wait().await;
            });
    }
}

/// The JSON body posted to the webhook (`{"text":"<title>: <body>"}`), with the
/// message safely escaped.
fn webhook_payload(title: &str, body: &str) -> String {
    format!(r#"{{"text":"{}: {}"}}"#, escape(title), escape(body))
}

/// Minimal JSON string escaping for the webhook payload.
fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push(' '),
            c => out.push(c),
        }
    }
    out
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let cut: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{cut}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn webhook_payload_escapes_json() {
        let p = webhook_payload("loop#3", "CI \"red\": 3 failures\nline2");
        assert!(p.starts_with(r#"{"text":"loop#3: "#), "{p}");
        assert!(p.contains(r#"\"red\""#), "quotes escaped: {p}");
        assert!(p.contains(r"\n"), "newline escaped: {p}");
        // Must be valid JSON.
        let v: serde_json::Value = serde_json::from_str(&p).expect("valid json");
        assert!(
            v.get("text")
                .and_then(|t| t.as_str())
                .unwrap()
                .contains("red")
        );
    }

    #[test]
    fn config_default_is_disabled() {
        let cfg = NotifyConfig::default();
        assert!(!cfg.enabled());
    }

    #[test]
    fn config_enables_with_a_sink() {
        let cfg = NotifyConfig {
            webhook: Some("https://example.com/hook".into()),
            desktop: false,
        };
        assert!(cfg.enabled());
    }
}
