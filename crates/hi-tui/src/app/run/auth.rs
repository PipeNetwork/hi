//! Interactive API-key parsing, validation, and profile persistence.

use ratatui::text::Line;

use crate::render::dim;

pub(super) fn parse_tui_auth_arg(arg: &str) -> Result<(String, Option<String>), String> {
    let arg = arg.trim();
    if arg.is_empty() {
        return Err(
            "usage: /auth openai|anthropic|pipenetwork|xai [api-key]  (subscription pairing stays /login)"
                .into(),
        );
    }
    let (name, rest) = match arg.split_once(char::is_whitespace) {
        Some((name, rest)) => (name, rest.trim()),
        None => (arg, ""),
    };
    let provider = match name.to_ascii_lowercase().as_str() {
        "openai" | "openrouter" => "openai",
        "anthropic" => "anthropic",
        "pipenetwork" | "pipe" => "pipenetwork",
        "xai" | "grok" => "xai",
        other => {
            return Err(format!(
                "'{other}' has no pasted-key flow. Supported: openai, anthropic, pipenetwork, xai. \
                 Subscription pairing stays /login."
            ));
        }
    };
    let key = if rest.is_empty() {
        None
    } else {
        Some(rest.to_string())
    };
    Ok((provider.to_string(), key))
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum SubmittedLine {
    Auth { provider: String, key: String },
    Shell(String),
    Tutorial,
    Normal(String),
}

/// Classify a committed line before tracing or command handling. Queued lines
/// already crossed the trace boundary, so they must never become credentials.
pub(super) fn route_submitted_line(
    app: &mut crate::App,
    line: String,
    line_was_queued: bool,
) -> anyhow::Result<SubmittedLine> {
    app.completion = None;
    if !line_was_queued && let Some(provider) = app.pending_auth.take() {
        return Ok(SubmittedLine::Auth {
            provider,
            key: line.trim().to_string(),
        });
    }
    if !line_was_queued {
        app.trace_immediate_prompt(&line)?;
    }
    if let Some(shell_cmd) = line.strip_prefix('!').filter(|s| !s.trim().is_empty()) {
        return Ok(SubmittedLine::Shell(shell_cmd.to_string()));
    }
    if matches!(line.trim(), "/tutorial" | "/tour" | "/onboarding") {
        return Ok(SubmittedLine::Tutorial);
    }
    Ok(SubmittedLine::Normal(line))
}

pub(super) async fn apply_tui_auth(app: &mut crate::App, provider: &str, key: &str) {
    app.input.secret = false;
    if key.trim().is_empty() {
        app.push(Line::styled("no API key entered".to_string(), dim()));
        app.follow();
        return;
    }
    let (base_url, check) = match provider {
        "anthropic" => {
            let base = "https://api.anthropic.com";
            let p = hi_ai::AnthropicProvider::new(base.to_string(), key.to_string());
            (
                base.to_string(),
                hi_ai::KeyCheck::from_list_models(hi_ai::Provider::list_models(&p).await),
            )
        }
        "xai" => {
            let base = "https://api.x.ai/v1";
            let p = hi_ai::OpenAiProvider::new(base.to_string(), key.to_string());
            (
                base.to_string(),
                hi_ai::KeyCheck::from_list_models(hi_ai::Provider::list_models(&p).await),
            )
        }
        "pipenetwork" => {
            let base = "https://api.pipenetwork.ai/v1";
            let p = hi_ai::OpenAiProvider::new(base.to_string(), key.to_string());
            (
                base.to_string(),
                hi_ai::KeyCheck::from_list_models(hi_ai::Provider::list_models(&p).await),
            )
        }
        _ => {
            let base = "https://openrouter.ai/api/v1";
            let p = hi_ai::OpenAiProvider::new(base.to_string(), key.to_string());
            (
                base.to_string(),
                hi_ai::KeyCheck::from_list_models(hi_ai::Provider::list_models(&p).await),
            )
        }
    };
    save_checked_tui_auth(app, provider, key, base_url, check);
}

fn save_checked_tui_auth(
    app: &mut crate::App,
    provider: &str,
    key: &str,
    base_url: String,
    check: hi_ai::KeyCheck,
) {
    let unverified = match check {
        hi_ai::KeyCheck::Accepted => None,
        hi_ai::KeyCheck::Unverified(message) => Some(message),
        hi_ai::KeyCheck::Rejected(message) => {
            app.push(Line::styled(format!("not saved: {message}"), dim()));
            app.follow();
            return;
        }
    };
    let form = crate::ProfileFormData {
        name: provider.to_string(),
        provider: provider.to_string(),
        api_key: key.to_string(),
        store_as_env: false,
        model: String::new(),
        base_url,
    };
    match (app.saver)(&form) {
        Ok(profiles) => {
            app.profiles = profiles;
            let note = match unverified {
                Some(msg) => {
                    format!("saved {provider} (unverified: {msg}) — /provider {provider} to use it")
                }
                None => format!("saved {provider} — /provider {provider} to use it"),
            };
            app.push(Line::styled(note, dim()));
        }
        Err(err) => {
            app.push(Line::styled(format!("/auth failed: {err:#}"), dim()));
        }
    }
    app.follow();
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::{SubmittedLine, parse_tui_auth_arg, route_submitted_line, save_checked_tui_auth};

    #[test]
    fn pipenetwork_api_keys_are_accepted_without_conflating_subscription_login() {
        assert_eq!(
            parse_tui_auth_arg("pipenetwork api_test"),
            Ok(("pipenetwork".into(), Some("api_test".into())))
        );
        assert_eq!(parse_tui_auth_arg("pipe"), Ok(("pipenetwork".into(), None)));
    }

    #[test]
    fn pending_auth_routes_bang_and_slash_keys_only_to_the_auth_saver() {
        for key in ["!printf shell-sentinel", "/tutorial", "/quit"] {
            let dir = tempfile::tempdir().unwrap();
            let trace_path = dir.path().join("events.jsonl");
            let saved = Arc::new(Mutex::new(Vec::<crate::ProfileFormData>::new()));
            let captured = Arc::clone(&saved);
            let mut app = crate::tests::test_app("pipenetwork", "pipe/test");
            app.saver = Box::new(move |form| {
                captured.lock().unwrap().push(form.clone());
                Ok(Vec::new())
            });
            app.tui_event_trace = Some(crate::TuiEventTrace::open(&trace_path).unwrap());
            app.pending_auth = Some("pipenetwork".into());
            app.input.secret = true;
            app.input.set(key);
            app.sync_completion();
            assert!(
                app.completion.is_none(),
                "secret input must not complete: {key}"
            );
            let submitted = app.input.submit();
            assert_eq!(submitted, key);
            assert!(app.input.history.is_empty());

            let (provider, routed_key) = match route_submitted_line(&mut app, submitted, false)
                .expect("route pending auth")
            {
                SubmittedLine::Auth { provider, key } => (provider, key),
                other => panic!("pending key escaped auth routing: {other:?}"),
            };
            save_checked_tui_auth(
                &mut app,
                &provider,
                &routed_key,
                "https://api.pipenetwork.ai/v1".into(),
                hi_ai::KeyCheck::Accepted,
            );

            let saved = saved.lock().unwrap();
            assert_eq!(saved.len(), 1);
            assert_eq!(saved[0].provider, "pipenetwork");
            assert_eq!(saved[0].api_key, key);
            assert!(app.pending_auth.is_none());
            assert!(app.tutorial.is_none());
            assert!(app.queue.is_empty());
            assert!(app.last_prompt.is_none());
            assert_eq!(app.transcript.len(), 1, "only the auth result is shown");

            let trace = std::fs::read_to_string(&trace_path).unwrap();
            assert!(trace.is_empty(), "pending secret reached trace: {trace}");
            assert!(!trace.contains(key));
            assert!(!trace.contains("prompt_fingerprint"));
            assert!(!trace.contains("prompt_chars"));
        }
    }

    #[test]
    fn inline_auth_reaches_the_saver_with_only_redacted_trace_records() {
        let dir = tempfile::tempdir().unwrap();
        let trace_path = dir.path().join("events.jsonl");
        let saved = Arc::new(Mutex::new(Vec::<crate::ProfileFormData>::new()));
        let captured = Arc::clone(&saved);
        let mut app = crate::tests::test_app("pipenetwork", "pipe/test");
        app.saver = Box::new(move |form| {
            captured.lock().unwrap().push(form.clone());
            Ok(Vec::new())
        });
        app.tui_event_trace = Some(crate::TuiEventTrace::open(&trace_path).unwrap());

        let key = "inline-test-only-secret";
        let line = format!("/auth pipenetwork {key}");
        let routed = route_submitted_line(&mut app, line.clone(), false).unwrap();
        assert_eq!(routed, SubmittedLine::Normal(line.clone()));
        // Mid-turn input crosses this queue trace path before the main run loop.
        app.trace_prompt_queued(&line);

        let Some(hi_agent::Command::Auth(arg)) = hi_agent::command::parse(&line) else {
            panic!("inline auth command did not parse");
        };
        let (provider, parsed_key) = parse_tui_auth_arg(&arg).unwrap();
        save_checked_tui_auth(
            &mut app,
            &provider,
            parsed_key.as_deref().unwrap(),
            "https://api.pipenetwork.ai/v1".into(),
            hi_ai::KeyCheck::Accepted,
        );

        let saved = saved.lock().unwrap();
        assert_eq!(saved.len(), 1);
        assert_eq!(saved[0].api_key, key);
        let trace = std::fs::read_to_string(&trace_path).unwrap();
        assert!(trace.contains("\"sensitive\":true"));
        assert!(!trace.contains(key));
        assert!(!trace.contains(&blake3::hash(line.as_bytes()).to_hex().to_string()));
        let redacted = "/auth [redacted]";
        assert!(trace.contains(&blake3::hash(redacted.as_bytes()).to_hex().to_string()));
        assert!(trace.contains(&format!("\"prompt_chars\":{}", redacted.chars().count())));
    }
}
