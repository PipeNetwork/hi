//! Interactive API-key parsing, validation, and profile persistence.

use ratatui::text::Line;

use crate::render::dim;

pub(super) fn parse_tui_auth_arg(arg: &str) -> Result<(String, Option<String>), String> {
    let arg = arg.trim();
    if arg.is_empty() {
        return Err("usage: /auth openai|anthropic|xai [api-key]  (pairing stays /login)".into());
    }
    let (name, rest) = match arg.split_once(char::is_whitespace) {
        Some((name, rest)) => (name, rest.trim()),
        None => (arg, ""),
    };
    let provider = match name.to_ascii_lowercase().as_str() {
        "openai" | "openrouter" => "openai",
        "anthropic" => "anthropic",
        "xai" | "grok" => "xai",
        other => {
            return Err(format!(
                "'{other}' has no pasted-key flow. Supported: openai, anthropic, xai. \
                 Pairing stays /login."
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
        _ => {
            let base = "https://openrouter.ai/api/v1";
            let p = hi_ai::OpenAiProvider::new(base.to_string(), key.to_string());
            (
                base.to_string(),
                hi_ai::KeyCheck::from_list_models(hi_ai::Provider::list_models(&p).await),
            )
        }
    };
    if let hi_ai::KeyCheck::Rejected(msg) = check {
        app.push(Line::styled(format!("not saved: {msg}"), dim()));
        app.follow();
        return;
    }
    let unverified = match &check {
        hi_ai::KeyCheck::Unverified(msg) => Some(msg.clone()),
        _ => None,
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
