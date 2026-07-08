use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::config::{ProviderName, Settings};

#[derive(Debug, Default, Serialize, Deserialize)]
struct FeedbackState {
    last_prompt_day: Option<u64>,
}

pub(crate) fn session_id_from_path(path: &Path) -> String {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.trim().is_empty())
        .unwrap_or("unknown")
        .to_string()
}

pub(crate) async fn maybe_prompt_and_submit(settings: &Settings, session_id: &str) {
    if settings.provider != ProviderName::Pipenetwork
        || !io::stdin().is_terminal()
        || !io::stdout().is_terminal()
    {
        return;
    }

    let Some(path) = state_path() else {
        return;
    };
    let today = current_day();
    let mut state = read_state(&path).unwrap_or_default();
    if state.last_prompt_day == Some(today) {
        return;
    }

    let Some(choice) = prompt_choice() else {
        return;
    };
    state.last_prompt_day = Some(today);
    let _ = write_state(&path, &state);
    if let Err(err) = submit_feedback(settings, session_id, choice).await {
        eprintln!("\x1b[33mfeedback not recorded: {err:#}\x1b[0m");
    }
}

#[derive(Clone, Copy)]
enum FeedbackChoice {
    Bad,
    Fine,
    Good,
    Dismiss,
}

fn prompt_choice() -> Option<FeedbackChoice> {
    println!("● How is Pipe doing this session? (optional)");
    println!("  1: Bad    2: Fine   3: Good   0: Dismiss");
    print!("› ");
    let _ = io::stdout().flush();

    let mut line = String::new();
    io::stdin().read_line(&mut line).ok()?;
    match line.trim() {
        "1" => Some(FeedbackChoice::Bad),
        "2" => Some(FeedbackChoice::Fine),
        "3" => Some(FeedbackChoice::Good),
        "0" | "" => Some(FeedbackChoice::Dismiss),
        _ => None,
    }
}

async fn submit_feedback(
    settings: &Settings,
    session_id: &str,
    choice: FeedbackChoice,
) -> Result<()> {
    let (rating, dismissed, label) = match choice {
        FeedbackChoice::Bad => (Some(-1), false, "bad"),
        FeedbackChoice::Fine => (Some(0), false, "fine"),
        FeedbackChoice::Good => (Some(1), false, "good"),
        FeedbackChoice::Dismiss => (None, true, "dismiss"),
    };
    let mut payload = json!({
        "session_id": session_id,
        "client": "hi",
        "dismissed": dismissed,
        "choice": label,
        "provider": "pipenetwork",
        "model": settings.model,
    });
    if let Some(rating) = rating {
        payload["rating"] = json!(rating);
    }

    let url = format!(
        "{}/agent-sessions/feedback",
        settings.base_url.trim_end_matches('/')
    );
    let response = hi_ai::agent_http_client()
        .post(url)
        .bearer_auth(&settings.api_key)
        .json(&payload)
        .send()
        .await
        .context("sending session feedback")?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("API error {status}: {text}");
    }
    Ok(())
}

fn read_state(path: &Path) -> Result<FeedbackState> {
    let text = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

fn write_state(path: &Path, state: &FeedbackState) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(state)?).map_err(Into::into)
}

fn state_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share"))
        })?;
    Some(base.join("hi").join("feedback").join("pipe-session.json"))
}

fn current_day() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() / 86_400)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::session_id_from_path;
    use std::path::Path;

    #[test]
    fn session_id_uses_file_stem() {
        assert_eq!(
            session_id_from_path(Path::new("/tmp/hi/1234567890000.jsonl")),
            "1234567890000"
        );
    }
}
