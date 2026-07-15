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

/// Session id for feedback and portal sync, derived from the session path.
///
/// The id is the portal's ROUTING KEY (a URL path segment on every sync
/// call), so it must be unique per logical session and stable across resumes
/// of the same file. Auto-generated paths under the sessions dir already have
/// machine/process/counter-unique stems and keep their plain stem for portal
/// continuity. Explicit `--session-file` paths get a canonical-path hash
/// suffix — a bare stem lets two `session.json` files in different
/// directories merge into one remote session (and previously share one sync
/// offset row, the wedged-offset bug).
pub(crate) fn session_id_from_path(path: &Path) -> String {
    session_id_for(path, crate::session::sessions_dir().as_deref())
}

fn session_id_for(path: &Path, sessions_dir: Option<&Path>) -> String {
    const MAX_STEM: usize = 40;
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.trim().is_empty())
        .unwrap_or("unknown");
    let mut sanitized: String = stem
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '-'
            }
        })
        .take(MAX_STEM)
        .collect();
    if sanitized.trim_matches(['.', '-']).is_empty() {
        sanitized = "session".to_string();
    }
    let identity = stable_path_identity(path);
    if let Some(dir) = sessions_dir
        && identity.starts_with(stable_path_identity(dir))
    {
        return sanitized;
    }
    let digest = {
        use sha2::{Digest, Sha256};
        format!(
            "{:x}",
            Sha256::digest(identity.to_string_lossy().as_bytes())
        )
    };
    format!("{sanitized}-{}", &digest[..8])
}

/// A path identity stable across the file's creation: the canonical path when
/// it exists, else the canonicalized parent joined with the file name.
fn stable_path_identity(path: &Path) -> PathBuf {
    if let Ok(canonical) = path.canonicalize() {
        return canonical;
    }
    if let (Some(parent), Some(name)) = (path.parent(), path.file_name()) {
        let parent = if parent.as_os_str().is_empty() {
            Path::new(".")
        } else {
            parent
        };
        if let Ok(parent) = parent.canonicalize() {
            return parent.join(name);
        }
    }
    path.to_path_buf()
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
        _ => Some(FeedbackChoice::Dismiss),
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
    use super::session_id_for;

    fn scratch(label: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hi-session-id-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn sessions_dir_paths_keep_their_plain_stem() {
        let sessions = scratch("auto");
        let path = sessions.join("1234567890000-ab12-cd-1.jsonl");
        std::fs::write(&path, "").unwrap();
        assert_eq!(
            session_id_for(&path, Some(&sessions)),
            "1234567890000-ab12-cd-1"
        );
        let _ = std::fs::remove_dir_all(sessions);
    }

    #[test]
    fn explicit_session_files_get_a_stable_path_hash_suffix() {
        let dir_a = scratch("a");
        let dir_b = scratch("b");
        let a = dir_a.join("session.json");
        let b = dir_b.join("session.json");
        std::fs::write(&a, "").unwrap();

        let id_a = session_id_for(&a, None);
        let id_b = session_id_for(&b, None);
        assert!(id_a.starts_with("session-"), "{id_a}");
        assert_ne!(id_a, id_b, "same stem in different dirs must not collide");
        // Stable across calls and across the file's creation (b not written).
        assert_eq!(id_a, session_id_for(&a, None));
        assert_eq!(id_b, {
            std::fs::write(&b, "").unwrap();
            session_id_for(&b, None)
        });
        crate::sync::validate_session_id(&id_a).expect("derived id is valid");
        let _ = std::fs::remove_dir_all(dir_a);
        let _ = std::fs::remove_dir_all(dir_b);
    }

    #[test]
    fn hostile_stems_are_sanitized_and_bounded() {
        let dir = scratch("hostile");
        let path = dir.join("my session (v2)!.json");
        std::fs::write(&path, "").unwrap();
        let id = session_id_for(&path, None);
        crate::sync::validate_session_id(&id).expect("sanitized id is valid");
        assert!(id.starts_with("my-session--v2--"), "{id}");

        let dots = dir.join("....json");
        let id = session_id_for(&dots, None);
        crate::sync::validate_session_id(&id).expect("dot stem replaced");
        assert!(id.starts_with("session-"), "{id}");

        let long = dir.join(format!("{}.json", "x".repeat(200)));
        let id = session_id_for(&long, None);
        crate::sync::validate_session_id(&id).expect("long stem bounded");
        assert!(id.len() <= 49, "{}", id.len());
        let _ = std::fs::remove_dir_all(dir);
    }
}
