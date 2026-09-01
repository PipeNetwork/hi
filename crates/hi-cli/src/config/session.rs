use super::*;

/// Per-workspace snapshot of the last interactive provider/model selection.
/// Written under `.hi/last_session.toml` so the next bare `hi` in this
/// workspace resumes with the same routing without requiring a config edit.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct LastSession {
    /// Active profile name when one was selected (`None` for provider presets).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
    /// Provider label (`openai`, `anthropic`, `xai`, …).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Model id in force when the session ended.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// Path of the workspace last-session file under `root` (default `.`).
pub fn last_session_path(root: &Path) -> PathBuf {
    root.join(".hi").join("last_session.toml")
}

/// Load the last interactive session snapshot for `root`, if present.
pub fn load_last_session(root: &Path) -> Option<LastSession> {
    let path = last_session_path(root);
    let text = std::fs::read_to_string(&path).ok()?;
    toml::from_str(&text).ok()
}

/// Persist the active provider/model (and profile, when one is selected) so the
/// next bare `hi` in this workspace restores the same routing.
pub fn save_last_session(root: &Path, session: &LastSession) -> Result<()> {
    let path = last_session_path(root);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let toml = toml::to_string_pretty(session)
        .with_context(|| format!("serializing {}", path.display()))?;
    std::fs::write(&path, toml).with_context(|| format!("writing {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

/// Convenience: build + write a last-session snapshot.
pub fn remember_session(
    root: &Path,
    profile: Option<&str>,
    provider: &str,
    model: &str,
) -> Result<()> {
    // Skip placeholder model ids that mean "not configured yet".
    let model = model.trim();
    if model.is_empty() || model == "__model_not_configured__" {
        return Ok(());
    }
    let session = LastSession {
        profile: profile
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string),
        provider: {
            let p = provider.trim();
            (!p.is_empty()).then(|| p.to_string())
        },
        model: Some(model.to_string()),
    };
    save_last_session(root, &session)
}

/// Profile name the interactive shell should treat as active on startup.
///
/// Order: explicit `--profile` → last-session profile (when it still exists) →
/// config `default_profile`. A last-session snapshot with no profile means the
/// user was on a provider preset (`/provider xai`); in that case we deliberately
/// return `None` instead of falling back to `default_profile`, otherwise the
/// next exit would rewrite `.hi/last_session.toml` under the default profile and
/// discard the preset routing on the following launch.
pub fn resolve_active_profile(cli: &Cli, config: &Config, root: &Path) -> Option<String> {
    if let Some(name) = cli.profile.clone() {
        return Some(name);
    }
    // CLI provider/model overrides mean last-session routing does not apply.
    if cli.model.is_some() || cli.provider.is_some() {
        return config.default_profile.clone();
    }
    let last = load_last_session(root);
    if let Some(name) = last
        .as_ref()
        .and_then(|s| s.profile.clone())
        .filter(|name| config.profiles.contains_key(name))
    {
        return Some(name);
    }
    // Preset last-session: keep active_profile unset so remember_session writes
    // profile=None again on exit.
    if last.is_some() {
        return None;
    }
    config.default_profile.clone()
}
