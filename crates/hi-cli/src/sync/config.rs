//! Public configuration and identity validation for remote session sync.

use anyhow::Result;

/// Constrain session IDs to one safe URL and filename segment.
pub fn validate_session_id(id: &str) -> Result<()> {
    if id.is_empty()
        || id.len() > 128
        || matches!(id, "." | "..")
        || !id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
    {
        anyhow::bail!("invalid session id: use 1-128 ASCII letters, digits, '.', '_' or '-'");
    }
    Ok(())
}

/// Configuration for syncing a session to ipop.
#[derive(Clone, Debug)]
pub struct SyncConfig {
    /// The ipop API base URL, e.g. `https://api.pipenetwork.ai/v1`.
    pub base_url: String,
    /// The project API key for authentication.
    pub api_key: String,
    /// A stable identifier for this machine (so a remote viewer knows where
    /// the coding work runs). If `None`, the server omits it.
    pub machine_id: Option<String>,
    /// The hi cwd digest (16 hex chars) — groups sessions by project.
    pub cwd_digest: Option<String>,
}

/// What the host reports to the control plane on heartbeat.
#[derive(Clone, Default)]
pub struct HeartbeatTelemetry {
    pub model: Option<String>,
    pub context_used_tokens: Option<u64>,
    pub context_max_tokens: Option<u64>,
}
