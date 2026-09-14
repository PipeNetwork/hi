//! Supervisor and monitor knobs. Machine `[autoharnessfix]` only — never `--config PATH`.

use std::collections::HashMap;
use std::ffi::OsString;
use std::path::PathBuf;
use std::time::Duration;

use serde::Deserialize;

use crate::paths;

/// Cheap Pipe id for diagnose. Same string as pipenetwork `planner_model_default`.
pub const DEFAULT_DIAGNOSE_MODEL: &str = "pipe/glm-5.2-fast";
/// Stronger Pipe id when the machine profile has no `model` (hi-harness DEFAULT_MODEL).
pub const DEFAULT_PATCH_MODEL: &str = "pipe/deepseek-v4-flash-0731";

pub const PIPE_MODEL_ERROR: &str = "autoharnessfix diagnose_model/patch_model must be a pipe/… id";

#[derive(Clone, Debug)]
pub struct MonitorConfig {
    pub liveness_timeout: Duration,
    pub progress_timeout: Duration,
    pub live_pgid_report_cap: Duration,
    pub start_grace: Duration,
    pub poll_interval: Duration,
    pub term_grace: Duration,
    pub confirmation_exempt: bool,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            liveness_timeout: Duration::from_secs(15),
            progress_timeout: Duration::from_secs(180),
            live_pgid_report_cap: Duration::from_secs(900),
            start_grace: Duration::from_secs(5),
            poll_interval: Duration::from_millis(250),
            term_grace: Duration::from_secs(2),
            confirmation_exempt: true,
        }
    }
}

impl MonitorConfig {
    pub fn from_machine_and_env() -> Self {
        let mut cfg = Self::default();
        if let Some(section) = peek_machine() {
            if let Some(secs) = section.liveness_secs {
                cfg.liveness_timeout = Duration::from_secs(secs);
            }
            if let Some(secs) = section.stall_progress_secs {
                cfg.progress_timeout = Duration::from_secs(secs);
            }
            if let Some(secs) = section.live_pgid_report_secs {
                cfg.live_pgid_report_cap = Duration::from_secs(secs);
            }
        }
        override_ms("HI_SENTINEL_LIVENESS_MS", &mut cfg.liveness_timeout);
        override_ms("HI_SENTINEL_PROGRESS_MS", &mut cfg.progress_timeout);
        override_ms("HI_SENTINEL_START_GRACE_MS", &mut cfg.start_grace);
        override_ms("HI_SENTINEL_POLL_MS", &mut cfg.poll_interval);
        override_ms("HI_SENTINEL_TERM_GRACE_MS", &mut cfg.term_grace);
        override_ms(
            "HI_SENTINEL_LIVE_PGID_REPORT_MS",
            &mut cfg.live_pgid_report_cap,
        );
        cfg
    }
}

fn override_ms(name: &str, slot: &mut Duration) {
    if let Ok(raw) = std::env::var(name)
        && let Ok(ms) = raw.parse::<u64>()
    {
        *slot = Duration::from_millis(ms);
    }
}

#[derive(Clone, Debug)]
pub struct SupervisorConfig {
    pub child_program: PathBuf,
    pub child_args: Vec<OsString>,
    pub hi_binary: PathBuf,
    pub original_argv: Vec<String>,
    pub checkout: Option<PathBuf>,
    pub apply: bool,
    pub monitor: MonitorConfig,
    pub state_dir: PathBuf,
    pub workspace: PathBuf,
    pub generation: u32,
    pub inherit_stdio: bool,
    pub extra_env: Vec<(String, String)>,
    /// Tests: return after the first NotHarness/ReportOnly/HarnessBug.
    pub once: bool,
    /// Copy onto the new runtime `turn-intent.json` before a resume spawn.
    pub seed_turn_intent: Option<PathBuf>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct MachineSection {
    #[serde(default)]
    #[allow(dead_code)]
    pub enabled: bool,
    #[serde(default)]
    pub apply: bool,
    pub checkout: Option<PathBuf>,
    pub diagnose_model: Option<String>,
    pub patch_model: Option<String>,
    pub stall_progress_secs: Option<u64>,
    pub liveness_secs: Option<u64>,
    pub live_pgid_report_secs: Option<u64>,
    pub max_repairs_per_session: Option<u32>,
    pub max_attempts_per_incident: Option<u32>,
    pub max_modifications_per_hour: Option<u32>,
    pub incident_retention_days: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct RepairModels {
    pub diagnose: String,
    pub patch: String,
}

#[derive(Clone, Debug)]
pub struct RepairConfig {
    pub diagnose_timeout: Duration,
    pub patch_timeout: Duration,
    pub max_repairs_per_session: u32,
    pub max_attempts_per_incident: u32,
    pub max_modifications_per_hour: u32,
}

impl Default for RepairConfig {
    fn default() -> Self {
        Self {
            diagnose_timeout: Duration::from_secs(8 * 60),
            patch_timeout: Duration::from_secs(20 * 60),
            max_repairs_per_session: 2,
            max_attempts_per_incident: 1,
            max_modifications_per_hour: 3,
        }
    }
}

impl RepairConfig {
    pub fn from_machine_and_env() -> Self {
        let mut cfg = Self::default();
        if let Some(section) = peek_machine() {
            if let Some(n) = section.max_repairs_per_session {
                cfg.max_repairs_per_session = n.max(1);
            }
            if let Some(n) = section.max_attempts_per_incident {
                cfg.max_attempts_per_incident = n.max(1);
            }
            if let Some(n) = section.max_modifications_per_hour {
                cfg.max_modifications_per_hour = n;
            }
        }
        override_ms("HI_SENTINEL_DIAGNOSE_MS", &mut cfg.diagnose_timeout);
        override_ms("HI_SENTINEL_PATCH_MS", &mut cfg.patch_timeout);
        cfg
    }
}

#[derive(Default, Deserialize)]
struct MachineFile {
    default_profile: Option<String>,
    #[serde(default)]
    autoharnessfix: Option<MachineSection>,
    #[serde(default)]
    profiles: HashMap<String, ProfilePeek>,
}

#[derive(Default, Deserialize)]
struct ProfilePeek {
    model: Option<String>,
}

pub fn peek_machine() -> Option<MachineSection> {
    peek_machine_file()?.autoharnessfix
}

fn peek_machine_file() -> Option<MachineFile> {
    let path = paths::default_config_path()?;
    let text = std::fs::read_to_string(path).ok()?;
    toml::from_str(&text).ok()
}

pub fn peek_machine_profile_model() -> Option<String> {
    let file = peek_machine_file()?;
    let name = file
        .default_profile
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or("default");
    file.profiles
        .get(name)
        .and_then(|p| p.model.clone())
        .filter(|m| !m.trim().is_empty())
}

/// Both ids must be `pipe/…`. Empty / non-Pipe fail closed (skip repair).
pub fn resolve_repair_models(
    section: Option<&MachineSection>,
) -> Result<RepairModels, &'static str> {
    let diagnose = match section.and_then(|s| s.diagnose_model.as_deref()) {
        None => DEFAULT_DIAGNOSE_MODEL.to_string(),
        Some(id) => require_pipe(id)?,
    };
    let patch = match section.and_then(|s| s.patch_model.as_deref()) {
        None => {
            let fallback =
                peek_machine_profile_model().unwrap_or_else(|| DEFAULT_PATCH_MODEL.to_string());
            require_pipe(&fallback)?
        }
        Some(id) => require_pipe(id)?,
    };
    Ok(RepairModels { diagnose, patch })
}

fn require_pipe(id: &str) -> Result<String, &'static str> {
    let trimmed = id.trim();
    if is_pipe_model(trimmed) {
        Ok(trimmed.to_string())
    } else {
        Err(PIPE_MODEL_ERROR)
    }
}

pub fn is_pipe_model(id: &str) -> bool {
    let rest = id.strip_prefix("pipe/").unwrap_or("");
    !rest.is_empty() && !rest.chars().any(char::is_whitespace)
}

/// Read-modify-write `[autoharnessfix].enabled` on the machine file only.
/// Never `./hi.toml` — a repo must not opt the machine into rewriting Hi.
pub fn set_machine_enabled(enabled: bool) -> anyhow::Result<std::path::PathBuf> {
    let path = paths::default_config_path()
        .ok_or_else(|| anyhow::anyhow!("could not determine ~/.config/hi/config.toml"))?;
    let mut table = if path.exists() {
        let text = std::fs::read_to_string(&path)?;
        text.parse::<toml::Table>().map_err(|err| {
            anyhow::anyhow!(
                "machine config {} is not valid TOML ({err}); not rewriting",
                path.display()
            )
        })?
    } else {
        toml::Table::new()
    };
    match table.get_mut("autoharnessfix") {
        Some(toml::Value::Table(existing)) => {
            existing.insert("enabled".into(), toml::Value::Boolean(enabled));
        }
        Some(_) => anyhow::bail!("[autoharnessfix] must be a table"),
        None => {
            let mut section = toml::Table::new();
            section.insert("enabled".into(), toml::Value::Boolean(enabled));
            table.insert("autoharnessfix".into(), toml::Value::Table(section));
        }
    }
    let body = toml::to_string_pretty(&table)?;
    if let Some(parent) = path.parent() {
        crate::fsutil::mkdir_0700(parent)?;
    }
    crate::fsutil::write_atomic_0600(&path, body.as_bytes())?;
    Ok(path)
}

pub fn instance_token() -> String {
    let pid = std::process::id() as u128;
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mixed = pid.wrapping_shl(64) ^ ns ^ 0xA5A5_5A5A_C3C3_3C3C_F00D_D00F_BEEF_FEED;
    format!("{mixed:032x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_pipe_ids() {
        let models = resolve_repair_models(None).unwrap();
        assert!(is_pipe_model(&models.diagnose));
        assert!(is_pipe_model(&models.patch));
        assert_eq!(models.diagnose, DEFAULT_DIAGNOSE_MODEL);
        assert_eq!(models.patch, DEFAULT_PATCH_MODEL);
    }

    #[test]
    fn empty_or_non_pipe_fails_closed() {
        let empty = MachineSection {
            diagnose_model: Some(String::new()),
            ..MachineSection::default()
        };
        assert_eq!(
            resolve_repair_models(Some(&empty)).unwrap_err(),
            PIPE_MODEL_ERROR
        );
        let openai = MachineSection {
            patch_model: Some("openai/gpt-4".into()),
            ..MachineSection::default()
        };
        assert_eq!(
            resolve_repair_models(Some(&openai)).unwrap_err(),
            PIPE_MODEL_ERROR
        );
        let bare = MachineSection {
            diagnose_model: Some("pipe/".into()),
            ..MachineSection::default()
        };
        assert!(resolve_repair_models(Some(&bare)).is_err());
    }

    #[test]
    fn explicit_pipe_ids_win() {
        let section = MachineSection {
            diagnose_model: Some("pipe/glm-5.2-fast".into()),
            patch_model: Some("pipe/deepseek-v4-flash-0731".into()),
            ..MachineSection::default()
        };
        let models = resolve_repair_models(Some(&section)).unwrap();
        assert_eq!(models.diagnose, "pipe/glm-5.2-fast");
        assert_eq!(models.patch, "pipe/deepseek-v4-flash-0731");
    }

    #[test]
    fn set_machine_enabled_does_not_touch_project_hi_toml() {
        let _lock = crate::ENV_LOCK.lock().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let config_home = tmp.path().join("config");
        let proj = tmp.path().join("proj");
        std::fs::create_dir_all(&proj).unwrap();
        let project_toml = proj.join("hi.toml");
        std::fs::write(
            &project_toml,
            "[autoharnessfix]\nenabled = false\ncheckout = \"/tmp/evil\"\n",
        )
        .unwrap();
        let previous_config = std::env::var_os("XDG_CONFIG_HOME");
        let previous_cwd = std::env::current_dir().ok();
        unsafe {
            std::env::set_var("XDG_CONFIG_HOME", &config_home);
        }
        std::env::set_current_dir(&proj).unwrap();
        let path = set_machine_enabled(true).unwrap();
        let machine = std::fs::read_to_string(&path).unwrap();
        let project = std::fs::read_to_string(&project_toml).unwrap();
        if let Some(cwd) = previous_cwd {
            let _ = std::env::set_current_dir(cwd);
        }
        unsafe {
            match previous_config {
                Some(value) => std::env::set_var("XDG_CONFIG_HOME", value),
                None => std::env::remove_var("XDG_CONFIG_HOME"),
            }
        }
        assert!(
            path.ends_with("hi/config.toml"),
            "must write default_config_path, got {}",
            path.display()
        );
        assert!(machine.contains("enabled = true"));
        assert!(
            project.contains("enabled = false"),
            "project hi.toml must stay untouched"
        );
        assert!(project.contains("/tmp/evil"));
    }

    #[test]
    fn set_machine_enabled_fails_closed_on_invalid_toml() {
        let _lock = crate::ENV_LOCK.lock().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let config_home = tmp.path().join("config");
        std::fs::create_dir_all(config_home.join("hi")).unwrap();
        let machine = config_home.join("hi/config.toml");
        std::fs::write(&machine, "this is not [toml\n").unwrap();
        let previous = std::env::var_os("XDG_CONFIG_HOME");
        unsafe {
            std::env::set_var("XDG_CONFIG_HOME", &config_home);
        }
        let err = set_machine_enabled(true);
        let after = std::fs::read_to_string(&machine).unwrap();
        unsafe {
            match previous {
                Some(value) => std::env::set_var("XDG_CONFIG_HOME", value),
                None => std::env::remove_var("XDG_CONFIG_HOME"),
            }
        }
        assert!(err.is_err(), "parse failure must not rewrite");
        assert_eq!(after, "this is not [toml\n");
    }

    #[test]
    fn set_machine_enabled_keeps_existing_keys() {
        let _lock = crate::ENV_LOCK.lock().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let config_home = tmp.path().join("config");
        std::fs::create_dir_all(config_home.join("hi")).unwrap();
        let machine = config_home.join("hi/config.toml");
        std::fs::write(&machine, "[profiles.pipenetwork]\napi_key = \"pk_keep\"\n").unwrap();
        let previous = std::env::var_os("XDG_CONFIG_HOME");
        unsafe {
            std::env::set_var("XDG_CONFIG_HOME", &config_home);
        }
        set_machine_enabled(true).unwrap();
        let after = std::fs::read_to_string(&machine).unwrap();
        unsafe {
            match previous {
                Some(value) => std::env::set_var("XDG_CONFIG_HOME", value),
                None => std::env::remove_var("XDG_CONFIG_HOME"),
            }
        }
        assert!(after.contains("pk_keep"), "{after}");
        assert!(after.contains("enabled = true"), "{after}");
    }
}
