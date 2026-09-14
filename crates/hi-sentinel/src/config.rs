//! Supervisor and monitor knobs. Machine `[autoharnessfix]` only — never `--config PATH`.

use std::ffi::OsString;
use std::path::PathBuf;
use std::time::Duration;

use serde::Deserialize;

use crate::paths;

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
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct MachineSection {
    #[serde(default)]
    #[allow(dead_code)]
    pub enabled: bool,
    #[serde(default)]
    #[allow(dead_code)]
    pub apply: bool,
    pub checkout: Option<PathBuf>,
    pub stall_progress_secs: Option<u64>,
    pub liveness_secs: Option<u64>,
    pub live_pgid_report_secs: Option<u64>,
    #[allow(dead_code)]
    pub max_repairs_per_session: Option<u32>,
    pub incident_retention_days: Option<u64>,
}

#[derive(Default, Deserialize)]
struct MachineFile {
    #[serde(default)]
    autoharnessfix: Option<MachineSection>,
}

pub fn peek_machine() -> Option<MachineSection> {
    let path = paths::default_config_path()?;
    let text = std::fs::read_to_string(path).ok()?;
    toml::from_str::<MachineFile>(&text)
        .ok()
        .and_then(|file| file.autoharnessfix)
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
