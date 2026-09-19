//! Machine-only `[autoharnessfix]`. Project `hi.toml` cannot enable Sentinel.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::Config;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct AutoHarnessFixSection {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub apply: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkout: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnose_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub patch_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stall_progress_secs: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub liveness_secs: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub live_pgid_report_secs: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_repairs_per_session: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_attempts_per_incident: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_modifications_per_hour: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub incident_retention_days: Option<u64>,
}

#[derive(Default, Deserialize)]
struct MachinePeek {
    #[serde(default)]
    autoharnessfix: Option<AutoHarnessFixSection>,
    #[serde(default)]
    rsi: Option<RsiEnabledPeek>,
}

#[derive(Default, Deserialize)]
struct RsiEnabledPeek {
    enabled: Option<bool>,
}

pub fn peek_machine_autoharnessfix() -> AutoHarnessFixSection {
    peek_machine()
        .and_then(|m| m.autoharnessfix)
        .unwrap_or_default()
}

pub fn peek_machine_rsi_enabled() -> bool {
    peek_machine()
        .and_then(|m| m.rsi)
        .and_then(|rsi| rsi.enabled)
        .unwrap_or(false)
}

fn peek_machine() -> Option<MachinePeek> {
    let path = super::default_config_path()?;
    let text = std::fs::read_to_string(path).ok()?;
    toml::from_str(&text).ok()
}

/// Project files must not opt the machine into rewriting Hi.
pub(crate) fn drop_project_overlay(overlay: &mut Config) {
    if overlay.autoharnessfix.take().is_some() {
        tracing::debug!("dropping project [autoharnessfix]; Sentinel is machine-scoped");
    }
    if overlay.typesafe.take().is_some() {
        tracing::debug!("dropping project [typesafe]; next-action gate is machine-scoped");
    }
}
