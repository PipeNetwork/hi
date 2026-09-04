//! Typed, versioned harness overrides carried by an append-only session.

use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::{BufRead, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use hi_workspace::{SettingLayer, SettingSource, SettingValue, standard_harness_settings};
use serde::{Deserialize, Serialize};

pub(crate) const RECORD_TYPE: &str = "harness_settings";
const SCHEMA_VERSION: u16 = 1;
const MAX_RECORD_BYTES: usize = 64 * 1024;

#[derive(Deserialize)]
struct RecordHeader {
    #[serde(default, rename = "type")]
    record_type: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct HarnessSettingsRecord {
    #[serde(rename = "type")]
    record_type: String,
    schema_version: u16,
    values: BTreeMap<String, SettingValue>,
}

pub(crate) fn empty_layer() -> SettingLayer {
    SettingLayer {
        source: SettingSource::Session,
        values: BTreeMap::new(),
    }
}

/// Decode a local or remote record. Non-harness records return `None`; a
/// recognized but malformed/future record fails closed instead of silently
/// reverting to lower-precedence settings.
pub(crate) fn parse_record(payload: &str) -> Result<Option<SettingLayer>> {
    let header = match serde_json::from_str::<RecordHeader>(payload) {
        Ok(header) => header,
        Err(_) => return Ok(None),
    };
    if header.record_type.as_deref() != Some(RECORD_TYPE) {
        return Ok(None);
    }
    if payload.len() > MAX_RECORD_BYTES {
        bail!("session harness settings record exceeds {MAX_RECORD_BYTES} bytes");
    }
    let record: HarnessSettingsRecord =
        serde_json::from_str(payload).context("parsing persisted session harness settings")?;
    if record.schema_version != SCHEMA_VERSION {
        bail!(
            "unsupported session harness settings schema {}; this hi supports {}",
            record.schema_version,
            SCHEMA_VERSION
        );
    }
    if record.values.len() > standard_harness_settings().specs().count() {
        bail!("session harness settings record contains too many entries");
    }
    let layer = SettingLayer {
        source: SettingSource::Session,
        values: record.values,
    };
    standard_harness_settings()
        .validate_layer(&layer)
        .context("validating persisted session harness settings")?;
    Ok(Some(layer))
}

/// Read the last complete settings record from a bounded snapshot of a JSONL
/// session. Missing files and sessions without a record have an empty layer.
pub(crate) fn load(path: &Path) -> Result<SettingLayer> {
    let mut layer = empty_layer();
    let reader = match crate::session::session_snapshot_reader(path) {
        Ok(reader) => reader,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(layer),
        Err(error) => return Err(error).with_context(|| format!("opening {}", path.display())),
    };
    for line in reader.lines() {
        let line = line.with_context(|| format!("reading {}", path.display()))?;
        if let Some(next) = parse_record(&line)
            .with_context(|| format!("reading harness settings from {}", path.display()))?
        {
            layer = next;
        }
    }
    Ok(layer)
}

pub(crate) fn encode(layer: &SettingLayer) -> Result<String> {
    if layer.source != SettingSource::Session {
        bail!("persisted harness settings must use the session source");
    }
    standard_harness_settings()
        .validate_layer(layer)
        .context("validating session harness settings before persistence")?;
    let encoded = serde_json::to_string(&HarnessSettingsRecord {
        record_type: RECORD_TYPE.to_string(),
        schema_version: SCHEMA_VERSION,
        values: layer.values.clone(),
    })?;
    if encoded.len() > MAX_RECORD_BYTES {
        bail!("session harness settings record exceeds {MAX_RECORD_BYTES} bytes");
    }
    Ok(encoded)
}

/// Append one complete last-write-wins session layer with a single write.
pub(crate) fn append(path: &Path, layer: &SettingLayer) -> Result<()> {
    let mut encoded = encode(layer)?;
    encoded.push('\n');
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    file.write_all(encoded.as_bytes())
        .with_context(|| format!("appending harness settings to {}", path.display()))
}

pub(crate) fn ensure_agent_compatible(agent: &hi_agent::Agent, layer: &SettingLayer) -> Result<()> {
    let active = agent
        .harness_session_layer()
        .map(|active| &active.values)
        .cloned()
        .unwrap_or_default();
    if layer.values != active {
        bail!(
            "the selected session has different harness settings; restart it with `hi --resume <SESSION_ID>`"
        );
    }
    Ok(())
}

/// Apply a loaded session only when its process-wide harness contract matches.
pub(crate) fn apply_loaded_session(
    agent: &mut hi_agent::Agent,
    loaded: crate::session::LoadedSession,
) -> Result<()> {
    ensure_agent_compatible(agent, &loaded.harness_settings)?;
    let crate::session::LoadedSession {
        messages,
        usage,
        checkpoint_refs,
        goal,
        decisions,
        plan,
        plan_drive_paused,
        plan_drive_resume_on_user_input,
        plan_approval_parked,
        plan_drive_stall,
        goal_drive_stall,
        plan_drive_evidence,
        goal_drive_evidence,
        ..
    } = loaded;
    agent.apply_loaded_session(messages, usage, checkpoint_refs, goal, decisions, plan)?;
    agent.restore_plan_drive_with_policy(
        plan_drive_paused,
        plan_drive_resume_on_user_input,
        plan_drive_stall,
        plan_drive_evidence,
    );
    agent.restore_plan_approval_parked(plan_approval_parked);
    agent.restore_goal_drive(goal_drive_stall, goal_drive_evidence);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer(max_active: i64) -> SettingLayer {
        SettingLayer {
            source: SettingSource::Session,
            values: BTreeMap::from([(
                hi_workspace::JOB_MAX_ACTIVE.to_string(),
                SettingValue::Integer(max_active),
            )]),
        }
    }

    #[test]
    fn append_and_load_are_last_write_wins() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("session.jsonl");
        std::fs::write(&path, "{\"role\":\"user\",\"content\":[]}\n").unwrap();
        append(&path, &layer(7)).unwrap();
        append(&path, &layer(3)).unwrap();
        assert_eq!(load(&path).unwrap(), layer(3));
    }

    #[test]
    fn recognized_future_or_invalid_records_fail_closed() {
        let future = r#"{"type":"harness_settings","schema_version":2,"values":{}}"#;
        assert!(
            parse_record(future)
                .unwrap_err()
                .to_string()
                .contains("schema 2")
        );
        let unknown = r#"{"type":"harness_settings","schema_version":1,"values":{"not.registered":{"kind":"boolean","value":true}}}"#;
        assert!(
            parse_record(unknown)
                .unwrap_err()
                .to_string()
                .contains("validating")
        );
    }

    #[test]
    fn ordinary_session_records_are_ignored() {
        assert!(
            parse_record(r#"{"type":"usage","input_tokens":1}"#)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn live_agent_rejects_a_different_session_contract() {
        let temp = tempfile::tempdir().unwrap();
        let active = layer(7);
        let config = hi_agent::AgentConfig {
            paths: hi_agent::AgentPaths {
                workspace_root: temp.path().to_path_buf(),
                state_root: temp.path().join("state"),
            },
            harness_session: Some(active.clone()),
            ..hi_agent::AgentConfig::default()
        };
        let provider = std::sync::Arc::new(hi_ai::OpenAiProvider::new(
            "http://127.0.0.1:1/v1".into(),
            "unused".into(),
        ));
        let agent = hi_agent::Agent::new(provider, config).unwrap();
        ensure_agent_compatible(&agent, &active).unwrap();
        assert!(ensure_agent_compatible(&agent, &layer(3)).is_err());
    }
}
