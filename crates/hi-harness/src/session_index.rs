//! Per-session `<id>.index.json` so listing does not parse huge JSONL files.

use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use hi_ai::{Message, Role};
use hi_tools::{PlanStatus, PlanStep};
use serde::{Deserialize, Serialize};

use crate::session::SessionMeta;

/// Sidecar written beside a session JSONL.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionIndex {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub last_user: String,
    pub pending_turn: Option<u32>,
    #[serde(default)]
    pub plan_done: u32,
    #[serde(default)]
    pub plan_total: u32,
    pub plan_active: Option<String>,
    #[serde(default)]
    pub drive_paused: bool,
    #[serde(default)]
    pub mtime_unix_ms: u64,
}

impl SessionIndex {
    pub fn plan_is_open(&self) -> bool {
        self.plan_total > 0 && self.plan_done < self.plan_total
    }

    pub fn needs_attention(&self) -> bool {
        self.pending_turn.is_some() || self.plan_is_open()
    }

    pub fn plan_flag(&self) -> Option<String> {
        if self.plan_total == 0 {
            return None;
        }
        let mut flag = format!("PLAN {}/{}", self.plan_done, self.plan_total);
        if let Some(active) = &self.plan_active
            && !active.is_empty()
        {
            flag.push_str(" · active: ");
            flag.push_str(active);
        }
        Some(flag)
    }
}

/// `<id>.index.json` next to `<id>.jsonl`.
pub fn index_path(jsonl: &Path) -> PathBuf {
    let stem = jsonl
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("session");
    jsonl.with_file_name(format!("{stem}.index.json"))
}

pub fn session_id_from_path(jsonl: &Path) -> String {
    jsonl
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("session")
        .to_string()
}

/// Load the sidecar, or backfill from a meta-only JSONL scan when missing/stale.
pub fn load_or_refresh_index(jsonl: &Path) -> SessionIndex {
    let path = index_path(jsonl);
    if index_is_fresh(jsonl, &path)
        && let Ok(bytes) = fs::read(&path)
        && let Ok(index) = serde_json::from_slice::<SessionIndex>(&bytes)
    {
        return index;
    }
    let index = scan_index(jsonl);
    let _ = write_index(jsonl, &index);
    index
}

pub fn write_index(jsonl: &Path, index: &SessionIndex) -> std::io::Result<()> {
    let path = index_path(jsonl);
    fs::write(path, serde_json::to_vec_pretty(index)?)
}

pub fn patch_index(jsonl: &Path, patch: impl FnOnce(&mut SessionIndex)) {
    let mut index = load_index_loose(jsonl);
    if index.id.is_empty() {
        index.id = session_id_from_path(jsonl);
    }
    patch(&mut index);
    index.mtime_unix_ms = now_ms();
    let _ = write_index(jsonl, &index);
}

pub fn index_from_loaded(jsonl: &Path, state: &crate::LoadedSession) -> SessionIndex {
    let (plan_done, plan_total, plan_active) = plan_progress(&state.plan);
    let (title, last_user) = title_from_messages(&state.messages);
    SessionIndex {
        id: session_id_from_path(jsonl),
        title,
        last_user,
        pending_turn: state.pending_turn.as_ref().map(|p| p.turn_index),
        plan_done,
        plan_total,
        plan_active,
        drive_paused: state.plan_drive.paused,
        mtime_unix_ms: now_ms(),
    }
}

pub fn plan_progress(steps: &[PlanStep]) -> (u32, u32, Option<String>) {
    let plan_total = steps.len() as u32;
    let plan_done = steps
        .iter()
        .filter(|step| step.status == PlanStatus::Done)
        .count() as u32;
    let plan_active = steps
        .iter()
        .find(|step| step.status == PlanStatus::Active)
        .or_else(|| steps.iter().find(|step| step.status == PlanStatus::Pending))
        .map(|step| step.title.clone());
    (plan_done, plan_total, plan_active)
}

/// Harness-injected user lines are not session titles.
pub fn is_harness_injection(text: &str) -> bool {
    let text = text.trim_start();
    text.starts_with("[hi:")
}

fn load_index_loose(jsonl: &Path) -> SessionIndex {
    let path = index_path(jsonl);
    fs::read(&path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_else(|| SessionIndex {
            id: session_id_from_path(jsonl),
            ..SessionIndex::default()
        })
}

fn index_is_fresh(jsonl: &Path, index: &Path) -> bool {
    let Ok(jsonl_meta) = fs::metadata(jsonl) else {
        return false;
    };
    let Ok(index_meta) = fs::metadata(index) else {
        return false;
    };
    let jsonl_mtime = jsonl_meta.modified().unwrap_or(UNIX_EPOCH);
    let index_mtime = index_meta.modified().unwrap_or(UNIX_EPOCH);
    index_mtime >= jsonl_mtime
}

fn scan_index(jsonl: &Path) -> SessionIndex {
    let mut index = SessionIndex {
        id: session_id_from_path(jsonl),
        mtime_unix_ms: now_ms(),
        ..SessionIndex::default()
    };
    let Ok(file) = File::open(jsonl) else {
        return index;
    };
    let mut pending: Option<u32> = None;
    let mut saw_plan_meta = false;
    let mut fallback_plan: Option<Vec<PlanStep>> = None;
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(meta) = serde_json::from_str::<SessionMeta>(line) {
            match meta {
                SessionMeta::PendingTurn { turn_index, .. } => pending = Some(turn_index),
                SessionMeta::TurnClosed { turn_index } => {
                    if pending == Some(turn_index) {
                        pending = None;
                    }
                }
                SessionMeta::Plan { steps } => {
                    saw_plan_meta = true;
                    let (done, total, active) = plan_progress(&steps);
                    index.plan_done = done;
                    index.plan_total = total;
                    index.plan_active = active;
                }
                SessionMeta::PlanDrive { paused, .. } => index.drive_paused = paused,
                _ => {}
            }
            continue;
        }
        if let Ok(message) = serde_json::from_str::<Message>(line) {
            if message.role == Role::User {
                let text = message.text();
                if !is_harness_injection(&text) {
                    let preview = user_preview(&text);
                    if !preview.is_empty() {
                        index.last_user.clone_from(&preview);
                        if index.title.is_empty() {
                            index.title = preview;
                        }
                    }
                }
            } else if !saw_plan_meta {
                let steps = crate::completion::plan_from_messages(std::slice::from_ref(&message));
                if !steps.is_empty() {
                    fallback_plan = Some(steps);
                }
            }
        }
    }
    index.pending_turn = pending;
    if !saw_plan_meta && let Some(steps) = fallback_plan {
        let (done, total, active) = plan_progress(&steps);
        index.plan_done = done;
        index.plan_total = total;
        index.plan_active = active;
    }
    index
}

fn title_from_messages(messages: &[Message]) -> (String, String) {
    let mut title = String::new();
    let mut last_user = String::new();
    for message in messages {
        if message.role != Role::User {
            continue;
        }
        let text = message.text();
        if is_harness_injection(&text) {
            continue;
        }
        let preview = user_preview(&text);
        if preview.is_empty() {
            continue;
        }
        last_user.clone_from(&preview);
        if title.is_empty() {
            title = preview;
        }
    }
    (title, last_user)
}

fn user_preview(text: &str) -> String {
    text.trim()
        .lines()
        .next()
        .unwrap_or("")
        .chars()
        .take(96)
        .collect()
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{JsonlSession, LoadedSession, PendingTurn};
    use hi_ai::Message;

    #[test]
    fn index_round_trip_writes_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.jsonl");
        let mut session = JsonlSession::create(&path).unwrap();
        session
            .record_turn_start(
                &Message::user("ship pagination"),
                &PendingTurn {
                    turn_index: 1,
                    started_unix_ms: 1,
                    pre_checkpoint: None,
                },
            )
            .unwrap();
        let index = load_or_refresh_index(&path);
        assert_eq!(index.title, "ship pagination");
        assert_eq!(index.pending_turn, Some(1));
        assert!(index_path(&path).is_file());
    }

    #[test]
    fn title_skips_harness_nudges() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.jsonl");
        let mut session = JsonlSession::create(&path).unwrap();
        session
            .record_messages(&[
                Message::user("[hi:context — session state, not instructions] dump"),
                Message::user("[hi:nudge] keep going"),
                Message::user("real task"),
            ])
            .unwrap();
        let index = scan_index(&path);
        assert_eq!(index.title, "real task");
        assert_eq!(index.last_user, "real task");
    }

    #[test]
    fn rewrite_refreshes_plan_progress() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.jsonl");
        let mut session = JsonlSession::create(&path).unwrap();
        let steps = vec![
            PlanStep {
                title: "Forward HISTORY pagination".into(),
                status: PlanStatus::Done,
            },
            PlanStep {
                title: "Run tests".into(),
                status: PlanStatus::Pending,
            },
        ];
        session
            .rewrite(&LoadedSession {
                messages: vec![Message::user("do it")],
                plan: steps,
                ..LoadedSession::default()
            })
            .unwrap();
        let index = load_or_refresh_index(&path);
        assert_eq!(index.plan_done, 1);
        assert_eq!(index.plan_total, 2);
        assert_eq!(index.plan_active.as_deref(), Some("Run tests"));
        assert!(index.plan_is_open());
    }
}
