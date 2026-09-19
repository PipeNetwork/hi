//! JSONL session persistence: one message per line, plus usage and checkpoints.

use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use hi_ai::{Message, Role, Usage};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanDrive {
    pub paused: bool,
    #[serde(default)]
    pub resume_on_user_input: bool,
    #[serde(default)]
    pub stall: u32,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum SessionMeta {
    Usage {
        input_tokens: u64,
        output_tokens: u64,
        #[serde(default)]
        cache_read_tokens: u64,
        #[serde(default)]
        cache_creation_tokens: u64,
        #[serde(default)]
        estimated: bool,
    },
    Checkpoints {
        refs: Vec<String>,
    },
    Verify {
        command: Option<String>,
    },
    Model {
        id: String,
    },
    Name {
        name: String,
    },
    Knobs {
        permission: u8,
        #[serde(default)]
        effort: Option<String>,
    },
    PendingTurn {
        turn_index: u32,
        started_unix_ms: u64,
        pre_checkpoint: Option<String>,
    },
    TurnClosed {
        turn_index: u32,
    },
    Plan {
        steps: Vec<hi_tools::PlanStep>,
    },
    PlanDrive {
        paused: bool,
        #[serde(default)]
        resume_on_user_input: bool,
        #[serde(default)]
        stall: u32,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PendingTurn {
    pub turn_index: u32,
    pub started_unix_ms: u64,
    pub pre_checkpoint: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct LoadedSession {
    pub messages: Vec<Message>,
    pub usage: Usage,
    pub checkpoints: Vec<String>,
    pub verify_command: Option<String>,
    pub model: Option<String>,
    pub name: Option<String>,
    pub permission: Option<u8>,
    pub effort: Option<String>,
    pub pending_turn: Option<PendingTurn>,
    pub plan: Vec<hi_tools::PlanStep>,
    pub plan_drive: PlanDrive,
}

pub struct JsonlSession {
    path: PathBuf,
    _lease: crate::session_lease::SessionLease,
}

impl JsonlSession {
    pub fn create(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating session dir {}", parent.display()))?;
        }
        let lease = crate::session_lease::SessionLease::acquire(&path)?;
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("creating session {}", path.display()))?;
        Ok(Self {
            path,
            _lease: lease,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn load(path: impl AsRef<Path>) -> Result<LoadedSession> {
        let file = fs::File::open(path.as_ref())
            .with_context(|| format!("opening session {}", path.as_ref().display()))?;
        let mut loaded = LoadedSession::default();
        let mut pending: Option<PendingTurn> = None;
        for line in BufReader::new(file).lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(meta) = serde_json::from_str::<SessionMeta>(line) {
                match meta {
                    SessionMeta::Usage {
                        input_tokens,
                        output_tokens,
                        cache_read_tokens,
                        cache_creation_tokens,
                        estimated,
                    } => {
                        loaded.usage = Usage {
                            input_tokens,
                            output_tokens,
                            cache_read_tokens,
                            cache_creation_tokens,
                            estimated,
                            ..Usage::default()
                        };
                    }
                    SessionMeta::Checkpoints { refs } => loaded.checkpoints = refs,
                    SessionMeta::Verify { command } => loaded.verify_command = command,
                    SessionMeta::Model { id } => loaded.model = Some(id),
                    SessionMeta::Name { name } => {
                        loaded.name = if name.is_empty() { None } else { Some(name) };
                    }
                    SessionMeta::Knobs { permission, effort } => {
                        loaded.permission = Some(permission);
                        loaded.effort = effort;
                    }
                    SessionMeta::PendingTurn {
                        turn_index,
                        started_unix_ms,
                        pre_checkpoint,
                    } => {
                        pending = Some(PendingTurn {
                            turn_index,
                            started_unix_ms,
                            pre_checkpoint,
                        });
                    }
                    SessionMeta::TurnClosed { turn_index } => {
                        if pending
                            .as_ref()
                            .is_some_and(|open| open.turn_index == turn_index)
                        {
                            pending = None;
                        }
                    }
                    SessionMeta::Plan { steps } => loaded.plan = steps,
                    SessionMeta::PlanDrive {
                        paused,
                        resume_on_user_input,
                        stall,
                    } => {
                        loaded.plan_drive = PlanDrive {
                            paused,
                            resume_on_user_input,
                            stall,
                        };
                    }
                }
                continue;
            }
            if let Ok(message) = serde_json::from_str::<Message>(line) {
                loaded.messages.push(message);
            }
        }
        loaded.pending_turn = pending;
        if loaded.plan.is_empty() {
            loaded.plan = crate::completion::plan_from_messages(&loaded.messages);
        }
        Ok(loaded)
    }

    pub fn record_messages(&mut self, messages: &[Message]) -> Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        for message in messages {
            serde_json::to_writer(&mut file, message)?;
            file.write_all(b"\n")?;
        }
        // Turn-start persist must survive a mid-turn crash.
        file.flush()?;
        file.sync_all()?;
        self.note_user_messages(messages);
        Ok(())
    }

    pub fn record_pending_turn(&mut self, pending: &PendingTurn) -> Result<()> {
        self.write_meta_sync(&SessionMeta::PendingTurn {
            turn_index: pending.turn_index,
            started_unix_ms: pending.started_unix_ms,
            pre_checkpoint: pending.pre_checkpoint.clone(),
        })?;
        crate::session_index::patch_index(&self.path, |index| {
            index.pending_turn = Some(pending.turn_index);
        });
        Ok(())
    }

    pub fn record_plan(&mut self, steps: &[hi_tools::PlanStep]) -> Result<()> {
        self.write_meta_sync(&SessionMeta::Plan {
            steps: steps.to_vec(),
        })?;
        let (done, total, active) = crate::session_index::plan_progress(steps);
        crate::session_index::patch_index(&self.path, |index| {
            index.plan_done = done;
            index.plan_total = total;
            index.plan_active = active;
        });
        Ok(())
    }

    pub fn record_plan_drive(&mut self, drive: &PlanDrive) -> Result<()> {
        self.write_meta_sync(&SessionMeta::PlanDrive {
            paused: drive.paused,
            resume_on_user_input: drive.resume_on_user_input,
            stall: drive.stall,
        })?;
        crate::session_index::patch_index(&self.path, |index| {
            index.drive_paused = drive.paused;
        });
        Ok(())
    }

    /// User line + `PendingTurn` in one append/fsync so success/failure is one unit.
    pub fn record_turn_start(&mut self, message: &Message, pending: &PendingTurn) -> Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        serde_json::to_writer(&mut file, message)?;
        file.write_all(b"\n")?;
        write_meta_to(
            &mut file,
            &SessionMeta::PendingTurn {
                turn_index: pending.turn_index,
                started_unix_ms: pending.started_unix_ms,
                pre_checkpoint: pending.pre_checkpoint.clone(),
            },
        )?;
        file.flush()?;
        file.sync_all()?;
        self.note_user_messages(std::slice::from_ref(message));
        crate::session_index::patch_index(&self.path, |index| {
            index.pending_turn = Some(pending.turn_index);
        });
        Ok(())
    }

    pub fn last_recorded_user_text(&self) -> Option<String> {
        let file = fs::File::open(&self.path).ok()?;
        let mut last = None;
        for line in BufReader::new(file).lines().map_while(Result::ok) {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(message) = serde_json::from_str::<Message>(line)
                && message.role == Role::User
            {
                last = Some(message.text().to_string());
            }
        }
        last
    }

    pub fn record_turn_closed(&mut self, turn_index: u32) -> Result<()> {
        self.write_meta_sync(&SessionMeta::TurnClosed { turn_index })?;
        crate::session_index::patch_index(&self.path, |index| {
            if index.pending_turn == Some(turn_index) {
                index.pending_turn = None;
            }
        });
        Ok(())
    }

    pub fn record_usage(&mut self, usage: Usage) -> Result<()> {
        self.write_meta(&SessionMeta::Usage {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            cache_read_tokens: usage.cache_read_tokens,
            cache_creation_tokens: usage.cache_creation_tokens,
            estimated: usage.estimated,
        })
    }

    pub fn record_checkpoints(&mut self, refs: &[String]) -> Result<()> {
        self.write_meta(&SessionMeta::Checkpoints {
            refs: refs.to_vec(),
        })
    }

    pub fn record_verify(&mut self, command: Option<&str>) -> Result<()> {
        self.write_meta(&SessionMeta::Verify {
            command: command.map(str::to_string),
        })
    }

    pub fn record_model(&mut self, id: &str) -> Result<()> {
        self.write_meta(&SessionMeta::Model { id: id.to_string() })
    }

    pub fn record_knobs(&mut self, permission: u8, effort: Option<&str>) -> Result<()> {
        self.write_meta(&SessionMeta::Knobs {
            permission,
            effort: effort.map(str::to_string),
        })
    }

    /// Replace the file with `state` so rewind/clear/retry cannot resurrect
    /// dropped messages on the next load.
    pub fn rewrite(&mut self, state: &LoadedSession) -> Result<()> {
        let mut tmp_os = self.path.as_os_str().to_os_string();
        tmp_os.push(".tmp");
        let tmp = PathBuf::from(tmp_os);
        let result = (|| {
            let mut file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&tmp)
                .with_context(|| format!("creating {}", tmp.display()))?;
            for message in &state.messages {
                serde_json::to_writer(&mut file, message)?;
                file.write_all(b"\n")?;
            }
            write_meta_to(
                &mut file,
                &SessionMeta::Usage {
                    input_tokens: state.usage.input_tokens,
                    output_tokens: state.usage.output_tokens,
                    cache_read_tokens: state.usage.cache_read_tokens,
                    cache_creation_tokens: state.usage.cache_creation_tokens,
                    estimated: state.usage.estimated,
                },
            )?;
            write_meta_to(
                &mut file,
                &SessionMeta::Checkpoints {
                    refs: state.checkpoints.clone(),
                },
            )?;
            write_meta_to(
                &mut file,
                &SessionMeta::Verify {
                    command: state.verify_command.clone(),
                },
            )?;
            if let Some(id) = &state.model {
                write_meta_to(&mut file, &SessionMeta::Model { id: id.clone() })?;
            }
            if let Some(name) = &state.name {
                write_meta_to(&mut file, &SessionMeta::Name { name: name.clone() })?;
            }
            if let Some(permission) = state.permission {
                write_meta_to(
                    &mut file,
                    &SessionMeta::Knobs {
                        permission,
                        effort: state.effort.clone(),
                    },
                )?;
            }
            if let Some(pending) = &state.pending_turn {
                write_meta_to(
                    &mut file,
                    &SessionMeta::PendingTurn {
                        turn_index: pending.turn_index,
                        started_unix_ms: pending.started_unix_ms,
                        pre_checkpoint: pending.pre_checkpoint.clone(),
                    },
                )?;
            }
            if !state.plan.is_empty() {
                write_meta_to(
                    &mut file,
                    &SessionMeta::Plan {
                        steps: state.plan.clone(),
                    },
                )?;
            }
            write_meta_to(
                &mut file,
                &SessionMeta::PlanDrive {
                    paused: state.plan_drive.paused,
                    resume_on_user_input: state.plan_drive.resume_on_user_input,
                    stall: state.plan_drive.stall,
                },
            )?;
            file.flush()?;
            file.sync_all()?;
            drop(file);
            fs::rename(&tmp, &self.path)
                .with_context(|| format!("replacing session {}", self.path.display()))?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&tmp);
        }
        if result.is_ok() {
            let index = crate::session_index::index_from_loaded(&self.path, state);
            let _ = crate::session_index::write_index(&self.path, &index);
        }
        result
    }

    fn note_user_messages(&self, messages: &[Message]) {
        for message in messages {
            if message.role != Role::User {
                continue;
            }
            let text = message.text();
            if crate::session_index::is_harness_injection(&text) {
                continue;
            }
            let preview: String = text
                .trim()
                .lines()
                .next()
                .unwrap_or("")
                .chars()
                .take(96)
                .collect();
            if preview.is_empty() {
                continue;
            }
            crate::session_index::patch_index(&self.path, |index| {
                index.last_user.clone_from(&preview);
                if index.title.is_empty() {
                    index.title = preview.clone();
                }
            });
        }
    }

    fn write_meta(&mut self, meta: &SessionMeta) -> Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        write_meta_to(&mut file, meta)
    }

    fn write_meta_sync(&mut self, meta: &SessionMeta) -> Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        write_meta_to(&mut file, meta)?;
        file.flush()?;
        file.sync_all()?;
        Ok(())
    }
}

fn write_meta_to(file: &mut impl Write, meta: &SessionMeta) -> Result<()> {
    serde_json::to_writer(&mut *file, meta)?;
    file.write_all(b"\n")?;
    Ok(())
}

#[derive(Clone, Debug)]
pub struct UserTurn {
    pub n: usize,
    pub message_index: usize,
    pub preview: String,
}

pub fn list_user_turns(messages: &[Message]) -> Vec<UserTurn> {
    let mut out = Vec::new();
    for (message_index, msg) in messages.iter().enumerate() {
        if msg.role != Role::User {
            continue;
        }
        let text = msg.text();
        if text.trim().is_empty() {
            continue;
        }
        let preview: String = text.trim().chars().take(96).collect();
        out.push(UserTurn {
            n: out.len() + 1,
            message_index,
            preview,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_ai::Message;

    #[test]
    fn round_trips_messages_and_checkpoints() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.jsonl");
        let mut session = JsonlSession::create(&path).unwrap();
        session
            .record_messages(&[Message::user("hello"), Message::assistant(vec![])])
            .unwrap();
        session.record_checkpoints(&["abc".into()]).unwrap();
        let loaded = JsonlSession::load(&path).unwrap();
        assert_eq!(loaded.messages.len(), 2);
        assert_eq!(loaded.checkpoints, ["abc"]);
    }

    #[test]
    fn rewrite_drops_cleared_messages() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.jsonl");
        let mut session = JsonlSession::create(&path).unwrap();
        session
            .record_messages(&[Message::user("old"), Message::assistant(vec![])])
            .unwrap();
        session
            .rewrite(&LoadedSession {
                messages: vec![Message::user("new")],
                model: Some("pipe/kept".into()),
                ..LoadedSession::default()
            })
            .unwrap();
        let loaded = JsonlSession::load(&path).unwrap();
        assert_eq!(loaded.messages.len(), 1);
        assert_eq!(loaded.messages[0].text(), "new");
        assert_eq!(loaded.model.as_deref(), Some("pipe/kept"));
    }

    #[test]
    fn load_restores_unmatched_pending_turn() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.jsonl");
        let mut session = JsonlSession::create(&path).unwrap();
        session
            .record_messages(&[Message::user("in flight")])
            .unwrap();
        session
            .record_pending_turn(&PendingTurn {
                turn_index: 3,
                started_unix_ms: 9,
                pre_checkpoint: Some("pre".into()),
            })
            .unwrap();
        let loaded = JsonlSession::load(&path).unwrap();
        let pending = loaded.pending_turn.expect("unmatched pending");
        assert_eq!(pending.turn_index, 3);
        assert_eq!(pending.pre_checkpoint.as_deref(), Some("pre"));
        session.record_turn_closed(3).unwrap();
        let loaded = JsonlSession::load(&path).unwrap();
        assert!(loaded.pending_turn.is_none());
    }

    #[test]
    fn rewrite_round_trips_pending_turn_without_turn_closed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.jsonl");
        let mut session = JsonlSession::create(&path).unwrap();
        session.record_messages(&[Message::user("old")]).unwrap();
        session.record_turn_closed(1).unwrap();
        session
            .rewrite(&LoadedSession {
                messages: vec![Message::user("in flight")],
                pending_turn: Some(PendingTurn {
                    turn_index: 2,
                    started_unix_ms: 11,
                    pre_checkpoint: None,
                }),
                ..LoadedSession::default()
            })
            .unwrap();
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(raw.contains("pending_turn"));
        assert!(
            !raw.contains("turn_closed"),
            "rewrite must not emit TurnClosed for an unmatched pending turn"
        );
        let loaded = JsonlSession::load(&path).unwrap();
        assert_eq!(loaded.pending_turn.unwrap().turn_index, 2);
    }

    #[test]
    fn record_turn_start_writes_user_and_pending_together() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.jsonl");
        let mut session = JsonlSession::create(&path).unwrap();
        session
            .record_turn_start(
                &Message::user("in flight"),
                &PendingTurn {
                    turn_index: 1,
                    started_unix_ms: 7,
                    pre_checkpoint: None,
                },
            )
            .unwrap();
        assert_eq!(
            session.last_recorded_user_text().as_deref(),
            Some("in flight")
        );
        let loaded = JsonlSession::load(&path).unwrap();
        assert_eq!(loaded.pending_turn.unwrap().turn_index, 1);
    }

    #[test]
    fn rewrite_and_load_restore_an_open_plan() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.jsonl");
        let mut session = JsonlSession::create(&path).unwrap();
        let steps = vec![
            hi_tools::PlanStep {
                title: "Add /metrics".into(),
                status: hi_tools::PlanStatus::Active,
            },
            hi_tools::PlanStep {
                title: "Run tests".into(),
                status: hi_tools::PlanStatus::Pending,
            },
        ];
        session
            .rewrite(&LoadedSession {
                messages: vec![Message::user("do all of that")],
                plan: steps.clone(),
                ..LoadedSession::default()
            })
            .unwrap();
        let loaded = JsonlSession::load(&path).unwrap();
        assert_eq!(loaded.plan, steps);
        assert!(!hi_tools::PlanStep::all_complete(&loaded.plan));
    }

    #[test]
    fn load_rehydrates_plan_from_update_plan_tool_call() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.jsonl");
        let mut session = JsonlSession::create(&path).unwrap();
        let arguments = serde_json::json!({
            "steps": [
                {"title":"Add /metrics","status":"active"},
                {"title":"Run tests","status":"pending"}
            ]
        })
        .to_string();
        session
            .record_messages(&[Message::assistant(vec![hi_ai::Content::ToolCall {
                id: "p1".into(),
                name: "update_plan".into(),
                arguments,
            }])])
            .unwrap();
        let loaded = JsonlSession::load(&path).unwrap();
        assert_eq!(loaded.plan.len(), 2);
        assert_eq!(loaded.plan[0].title, "Add /metrics");
        assert!(!hi_tools::PlanStep::all_complete(&loaded.plan));
    }

    #[test]
    fn record_and_load_plan_and_plan_drive() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.jsonl");
        let mut session = JsonlSession::create(&path).unwrap();
        let steps = vec![hi_tools::PlanStep {
            title: "Add /metrics".into(),
            status: hi_tools::PlanStatus::Active,
        }];
        session.record_plan(&steps).unwrap();
        session
            .record_plan_drive(&PlanDrive {
                paused: true,
                resume_on_user_input: true,
                stall: 2,
            })
            .unwrap();
        let loaded = JsonlSession::load(&path).unwrap();
        assert_eq!(loaded.plan, steps);
        assert!(loaded.plan_drive.paused);
        assert!(loaded.plan_drive.resume_on_user_input);
        assert_eq!(loaded.plan_drive.stall, 2);
    }

    #[test]
    fn load_keeps_plan_from_messages_when_meta_missing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.jsonl");
        let mut session = JsonlSession::create(&path).unwrap();
        let arguments = serde_json::json!({
            "steps": [{"title":"from tool","status":"pending"}]
        })
        .to_string();
        session
            .record_messages(&[Message::assistant(vec![hi_ai::Content::ToolCall {
                id: "p1".into(),
                name: "update_plan".into(),
                arguments,
            }])])
            .unwrap();
        let loaded = JsonlSession::load(&path).unwrap();
        assert_eq!(loaded.plan[0].title, "from tool");
        assert!(!loaded.plan_drive.paused);
    }

    #[test]
    fn smoke_seed_plan_drive_json_loads_paused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.jsonl");
        std::fs::write(
            &path,
            concat!(
                r#"{"type":"plan","steps":[{"title":"Durable pending smoke step","status":"Pending"}]}"#,
                "\n",
                r#"{"type":"plan_drive","paused":true,"resume_on_user_input":false,"stall":1}"#,
                "\n",
            ),
        )
        .unwrap();
        let loaded = JsonlSession::load(&path).unwrap();
        assert_eq!(loaded.plan[0].title, "Durable pending smoke step");
        assert!(loaded.plan_drive.paused);
        assert_eq!(loaded.plan_drive.stall, 1);
    }
}
