//! JSONL session persistence: one message per line, plus usage and checkpoints.

use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use hi_ai::{Message, Role, Usage};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SessionMeta {
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
}

pub struct JsonlSession {
    path: PathBuf,
}

impl JsonlSession {
    pub fn create(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating session dir {}", parent.display()))?;
        }
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("creating session {}", path.display()))?;
        Ok(Self { path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn load(path: impl AsRef<Path>) -> Result<LoadedSession> {
        let file = fs::File::open(path.as_ref())
            .with_context(|| format!("opening session {}", path.as_ref().display()))?;
        let mut loaded = LoadedSession::default();
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
                }
                continue;
            }
            if let Ok(message) = serde_json::from_str::<Message>(line) {
                loaded.messages.push(message);
            }
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
        result
    }

    fn write_meta(&mut self, meta: &SessionMeta) -> Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        write_meta_to(&mut file, meta)
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
}
