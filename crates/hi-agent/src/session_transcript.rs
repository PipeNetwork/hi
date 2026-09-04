//! Durable transcript-block identities and lifecycle transitions.
//!
//! Presentation clients may still derive positional entries from legacy
//! messages, but v2 projections use these records when stable identity and
//! lifecycle settlement matter. A block ID is never reused, and a settled
//! block cannot be updated or settled a second time.

use std::collections::BTreeSet;
use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

const MAX_TRANSCRIPT_BLOCK_ID_BYTES: usize = 200;

/// Stable, session-scoped identity for one durable transcript block.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TranscriptBlockId(String);

impl TranscriptBlockId {
    pub fn new(value: impl Into<String>) -> Result<Self, TranscriptBlockIdError> {
        let value = value.into();
        validate_id(&value)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for TranscriptBlockId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl Serialize for TranscriptBlockId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for TranscriptBlockId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TranscriptBlockIdError {
    Empty,
    TooLong { bytes: usize, maximum: usize },
    InvalidCharacter,
}

impl fmt::Display for TranscriptBlockIdError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(formatter, "transcript block ID cannot be empty"),
            Self::TooLong { bytes, maximum } => write!(
                formatter,
                "transcript block ID is {bytes} bytes; maximum is {maximum}"
            ),
            Self::InvalidCharacter => write!(
                formatter,
                "transcript block ID may contain only ASCII letters, digits, '-', '_', '.', and ':'"
            ),
        }
    }
}

impl std::error::Error for TranscriptBlockIdError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptBlockKind {
    UserPrompt,
    Assistant,
    Reasoning,
    Tool,
    Workflow,
    Activity,
    Notice,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptBlockTerminal {
    Completed,
    Failed,
    Cancelled,
    Superseded,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum TranscriptBlockLifecycle {
    Open,
    Settled {
        terminal: TranscriptBlockTerminal,
        sequence: u64,
    },
}

impl TranscriptBlockLifecycle {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Settled { .. })
    }

    pub fn terminal(&self) -> Option<TranscriptBlockTerminal> {
        match self {
            Self::Open => None,
            Self::Settled { terminal, .. } => Some(*terminal),
        }
    }
}

/// Current durable projection of one transcript block.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscriptBlock {
    pub id: TranscriptBlockId,
    pub kind: TranscriptBlockKind,
    #[serde(default)]
    pub content: String,
    pub lifecycle: TranscriptBlockLifecycle,
    pub opened_sequence: u64,
    pub last_transition_sequence: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TranscriptBlockTransitionError {
    DuplicateId { id: TranscriptBlockId },
    UnknownId { id: TranscriptBlockId },
    AlreadySettled { id: TranscriptBlockId },
    UnsettledAtTurnEnd { ids: Vec<TranscriptBlockId> },
    InvalidSnapshot { id: TranscriptBlockId },
}

impl fmt::Display for TranscriptBlockTransitionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateId { id } => {
                write!(formatter, "transcript block '{id}' already exists")
            }
            Self::UnknownId { id } => write!(formatter, "unknown transcript block '{id}'"),
            Self::AlreadySettled { id } => {
                write!(formatter, "transcript block '{id}' is already settled")
            }
            Self::UnsettledAtTurnEnd { ids } => write!(
                formatter,
                "turn cannot settle while transcript blocks remain open: {}",
                ids.iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::InvalidSnapshot { id } => {
                write!(
                    formatter,
                    "invalid lifecycle state for transcript block '{id}'"
                )
            }
        }
    }
}

impl std::error::Error for TranscriptBlockTransitionError {}

pub(crate) enum TranscriptBlockMutation<'a> {
    Open {
        id: &'a TranscriptBlockId,
        kind: TranscriptBlockKind,
        content: &'a str,
    },
    Append {
        id: &'a TranscriptBlockId,
        delta: &'a str,
    },
    Replace {
        id: &'a TranscriptBlockId,
        content: &'a str,
    },
    Settle {
        id: &'a TranscriptBlockId,
        terminal: TranscriptBlockTerminal,
    },
    Record {
        id: &'a TranscriptBlockId,
        kind: TranscriptBlockKind,
        content: &'a str,
        terminal: TranscriptBlockTerminal,
    },
}

pub(crate) fn apply_transcript_mutation(
    blocks: &mut Vec<TranscriptBlock>,
    mutation: TranscriptBlockMutation<'_>,
    sequence: u64,
) -> Result<(), TranscriptBlockTransitionError> {
    match mutation {
        TranscriptBlockMutation::Open { id, kind, content } => {
            ensure_new(blocks, id)?;
            blocks.push(TranscriptBlock {
                id: id.clone(),
                kind,
                content: content.to_owned(),
                lifecycle: TranscriptBlockLifecycle::Open,
                opened_sequence: sequence,
                last_transition_sequence: sequence,
            });
        }
        TranscriptBlockMutation::Append { id, delta } => {
            let block = open_block_mut(blocks, id)?;
            block.content.push_str(delta);
            block.last_transition_sequence = sequence;
        }
        TranscriptBlockMutation::Replace { id, content } => {
            let block = open_block_mut(blocks, id)?;
            content.clone_into(&mut block.content);
            block.last_transition_sequence = sequence;
        }
        TranscriptBlockMutation::Settle { id, terminal } => {
            let block = open_block_mut(blocks, id)?;
            block.lifecycle = TranscriptBlockLifecycle::Settled { terminal, sequence };
            block.last_transition_sequence = sequence;
        }
        TranscriptBlockMutation::Record {
            id,
            kind,
            content,
            terminal,
        } => {
            ensure_new(blocks, id)?;
            blocks.push(TranscriptBlock {
                id: id.clone(),
                kind,
                content: content.to_owned(),
                lifecycle: TranscriptBlockLifecycle::Settled { terminal, sequence },
                opened_sequence: sequence,
                last_transition_sequence: sequence,
            });
        }
    }
    Ok(())
}

pub(crate) fn ensure_all_transcript_blocks_settled(
    blocks: &[TranscriptBlock],
) -> Result<(), TranscriptBlockTransitionError> {
    let ids = blocks
        .iter()
        .filter(|block| !block.lifecycle.is_terminal())
        .map(|block| block.id.clone())
        .collect::<Vec<_>>();
    if ids.is_empty() {
        Ok(())
    } else {
        Err(TranscriptBlockTransitionError::UnsettledAtTurnEnd { ids })
    }
}

pub(crate) fn validate_transcript_snapshot(
    blocks: &[TranscriptBlock],
    through_sequence: u64,
) -> Result<(), TranscriptBlockTransitionError> {
    let mut ids = BTreeSet::new();
    let mut previous_opened_sequence = 0;
    for block in blocks {
        let valid_order = block.opened_sequence > 0
            && block.opened_sequence > previous_opened_sequence
            && block.opened_sequence <= block.last_transition_sequence
            && block.last_transition_sequence <= through_sequence;
        let valid_terminal = match block.lifecycle {
            TranscriptBlockLifecycle::Open => true,
            TranscriptBlockLifecycle::Settled { sequence, .. } => {
                sequence == block.last_transition_sequence
            }
        };
        if !ids.insert(block.id.clone()) || !valid_order || !valid_terminal {
            return Err(TranscriptBlockTransitionError::InvalidSnapshot {
                id: block.id.clone(),
            });
        }
        previous_opened_sequence = block.opened_sequence;
    }
    Ok(())
}

fn ensure_new(
    blocks: &[TranscriptBlock],
    id: &TranscriptBlockId,
) -> Result<(), TranscriptBlockTransitionError> {
    if blocks.iter().any(|block| block.id == *id) {
        Err(TranscriptBlockTransitionError::DuplicateId { id: id.clone() })
    } else {
        Ok(())
    }
}

fn open_block_mut<'a>(
    blocks: &'a mut [TranscriptBlock],
    id: &TranscriptBlockId,
) -> Result<&'a mut TranscriptBlock, TranscriptBlockTransitionError> {
    let block = blocks
        .iter_mut()
        .find(|block| block.id == *id)
        .ok_or_else(|| TranscriptBlockTransitionError::UnknownId { id: id.clone() })?;
    if block.lifecycle.is_terminal() {
        Err(TranscriptBlockTransitionError::AlreadySettled { id: id.clone() })
    } else {
        Ok(block)
    }
}

fn validate_id(value: &str) -> Result<(), TranscriptBlockIdError> {
    if value.is_empty() {
        return Err(TranscriptBlockIdError::Empty);
    }
    if value.len() > MAX_TRANSCRIPT_BLOCK_ID_BYTES {
        return Err(TranscriptBlockIdError::TooLong {
            bytes: value.len(),
            maximum: MAX_TRANSCRIPT_BLOCK_ID_BYTES,
        });
    }
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
    {
        return Err(TranscriptBlockIdError::InvalidCharacter);
    }
    Ok(())
}
