//! Typed values for the workspace store.

use std::fmt;

use crate::error::{Result, StoreError};

pub const WORKSPACE_CAPACITY: usize = 256;
pub const RANK_GAP: i64 = 1024;
pub const MAX_SESSION_ID_BYTES: usize = 255;
pub const MAX_CWD_BYTES: usize = 4096;
pub const MAX_TITLE_BYTES: usize = 1024;
pub const MAX_MODEL_BYTES: usize = 256;
pub const MAX_SUMMARY_BYTES: usize = 8192;
pub const MAX_ENUM_BYTES: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SessionId(String);

impl SessionId {
    pub fn new(raw: impl Into<String>) -> Result<Self> {
        let raw = raw.into();
        let reason = if raw.is_empty() {
            Some("empty")
        } else if raw.len() > MAX_SESSION_ID_BYTES {
            Some("longer than the byte cap")
        } else if raw == "." || raw == ".." {
            Some("reserved path component")
        } else if raw.contains(['/', '\\']) {
            Some("contains a path separator")
        } else if raw.contains(['<', '>', ':', '"', '|', '?', '*']) {
            Some("contains a character Windows forbids in file names")
        } else if raw.ends_with(['.', ' ']) {
            Some("has a trailing dot or space")
        } else if is_windows_reserved_name(&raw) {
            Some("is a reserved Windows device name")
        } else if raw.chars().any(is_identifier_control) {
            Some("contains a control character")
        } else {
            None
        };
        match reason {
            Some(reason) => Err(StoreError::InvalidSessionId { reason }),
            None => Ok(Self(raw)),
        }
    }
}

fn is_windows_reserved_name(raw: &str) -> bool {
    let stem_end = raw.find('.').unwrap_or(raw.len());
    let stem = raw[..stem_end].trim_end_matches(' ');
    if ["CON", "PRN", "AUX", "NUL"]
        .iter()
        .any(|reserved| stem.eq_ignore_ascii_case(reserved))
    {
        return true;
    }
    let Some(prefix) = stem.get(..3) else {
        return false;
    };
    let mut suffix = stem[3..].chars();
    (prefix.eq_ignore_ascii_case("COM") || prefix.eq_ignore_ascii_case("LPT"))
        && matches!(suffix.next(), Some('1'..='9' | '¹' | '²' | '³'))
        && suffix.next().is_none()
}

fn is_identifier_control(c: char) -> bool {
    c.is_control()
        || matches!(
            c,
            '\u{061c}'
                | '\u{200e}'
                | '\u{200f}'
                | '\u{202a}'..='\u{202e}'
                | '\u{2066}'..='\u{2069}'
        )
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for SessionId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

fn validate_unknown(column: &'static str, raw: &str) -> Result<()> {
    if raw.is_empty() {
        return Err(StoreError::InvalidEnumValue {
            column,
            reason: "empty",
        });
    }
    if raw.len() > MAX_ENUM_BYTES {
        return Err(StoreError::EnumValueTooLong {
            column,
            max: MAX_ENUM_BYTES,
        });
    }
    Ok(())
}

macro_rules! unknown_value {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $name(String);

        impl $name {
            fn parsed(column: &'static str, raw: &str) -> Result<Self> {
                validate_unknown(column, raw)?;
                Ok(Self(raw.to_owned()))
            }

            fn stored(raw: String) -> Self {
                Self(raw)
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }
    };
}

unknown_value!(UnknownMemberKind);
unknown_value!(UnknownMemberOrigin);
unknown_value!(UnknownGrouping);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MemberKind {
    Build,
    Conversation,
    Other(UnknownMemberKind),
}

impl MemberKind {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Build => "build",
            Self::Conversation => "conversation",
            Self::Other(value) => value.as_ref(),
        }
    }

    pub fn from_raw(raw: &str) -> Result<Self> {
        match raw {
            "build" => Ok(Self::Build),
            "conversation" => Ok(Self::Conversation),
            _ => Ok(Self::Other(UnknownMemberKind::parsed("kind", raw)?)),
        }
    }

    pub(crate) fn from_stored(raw: String) -> Self {
        match raw.as_str() {
            "build" => Self::Build,
            "conversation" => Self::Conversation,
            _ => Self::Other(UnknownMemberKind::stored(raw)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemberOrigin {
    Local,
    Remote,
    Other(UnknownMemberOrigin),
}

impl MemberOrigin {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Local => "local",
            Self::Remote => "remote",
            Self::Other(value) => value.as_ref(),
        }
    }

    pub fn from_raw(raw: &str) -> Result<Self> {
        match raw {
            "local" => Ok(Self::Local),
            "remote" => Ok(Self::Remote),
            _ => Ok(Self::Other(UnknownMemberOrigin::parsed("origin", raw)?)),
        }
    }

    pub(crate) fn from_stored(raw: String) -> Self {
        match raw.as_str() {
            "local" => Self::Local,
            "remote" => Self::Remote,
            _ => Self::Other(UnknownMemberOrigin::stored(raw)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Grouping {
    State,
    Directory,
    Other(UnknownGrouping),
}

impl Grouping {
    pub fn as_str(&self) -> &str {
        match self {
            Self::State => "state",
            Self::Directory => "directory",
            Self::Other(value) => value.as_ref(),
        }
    }

    pub fn from_raw(raw: &str) -> Result<Self> {
        match raw {
            "state" => Ok(Self::State),
            "directory" => Ok(Self::Directory),
            _ => Ok(Self::Other(UnknownGrouping::parsed("grouping", raw)?)),
        }
    }

    pub(crate) fn from_stored(raw: String) -> Self {
        match raw.as_str() {
            "state" => Self::State,
            "directory" => Self::Directory,
            _ => Self::Other(UnknownGrouping::stored(raw)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MemberKey {
    pub session_id: SessionId,
    pub kind: MemberKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemberMetadata {
    pub cwd: Option<String>,
    pub title: Option<String>,
    pub model: Option<String>,
    pub last_turn_summary: Option<String>,
    pub is_worktree: bool,
    pub last_change_unix_ms: i64,
}

impl MemberMetadata {
    pub(crate) fn validated(&self, kind: &MemberKind) -> Result<ValidatedMetadataRef<'_>> {
        if matches!(kind, MemberKind::Build) && self.cwd.is_none() {
            return Err(StoreError::CwdRequired);
        }
        if let Some(cwd) = &self.cwd {
            if !std::path::Path::new(cwd).is_absolute() {
                return Err(StoreError::CwdNotAbsolute);
            }
            if cwd.len() > MAX_CWD_BYTES {
                return Err(StoreError::CwdTooLong { max: MAX_CWD_BYTES });
            }
        }
        Ok(ValidatedMetadataRef {
            cwd: self.cwd.as_deref(),
            title: self
                .title
                .as_deref()
                .map(|text| truncate_at_char_boundary(text, MAX_TITLE_BYTES)),
            model: self
                .model
                .as_deref()
                .map(|text| truncate_at_char_boundary(text, MAX_MODEL_BYTES)),
            last_turn_summary: self
                .last_turn_summary
                .as_deref()
                .map(|text| truncate_at_char_boundary(text, MAX_SUMMARY_BYTES)),
            is_worktree: self.is_worktree,
            last_change_unix_ms: self.last_change_unix_ms,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ValidatedMetadataRef<'a> {
    pub cwd: Option<&'a str>,
    pub title: Option<&'a str>,
    pub model: Option<&'a str>,
    pub last_turn_summary: Option<&'a str>,
    pub is_worktree: bool,
    pub last_change_unix_ms: i64,
}

fn truncate_at_char_boundary(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut end = max_bytes;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

#[derive(Debug, Clone)]
pub struct NewMember {
    pub key: MemberKey,
    pub origin: MemberOrigin,
    pub metadata: MemberMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Member {
    pub session_id: SessionId,
    pub kind: MemberKind,
    pub origin: MemberOrigin,
    pub cwd: Option<String>,
    pub title: Option<String>,
    pub model: Option<String>,
    pub last_turn_summary: Option<String>,
    pub is_worktree: bool,
    pub last_change_unix_ms: i64,
    pub pin_rank: Option<i64>,
    pub order_rank: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceSnapshot {
    pub grouping: Grouping,
    pub members: Vec<Member>,
    pub data_version: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankAssignment {
    pub key: MemberKey,
    pub rank: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinAssignment {
    pub key: MemberKey,
    pub pinned: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutGrouping {
    State,
    Directory,
}

impl LayoutGrouping {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::State => "state",
            Self::Directory => "directory",
        }
    }
}

impl AsRef<str> for LayoutGrouping {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutPatch {
    pub pin_assignments: Vec<PinAssignment>,
    pub manual_order: Option<Vec<MemberKey>>,
    pub grouping: Option<LayoutGrouping>,
}

impl LayoutPatch {
    pub fn is_empty(&self) -> bool {
        self.pin_assignments.is_empty() && self.manual_order.is_none() && self.grouping.is_none()
    }
}

#[derive(Debug)]
#[must_use]
pub enum LayoutApplyOutcome {
    Committed(WorkspaceSnapshot),
    Rejected {
        error: StoreError,
        snapshot: WorkspaceSnapshot,
    },
    Failed {
        error: StoreError,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InsertOutcome {
    Inserted,
    InsertedEvicting(Vec<MemberKey>),
    UpdatedExisting,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoveOutcome {
    Removed,
    NotPresent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RekeyOutcome {
    Moved,
    MergedIntoExisting,
    NoChange,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaState {
    Current,
    NewerReadOnly { user_version: u32 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_id_rejects_path_separators() {
        assert!(SessionId::new("ok-id").is_ok());
        assert!(SessionId::new("../x").is_err());
        assert!(SessionId::new("a/b").is_err());
        assert!(SessionId::new("").is_err());
    }

    #[test]
    fn member_kind_round_trips_unknown() {
        let kind = MemberKind::from_raw("future-kind").unwrap();
        assert_eq!(kind.as_str(), "future-kind");
        assert_eq!(
            MemberKind::from_stored("build".to_string()),
            MemberKind::Build
        );
    }
}
