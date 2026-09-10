//! `/compact [kind] [extra instructions…]` parsing.

use crate::compaction::CompactionKind;

/// Parsed `/compact` argument: optional kind plus leftover summarizer notes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CompactArg {
    pub kind: Option<CompactionKind>,
    pub extra_instructions: Option<String>,
}

impl CompactArg {
    pub fn parse(arg: &str) -> Self {
        let trimmed = arg.trim();
        if trimmed.is_empty() {
            return Self::default();
        }
        let (first, rest) = match trimmed.split_once(|c: char| c.is_whitespace()) {
            Some((head, tail)) => (head, tail.trim()),
            None => (trimmed, ""),
        };
        if let Some(kind) = CompactionKind::from_arg(first) {
            return Self {
                kind: Some(kind),
                extra_instructions: if rest.is_empty() {
                    None
                } else {
                    Some(rest.to_string())
                },
            };
        }
        Self {
            kind: None,
            extra_instructions: Some(trimmed.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::DEFAULT_KEEP_RECENT;

    #[test]
    fn parse_kind_and_extra_instructions() {
        assert_eq!(
            CompactArg::parse("hybrid keep the API notes"),
            CompactArg {
                kind: Some(CompactionKind::Hybrid {
                    keep_recent: DEFAULT_KEEP_RECENT
                }),
                extra_instructions: Some("keep the API notes".into()),
            }
        );
        assert_eq!(
            CompactArg::parse("keep the API notes"),
            CompactArg {
                kind: None,
                extra_instructions: Some("keep the API notes".into()),
            }
        );
    }
}
