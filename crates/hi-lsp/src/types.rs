//! Public LSP result types and protocol-to-user position conversion.

use std::path::Path;

use serde_json::Value;

use crate::client::uri_to_path;

/// One LSP diagnostic (error/warning), flattened for callers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    pub severity: String,
    pub line: u32,
    /// Zero-based Unicode-scalar offset (not the protocol's UTF-16 units).
    pub col: u32,
    pub message: String,
    pub source: Option<String>,
}

/// Version-aware result of asking a language server about one document.
/// Empty diagnostics are represented explicitly so callers never confuse a
/// timeout, unsupported pull diagnostics, or a crashed server with a confirmed
/// clean document.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DiagnosticState {
    ConfirmedClean {
        document_version: u64,
    },
    DiagnosticsPresent {
        document_version: u64,
        diagnostics: Vec<Diagnostic>,
    },
    Unavailable {
        document_version: Option<u64>,
        reason: String,
    },
    Failed {
        document_version: Option<u64>,
        error: String,
    },
}

impl DiagnosticState {
    pub fn document_version(&self) -> Option<u64> {
        match self {
            Self::ConfirmedClean { document_version }
            | Self::DiagnosticsPresent {
                document_version, ..
            } => Some(*document_version),
            Self::Unavailable {
                document_version, ..
            }
            | Self::Failed {
                document_version, ..
            } => *document_version,
        }
    }

    pub fn diagnostics(&self) -> Option<&[Diagnostic]> {
        match self {
            Self::ConfirmedClean { .. } => Some(&[]),
            Self::DiagnosticsPresent { diagnostics, .. } => Some(diagnostics),
            Self::Unavailable { .. } | Self::Failed { .. } => None,
        }
    }
}

/// One location using zero-based line and Unicode-scalar column offsets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Location {
    pub path: String,
    pub line: u32,
    pub col: u32,
}

fn severity_label(n: u64) -> String {
    match n {
        1 => "error".into(),
        2 => "warning".into(),
        3 => "info".into(),
        4 => "hint".into(),
        _ => "note".into(),
    }
}

fn parse_diagnostic(value: &Value) -> Option<Diagnostic> {
    let severity = value.get("severity").and_then(Value::as_u64).unwrap_or(0);
    let start = value.get("range")?.get("start")?;
    Some(Diagnostic {
        severity: severity_label(severity),
        line: start.get("line")?.as_u64()? as u32,
        col: start.get("character")?.as_u64()? as u32,
        message: value.get("message")?.as_str()?.to_string(),
        source: value
            .get("source")
            .and_then(Value::as_str)
            .map(String::from),
    })
}

pub(crate) fn diagnostic_state_from_items(
    path: &Path,
    version: u64,
    items: &[Value],
) -> DiagnosticState {
    let mut diagnostics: Vec<Diagnostic> = items.iter().filter_map(parse_diagnostic).collect();
    for diagnostic in &mut diagnostics {
        diagnostic.col = file_utf16_to_character(path, diagnostic.line, diagnostic.col)
            .unwrap_or(diagnostic.col);
    }
    if diagnostics.is_empty() {
        DiagnosticState::ConfirmedClean {
            document_version: version,
        }
    } else {
        DiagnosticState::DiagnosticsPresent {
            document_version: version,
            diagnostics,
        }
    }
}

fn parse_location(value: &Value) -> Option<Location> {
    // Definition responses may contain either Location or LocationLink.
    let uri = value
        .get("uri")
        .or_else(|| value.get("targetUri"))?
        .as_str()?;
    let range = value
        .get("range")
        .or_else(|| value.get("targetSelectionRange"))
        .or_else(|| value.get("targetRange"))?;
    let start = range.get("start")?;
    Some(Location {
        path: uri_to_path(uri),
        line: start.get("line")?.as_u64()? as u32,
        col: start.get("character")?.as_u64()? as u32,
    })
}

pub(crate) fn parse_locations(value: &Value) -> Vec<Location> {
    match value {
        Value::Array(items) => items.iter().filter_map(parse_location).collect(),
        Value::Object(_) => parse_location(value).into_iter().collect(),
        _ => Vec::new(),
    }
}

pub(crate) fn parse_hover(value: &Value) -> Option<String> {
    fn render(content: &Value, out: &mut Vec<String>) {
        match content {
            Value::String(text) => out.push(text.clone()),
            Value::Array(items) => {
                for item in items {
                    render(item, out);
                }
            }
            Value::Object(object) => {
                if let Some(text) = object.get("value").and_then(Value::as_str) {
                    out.push(text.to_string());
                } else if let Some(contents) = object.get("contents") {
                    render(contents, out);
                }
            }
            _ => {}
        }
    }
    let content = value.get("contents").unwrap_or(value);
    let mut parts = Vec::new();
    render(content, &mut parts);
    (!parts.is_empty()).then(|| parts.join("\n\n"))
}

fn line_text(text: &str, line: u32) -> Option<&str> {
    text.split('\n')
        .nth(line as usize)
        .map(|line| line.strip_suffix('\r').unwrap_or(line))
}

/// Convert a user-facing Unicode-scalar column to the UTF-16 code-unit offset
/// required by LSP.
fn character_to_utf16(text: &str, line: u32, col: u32) -> Option<u32> {
    let line = line_text(text, line)?;
    Some(
        line.chars()
            .take(col as usize)
            .map(char::len_utf16)
            .sum::<usize>() as u32,
    )
}

fn utf16_to_character(text: &str, line: u32, col: u32) -> Option<u32> {
    let line = line_text(text, line)?;
    let mut units = 0_u32;
    let mut characters = 0_u32;
    for character in line.chars() {
        if units >= col {
            break;
        }
        units = units.saturating_add(character.len_utf16() as u32);
        characters += 1;
    }
    Some(characters)
}

pub(crate) fn file_character_to_utf16(path: &Path, line: u32, col: u32) -> Option<u32> {
    character_to_utf16(&std::fs::read_to_string(path).ok()?, line, col)
}

pub(crate) fn file_utf16_to_character(path: &Path, line: u32, col: u32) -> Option<u32> {
    utf16_to_character(&std::fs::read_to_string(path).ok()?, line, col)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn utf16_positions_round_trip_non_bmp_characters() {
        let text = "a😀βz\n";
        assert_eq!(character_to_utf16(text, 0, 0), Some(0));
        assert_eq!(character_to_utf16(text, 0, 2), Some(3));
        assert_eq!(character_to_utf16(text, 0, 3), Some(4));
        assert_eq!(utf16_to_character(text, 0, 3), Some(2));
        assert_eq!(utf16_to_character(text, 0, 4), Some(3));
    }

    #[test]
    fn location_link_uses_target_selection_range() {
        let value = json!({
            "targetUri": "file:///tmp/target.rs",
            "targetRange": {
                "start": { "line": 1, "character": 2 },
                "end": { "line": 1, "character": 8 }
            },
            "targetSelectionRange": {
                "start": { "line": 1, "character": 4 },
                "end": { "line": 1, "character": 8 }
            }
        });
        assert_eq!(
            parse_locations(&value),
            vec![Location {
                path: "/tmp/target.rs".into(),
                line: 1,
                col: 4,
            }]
        );
    }

    #[test]
    fn hover_renders_marked_string_arrays_and_markup() {
        let value = json!({
            "contents": [
                { "language": "rust", "value": "fn answer() -> u32" },
                "Returns the answer."
            ]
        });
        assert_eq!(
            parse_hover(&value).as_deref(),
            Some("fn answer() -> u32\n\nReturns the answer.")
        );
        assert_eq!(
            parse_hover(&json!({ "contents": { "kind": "markdown", "value": "**docs**" } }))
                .as_deref(),
            Some("**docs**")
        );
    }

    #[test]
    fn empty_items_are_confirmed_clean_not_unavailable() {
        assert_eq!(
            diagnostic_state_from_items(Path::new("/missing"), 7, &[]),
            DiagnosticState::ConfirmedClean {
                document_version: 7
            }
        );
    }
}
