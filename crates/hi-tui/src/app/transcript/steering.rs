//! Low-information assistant steering classification and coalescing.

use std::time::Duration;

use ratatui::text::Line;

use crate::TranscriptEntry;

pub(super) fn append_assistant_line(
    transcript: &mut Vec<TranscriptEntry>,
    line: &str,
    append_to_open_message: bool,
) {
    if append_to_open_message
        && let Some(TranscriptEntry::AssistantMessage { text }) = transcript.last_mut()
    {
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str(line);
        return;
    }
    transcript.push(TranscriptEntry::AssistantMessage {
        text: line.to_string(),
    });
}

pub(super) fn last_entry_is_blank(transcript: &[TranscriptEntry]) -> bool {
    match transcript.last() {
        None => true,
        Some(TranscriptEntry::Assistant(line) | TranscriptEntry::Line(line)) => {
            crate::render::line_text(line).trim().is_empty()
        }
        Some(TranscriptEntry::AssistantMessage { text }) => {
            text.lines().last().is_none_or(|l| l.trim().is_empty())
        }
        _ => false,
    }
}

const STEERING_MAX_CHARS: usize = 140;

/// Short “let me look…” chrome, not a real answer (heading / list / document).
pub(super) fn is_steering_assistant_text(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }
    if trimmed.chars().count() > STEERING_MAX_CHARS {
        return false;
    }
    if crate::render::markdown_heading(trimmed).is_some() {
        return false;
    }
    if trimmed.starts_with('▏') || trimmed.starts_with('─') || trimmed.contains('│') {
        return false;
    }
    if crate::render::is_markdown_list_line(text) {
        return false;
    }
    let lower = trimmed.to_ascii_lowercase();
    // Some providers emit this acknowledgement after every internal retry or
    // verification round. It carries no user-facing information, so treat the
    // exact short forms as steering and let the per-turn normalized-text set
    // keep one copy at most. Keep this an allowlist: nearby substantive prose
    // must remain visible even when it starts with "completed" or "done".
    if matches!(
        lower.as_str(),
        "completed the requested action"
            | "completed the requested action."
            | "action completed"
            | "action completed."
            | "done"
            | "done."
    ) {
        return true;
    }
    [
        "i need to ",
        "i'll ",
        "i will ",
        "let me ",
        "let's ",
        "now let me ",
        "checking ",
        "reading ",
        "looking at ",
        "reviewing ",
        "examining ",
    ]
    .iter()
    .any(|prefix| lower.starts_with(prefix))
}

pub(super) fn is_steering_assistant_line(line: &Line<'_>) -> bool {
    let text = crate::render::line_text(line);
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }
    if trimmed.chars().count() > STEERING_MAX_CHARS {
        return false;
    }
    if crate::render::line_looks_like_heading(line) {
        return false;
    }
    if trimmed.starts_with('▏') || trimmed.starts_with('─') || trimmed.contains('│') {
        return false;
    }
    if crate::render::is_markdown_list_line(&text) {
        return false;
    }
    is_steering_assistant_text(trimmed)
}

pub(super) fn normalize_steering_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[derive(Default)]
pub(super) struct ExploreChrome {
    pub(super) thinking: String,
    pub(super) thinking_elapsed: Duration,
    pub(super) steering: Vec<String>,
}

pub(super) fn absorb_explore_chrome(
    group: &mut crate::activity_feed::VerbGroup,
    chrome: ExploreChrome,
) {
    if !chrome.thinking.trim().is_empty() {
        if !group.thinking.is_empty() {
            group.thinking.push('\n');
        }
        group.thinking.push_str(&chrome.thinking);
        group.thinking_elapsed = group
            .thinking_elapsed
            .saturating_add(chrome.thinking_elapsed);
    }
    for line in chrome.steering {
        // A provider can replay the same short preamble around retries or
        // streamed tool boundaries. Keep one copy in the expandable explore
        // row; suppressing it here leaves ordinary assistant prose untouched.
        if !group
            .steering
            .iter()
            .any(|existing| existing.trim() == line.trim())
        {
            group.steering.push(line);
        }
    }
}
