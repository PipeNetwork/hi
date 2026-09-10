//! On-disk `/goal` plan file (`.hi/goal-plan.md`) and grok-style next-step mining.
//!
//! The structured [`crate::Goal`] remains source of truth. The markdown file is
//! the implementer's readable contract; the harness mines the first unchecked
//! `- [ ]` in `## Task checklist` as the per-turn next-step nudge.

use std::io::Read;
use std::path::Path;

const MAX_READ_BYTES: usize = 8 * 1024;

/// First unchecked checklist item in `path`, if the file is readable.
pub(crate) fn first_unchecked_plan_item(path: &Path) -> Option<String> {
    extract_first_unchecked(&read_capped(path)?)
}

fn read_capped(path: &Path) -> Option<String> {
    let file = std::fs::File::open(path).ok()?;
    let mut buf = Vec::with_capacity(MAX_READ_BYTES.min(4096));
    file.take(MAX_READ_BYTES as u64)
        .read_to_end(&mut buf)
        .ok()?;
    if buf.len() >= MAX_READ_BYTES
        && let Some(last_nl) = buf.iter().rposition(|b| *b == b'\n')
    {
        buf.truncate(last_nl);
    }
    Some(String::from_utf8_lossy(&buf).into_owned())
}

fn is_section_header(line: &str, name: &str) -> bool {
    let trimmed = line.trim_start();
    if !trimmed.starts_with('#') {
        return false;
    }
    let title = trimmed.trim_start_matches('#').trim();
    title.eq_ignore_ascii_case(name)
}

fn is_any_header(line: &str) -> bool {
    line.trim_start().starts_with('#')
}

fn header_level(line: &str) -> usize {
    line.trim_start().chars().take_while(|c| *c == '#').count()
}

fn first_unchecked_in_checklist(body: &str) -> Option<String> {
    let mut section_level: Option<usize> = None;
    for line in body.lines() {
        if is_section_header(line, "task checklist") {
            section_level = Some(header_level(line));
            continue;
        }
        let Some(level) = section_level else {
            continue;
        };
        if is_any_header(line) && header_level(line) <= level {
            return None;
        }
        if let Some(item) = parse_checkbox_item(line.trim_start()) {
            return Some(item);
        }
    }
    None
}

const EXCLUDED_SECTIONS: &[&str] = &["non-goals", "deviations", "acceptance criteria"];

fn extract_first_unchecked(body: &str) -> Option<String> {
    if body
        .lines()
        .any(|line| is_section_header(line, "task checklist"))
    {
        return first_unchecked_in_checklist(body);
    }
    let mut excluded = false;
    for line in body.lines() {
        if is_any_header(line) {
            excluded = EXCLUDED_SECTIONS
                .iter()
                .any(|name| is_section_header(line, name));
            continue;
        }
        if excluded {
            continue;
        }
        if let Some(item) = parse_checkbox_item(line.trim_start()) {
            return Some(item);
        }
    }
    None
}

fn strip_bullet_marker(trimmed: &str) -> Option<&str> {
    trimmed
        .strip_prefix("- ")
        .or_else(|| trimmed.strip_prefix("* "))
        .or_else(|| trimmed.strip_prefix("+ "))
        .map(str::trim_start)
}

fn parse_checkbox_item(trimmed: &str) -> Option<String> {
    let after_marker = strip_bullet_marker(trimmed)?;
    let after_checkbox = after_marker.strip_prefix("[ ]")?;
    let text = after_checkbox.trim();
    (!text.is_empty()).then(|| text.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mines_first_unchecked_in_task_checklist() {
        let body = "# Goal\n\n## Acceptance criteria\n1. ships\n\n## Task checklist\n- [x] done\n- [ ] write the lexer\n- [ ] write the parser\n";
        assert_eq!(
            extract_first_unchecked(body).as_deref(),
            Some("write the lexer")
        );
    }

    #[test]
    fn ignores_acceptance_numbered_items() {
        let body = "## Acceptance criteria\n- [ ] not a step\n## Task checklist\n- [ ] real step\n";
        assert_eq!(extract_first_unchecked(body).as_deref(), Some("real step"));
    }

    #[test]
    fn returns_none_when_all_checked() {
        let body = "## Task checklist\n- [x] one\n- [X] two\n";
        assert_eq!(extract_first_unchecked(body), None);
    }
}
