//! Conservative recognition of compound read-only shell inspections and
//! bounded foreground execution probes.

use super::{
    BashCommandKind, classify_bash_command, git_subcommand_is_read_only,
    shell_command_has_known_side_effects, shell_command_likely_edits_files,
    shell_command_likely_mutates_workspace, shell_command_no_progress_signature,
    simple_shell_words,
};

pub(crate) fn bash_command(arguments: &str) -> Option<String> {
    let value = serde_json::from_str::<serde_json::Value>(arguments).ok()?;
    value
        .get("command")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}

pub(crate) fn bash_no_progress_signature(arguments: &str) -> Option<&'static str> {
    let command = bash_command(arguments)?;
    shell_command_no_progress_signature(&command)
}

/// Stable text for a shell call that is provably inspection-only.
///
/// Unsupported shell syntax or any command outside the read-only allowlist
/// returns `None` and retains the mutation-safe path.
pub(crate) fn bash_inspection_signature(arguments: &str) -> Option<String> {
    let command = bash_command(arguments)?;
    (classify_bash_command(&command) == BashCommandKind::Inspection)
        .then(|| normalize_inspection_command(&command))
}

/// Identify a bounded foreground execution probe such as
/// `timeout 10 ./target/debug/app | head`. Callers must separately prove that
/// the command caused no workspace mutation before using the signature.
pub(crate) fn bash_bounded_execution_probe(arguments: &str) -> Option<String> {
    let command = bash_command(arguments)?;
    command
        .split([';', '\n'])
        .find_map(bounded_execution_segment)
}

fn bounded_execution_segment(segment: &str) -> Option<String> {
    let mut words = segment.split_whitespace();
    let timeout = words.next()?;
    if !matches!(timeout.rsplit('/').next(), Some("timeout" | "gtimeout")) {
        return None;
    }
    let duration = words.next()?.trim_end_matches(['s', 'm', 'h', 'd']);
    if duration.is_empty() || !duration.chars().all(|character| character.is_ascii_digit()) {
        return None;
    }
    let target = words.next()?;
    (!target.starts_with('-')
        && !target
            .chars()
            .any(|character| matches!(character, '|' | '&' | '>' | '<')))
    .then(|| target.to_string())
}

fn normalize_inspection_command(command: &str) -> String {
    command.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Recognize control-flow/pipeline wrappers made exclusively from commands
/// that cannot mutate the workspace. This is intentionally not a general
/// shell parser and rejects unfamiliar or stateful constructs.
pub(super) fn compound_shell_is_read_only_inspection(command: &str) -> bool {
    if shell_command_likely_mutates_workspace(command) || shell_command_likely_edits_files(command)
    {
        return false;
    }
    let Some(segments) = read_only_shell_segments(command) else {
        return false;
    };
    segments.len() >= 2
        && segments
            .iter()
            .all(|segment| shell_segment_is_read_only(segment))
}

fn read_only_shell_segments(command: &str) -> Option<Vec<String>> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut chars = command.chars().peekable();
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;

    while let Some(ch) = chars.next() {
        if escaped {
            current.push(ch);
            escaped = false;
            continue;
        }
        if in_single {
            current.push(ch);
            if ch == '\'' {
                in_single = false;
            }
            continue;
        }
        if in_double {
            current.push(ch);
            match ch {
                '"' => in_double = false,
                '\\' => escaped = true,
                '$' if chars.peek() == Some(&'(') => return None,
                '`' => return None,
                _ => {}
            }
            continue;
        }
        match ch {
            '\'' => {
                in_single = true;
                current.push(ch);
            }
            '"' => {
                in_double = true;
                current.push(ch);
            }
            '\\' => {
                escaped = true;
                current.push(ch);
            }
            '$' if chars.peek() == Some(&'(') => return None,
            '`' | '<' | '>' | '(' | ')' | '{' | '}' | '#' => return None,
            ';' | '\n' | '|' => {
                push_shell_segment(&mut segments, &mut current)?;
                if ch == '|' && chars.peek() == Some(&'|') {
                    chars.next();
                }
            }
            '&' => {
                if chars.peek() != Some(&'&') {
                    return None;
                }
                chars.next();
                push_shell_segment(&mut segments, &mut current)?;
            }
            _ => current.push(ch),
        }
    }
    if escaped || in_single || in_double {
        return None;
    }
    if !current.trim().is_empty() {
        segments.push(current.trim().to_string());
    }
    (!segments.is_empty()).then_some(segments)
}

fn push_shell_segment(segments: &mut Vec<String>, current: &mut String) -> Option<()> {
    let segment = current.trim();
    if segment.is_empty() {
        return None;
    }
    segments.push(segment.to_string());
    current.clear();
    Some(())
}

fn shell_segment_is_read_only(segment: &str) -> bool {
    let mut segment = segment.trim();
    for keyword in ["do", "then", "else"] {
        if segment == keyword {
            return true;
        }
        if let Some(rest) = segment
            .strip_prefix(keyword)
            .and_then(|rest| rest.strip_prefix(' '))
        {
            segment = rest.trim_start();
            break;
        }
    }
    if matches!(segment, "done" | "fi") {
        return true;
    }
    if let Some(header) = segment.strip_prefix("for ") {
        let words = header.split_whitespace().collect::<Vec<_>>();
        return words.len() >= 3
            && valid_shell_identifier(words[0])
            && words[1] == "in"
            && !header.contains("$(")
            && !header.contains('`');
    }

    let command = segment.split_whitespace().next().unwrap_or_default();
    if command == "git" {
        return simple_shell_words(segment)
            .is_some_and(|words| git_subcommand_is_read_only(&words[1..]));
    }
    matches!(
        command,
        "pwd"
            | "ls"
            | "find"
            | "rg"
            | "grep"
            | "cat"
            | "sed"
            | "nl"
            | "head"
            | "tail"
            | "echo"
            | "printf"
            | "tr"
            | "cut"
            | "sort"
            | "uniq"
            | "wc"
            | "stat"
            | "file"
            | "du"
            | "basename"
            | "dirname"
            | "readlink"
            | "realpath"
    ) && !shell_command_has_known_side_effects(
        &simple_shell_words(segment).unwrap_or_else(|| vec![command.to_string()]),
    )
}

fn valid_shell_identifier(word: &str) -> bool {
    let mut chars = word.chars();
    chars
        .next()
        .is_some_and(|ch| ch == '_' || ch.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_execution_probes_share_the_program_signature() {
        assert_eq!(
            bash_bounded_execution_probe(
                r#"{"command":"timeout 10 ./target/debug/app 2>&1 | head -30; echo exit=$?"}"#
            )
            .as_deref(),
            Some("./target/debug/app")
        );
        assert_eq!(
            bash_bounded_execution_probe(
                r#"{"command":"timeout 20 ./target/debug/app 2>&1 | tail -30"}"#
            )
            .as_deref(),
            Some("./target/debug/app")
        );
        assert!(bash_bounded_execution_probe(r#"{"command":"cargo test"}"#).is_none());
    }

    #[test]
    fn compound_read_only_shell_loops_are_inspections() {
        let command = "for f in blog_posts/txt/*.txt; do echo \"=== $f ===\"; head -2 \"$f\" | tr '\\n' ' '; echo; done | sed -n '20,46p'";
        assert_eq!(classify_bash_command(command), BashCommandKind::Inspection);

        let arguments = serde_json::json!({"command": command}).to_string();
        assert_eq!(
            bash_inspection_signature(&arguments),
            Some(normalize_inspection_command(command))
        );
    }

    #[test]
    fn compound_shell_mutations_and_ambiguous_commands_stay_conservative() {
        for command in [
            "for f in src/*.rs; do sed -i s/old/new/ \"$f\"; done",
            "for f in src/*.rs; do cat \"$f\" > combined.txt; done",
            "for f in src/*.rs; do rm \"$f\"; done",
            "find src -type f | xargs touch",
            "for f in src/*.rs; do sh -c 'cat \"$f\"'; done",
            "for f in src/*.rs; do echo $(cat \"$f\"); done",
        ] {
            assert_ne!(
                classify_bash_command(command),
                BashCommandKind::Inspection,
                "{command:?} must retain mutation-safe handling"
            );
        }
    }
}
