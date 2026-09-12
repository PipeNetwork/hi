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
    (classify_bash_command(&command) == BashCommandKind::Inspection).then(|| {
        inspection_core_identity(&command).unwrap_or_else(|| normalize_inspection_command(&command))
    })
}

/// Source-file operands of a read-only dump (`cat`/`sed`/`head`/`tail`/`nl`).
/// Used so `cd dir && cat src/foo.rs` cannot count as new evidence after the
/// `read` tool already returned that file.
pub(crate) fn bash_inspection_paths(arguments: &str) -> Vec<String> {
    let Some(command) = bash_command(arguments) else {
        return Vec::new();
    };
    if classify_bash_command(&command) != BashCommandKind::Inspection {
        return Vec::new();
    }
    let segments = read_only_shell_segments(&command).unwrap_or_else(|| vec![command.clone()]);
    let mut paths = Vec::new();
    for segment in segments {
        collect_dump_paths(&segment, &mut paths);
    }
    paths
}

fn collect_dump_paths(segment: &str, paths: &mut Vec<String>) {
    let Some(words) = simple_shell_words(segment) else {
        return;
    };
    let Some(cmd) = words.first() else {
        return;
    };
    let cmd = cmd.rsplit('/').next().unwrap_or(cmd);
    match cmd {
        "cat" | "nl" => {
            for word in words.iter().skip(1) {
                if looks_like_source_path(word) {
                    paths.push(word.clone());
                }
            }
        }
        "sed" | "head" | "tail" => {
            if let Some(word) = words.iter().rev().find(|word| looks_like_source_path(word)) {
                paths.push(word.clone());
            }
        }
        "grep" | "egrep" | "ggrep" => {
            if let Some(path) = grep_full_file_dump_operand(&words) {
                paths.push(path);
            }
        }
        _ => {}
    }
}

fn grep_full_file_dump_operand(words: &[String]) -> Option<String> {
    let mut pattern: Option<&str> = None;
    let mut path: Option<&str> = None;
    for word in words.iter().skip(1) {
        if word == "--" || matches!(word.as_str(), "-n" | "--line-number") {
            continue;
        }
        if word.starts_with('-') {
            return None;
        }
        if pattern.is_none() {
            pattern = Some(word);
        } else if path.is_none() {
            path = Some(word);
        } else {
            return None;
        }
    }
    if !matches!(pattern?, "" | "^" | "." | ".*" | "^.*$") {
        return None;
    }
    Some(path?.to_string())
}

fn looks_like_source_path(word: &str) -> bool {
    if word.starts_with('-') || word.is_empty() {
        return false;
    }
    word.contains('/')
        || word.ends_with(".rs")
        || word.ends_with(".toml")
        || word.ends_with(".md")
        || word.ends_with(".json")
        || word.ends_with(".lock")
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
    !segments.is_empty()
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
            '>' => {
                // Live stall: `grep … 2>&1; echo EXIT=$?` was Unknown because
                // `>` rejected the whole pipeline, so withhold/stationarity
                // never saw an inspection.
                if !skip_readonly_redirect(&mut chars, &mut current) {
                    return None;
                }
            }
            '`' | '<' | '(' | ')' | '{' | '}' | '#' => return None,
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
    // `cd dir && cat file` is how models re-dump files after the read tool
    // refuses a reread. Changing directory is not a workspace mutation.
    if command == "cd" {
        return true;
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
            | "fold"
            | "awk"
            | "od"
            | "base64"
            | "xxd"
            | "hexdump"
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

/// Consume `2>&1`, `>&2`, or `>/dev/null` after a `>` was already read.
/// `head -2 >/dev/null` must not treat `-2` as a file descriptor.
fn skip_readonly_redirect(
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
    current: &mut String,
) -> bool {
    let mut lookahead = chars.clone();
    while matches!(lookahead.peek(), Some(ch) if ch.is_whitespace()) {
        lookahead.next();
    }
    let rest: String = lookahead.take(12).collect();
    if rest.starts_with("&1") || rest.starts_with("&2") {
        while matches!(chars.peek(), Some(ch) if ch.is_whitespace()) {
            chars.next();
        }
        chars.next();
        chars.next();
        strip_trailing_standalone_fd(current);
        return true;
    }
    if rest.starts_with("/dev/null") {
        while matches!(chars.peek(), Some(ch) if ch.is_whitespace()) {
            chars.next();
        }
        for _ in 0.."/dev/null".len() {
            chars.next();
        }
        strip_trailing_standalone_fd(current);
        return true;
    }
    false
}

fn strip_trailing_standalone_fd(current: &mut String) {
    let trimmed = current.trim_end();
    let Some(fd) = trimmed.chars().last() else {
        return;
    };
    if !matches!(fd, '1' | '2') {
        return;
    }
    let without = &trimmed[..trimmed.len() - fd.len_utf8()];
    if without.is_empty() || without.ends_with(char::is_whitespace) {
        *current = without.trim_end().to_string();
    }
}

/// First grep/cat/sed of a source path, ignoring later `sed`/`tr` wrappers and
/// `2>&1` so a growing decode pipeline is the same inspection.
fn inspection_core_identity(command: &str) -> Option<String> {
    let segments = read_only_shell_segments(command)?;
    for segment in segments {
        let Some(words) = simple_shell_words(&segment) else {
            continue;
        };
        let Some(cmd) = words.first() else {
            continue;
        };
        let cmd = cmd.rsplit('/').next().unwrap_or(cmd);
        match cmd {
            "grep" | "egrep" | "ggrep" | "rg" => {
                let paths = words
                    .iter()
                    .skip(1)
                    .filter(|word| looks_like_source_path(word) && !word.starts_with('-'))
                    .cloned()
                    .collect::<Vec<_>>();
                if paths.is_empty() {
                    continue;
                }
                let pattern = words
                    .iter()
                    .skip(1)
                    .find(|word| !word.starts_with('-') && !looks_like_source_path(word))
                    .cloned()
                    .unwrap_or_default();
                return Some(format!("{cmd}:{pattern}:{}", paths.join("\u{1f}")));
            }
            "cat" | "nl" | "sed" | "head" | "tail" => {
                if let Some(path) = words.iter().rev().find(|word| looks_like_source_path(word)) {
                    return Some(format!("dump:{path}"));
                }
            }
            _ => {}
        }
    }
    None
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
    fn cd_and_cat_is_inspection_of_the_dumped_path() {
        let command = "cd /Users/david/chat && cat src/state.rs";
        assert_eq!(classify_bash_command(command), BashCommandKind::Inspection);
        let arguments = serde_json::json!({"command": command}).to_string();
        assert_eq!(
            bash_inspection_paths(&arguments),
            vec!["src/state.rs".to_string()]
        );
        let paged = "cd /Users/david/chat && sed -n '120,420p' src/server.rs";
        assert_eq!(classify_bash_command(paged), BashCommandKind::Inspection);
        assert_eq!(
            bash_inspection_paths(&serde_json::json!({"command": paged}).to_string()),
            vec!["src/server.rs".to_string()]
        );
        let grep_page = r#"grep -n "" src/ws.rs | sed -n '206,300p'"#;
        assert_eq!(
            classify_bash_command(grep_page),
            BashCommandKind::Inspection
        );
        assert_eq!(
            bash_inspection_paths(&serde_json::json!({"command": grep_page}).to_string()),
            vec!["src/ws.rs".to_string()]
        );
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
    fn stderr_redirect_and_exit_echo_stay_inspection() {
        let command = r#"grep -nE 'pub fn' src/db.rs | head -10 2>&1; echo "EXIT=$?""#;
        assert_eq!(classify_bash_command(command), BashCommandKind::Inspection);
        let arguments = serde_json::json!({"command": command}).to_string();
        assert_eq!(
            bash_inspection_signature(&arguments).as_deref(),
            Some("grep:pub fn:src/db.rs")
        );
        assert!(is_withheld_style(&arguments));
    }

    #[test]
    fn growing_sed_wrappers_share_the_grep_core_identity() {
        let first =
            r#"grep -nE 'pub fn' src/db.rs | head -10 | sed 's/^/GOT: /' 2>&1; echo "EXIT=$?""#;
        let grown = r#"grep -nE 'pub fn' src/db.rs | head -10 | sed 's/^/GOT: /' | sed 's/[a-z]/x/g' | tr -d '\n' 2>&1; echo "EXIT=$?""#;
        let first_args = serde_json::json!({"command": first}).to_string();
        let grown_args = serde_json::json!({"command": grown}).to_string();
        assert_eq!(
            bash_inspection_signature(&first_args),
            bash_inspection_signature(&grown_args)
        );
        assert_eq!(
            bash_inspection_signature(&first_args).as_deref(),
            Some("grep:pub fn:src/db.rs")
        );
    }

    #[test]
    fn workspace_redirects_are_not_inspection() {
        let command = "grep -nE 'pub fn' src/db.rs > src/out.txt";
        assert_ne!(classify_bash_command(command), BashCommandKind::Inspection);
    }

    fn is_withheld_style(arguments: &str) -> bool {
        bash_inspection_signature(arguments).is_some()
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
