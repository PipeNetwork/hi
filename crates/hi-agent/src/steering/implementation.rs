//! Implementation tool-call mutation/edit/validation classification and shell
//! command analysis.

use super::intent::contains_any;
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

/// A shell command that deliberately waits before (or while) sampling state —
/// "sleep 300 && du -sh models/" — the natural way an agent watches a slow
/// external process (a download, a long build, a warming server). Re-issuing
/// one verbatim is legitimate as long as its output keeps changing, so the
/// exact-repeat guard exempts it and the result-hash guard catches the static
/// case instead.
pub(crate) fn shell_command_waits(command: &str) -> bool {
    command
        .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
        .any(|word| matches!(word, "sleep" | "wait"))
}

/// Whether a `bash` tool call's command [waits](shell_command_waits).
pub(crate) fn bash_call_waits(arguments: &str) -> bool {
    bash_command(arguments).is_some_and(|command| shell_command_waits(&command))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BashCommandKind {
    Inspection,
    Validation,
    Mutation,
    Background,
    NoProgress,
    Unknown,
}

impl BashCommandKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Inspection => "inspection",
            Self::Validation => "validation",
            Self::Mutation => "mutation",
            Self::Background => "background",
            Self::NoProgress => "no_progress",
            Self::Unknown => "unknown",
        }
    }
}

pub(crate) fn classify_bash_command(command: &str) -> BashCommandKind {
    if shell_command_no_progress_signature(command).is_some() {
        return BashCommandKind::NoProgress;
    }
    let Some(words) = simple_shell_words(command) else {
        return BashCommandKind::Unknown;
    };
    let Some(cmd) = words.first().map(String::as_str) else {
        return BashCommandKind::Unknown;
    };
    if matches!(cmd, "nohup")
        || words
            .iter()
            .any(|word| matches!(word.as_str(), "&" | "disown" | "setsid"))
    {
        return BashCommandKind::Background;
    }
    if shell_command_likely_validates(command) {
        return BashCommandKind::Validation;
    }
    if shell_command_likely_mutates_workspace(command) || shell_command_likely_edits_files(command)
    {
        return BashCommandKind::Mutation;
    }
    if matches!(
        cmd,
        "pwd" | "ls" | "find" | "rg" | "grep" | "cat" | "sed" | "nl" | "head" | "tail" | "git"
    ) {
        return BashCommandKind::Inspection;
    }
    BashCommandKind::Unknown
}

pub(crate) fn shell_command_no_progress_signature(command: &str) -> Option<&'static str> {
    let words = simple_shell_words(command)?;
    match words.as_slice() {
        [cmd] if matches!(cmd.as_str(), "true" | ":") => Some("noop"),
        [cmd] if cmd == "exit" => Some("control-stop"),
        [cmd, code] if cmd == "exit" && code == "0" => Some("control-stop"),
        [cmd, rest @ ..] if cmd == "echo" => {
            let rest = strip_echo_options(rest);
            control_phrase_signature(rest)
        }
        [cmd, arg] if cmd == "printf" => control_phrase_signature(std::slice::from_ref(arg)),
        [cmd, format, rest @ ..]
            if cmd == "printf" && format.contains("%s") && !rest.is_empty() =>
        {
            control_phrase_signature(rest)
        }
        _ => None,
    }
}

pub(crate) fn implementation_tool_call_mutates(name: &str, arguments: &str) -> bool {
    if hi_tools::is_filesystem_mutating(name) {
        return true;
    }
    if name != "bash" {
        return false;
    }
    let Some(command) = bash_command(arguments) else {
        return false;
    };
    shell_command_likely_mutates_workspace(&command)
}

pub(crate) fn implementation_tool_call_substantively_edits(name: &str, arguments: &str) -> bool {
    if matches!(name, "write" | "edit" | "multi_edit" | "apply_patch") {
        return true;
    }
    if name != "bash" {
        return false;
    }
    let Some(command) = bash_command(arguments) else {
        return false;
    };
    shell_command_likely_edits_files(&command)
}

pub(crate) fn implementation_tool_call_validates(name: &str, arguments: &str) -> bool {
    if name != "bash" {
        return false;
    }
    let Some(command) = bash_command(arguments) else {
        return false;
    };
    shell_command_likely_validates(&command)
}

pub(crate) fn implementation_tool_result_landed_mutation(
    name: &str,
    arguments: &str,
    output: &str,
) -> bool {
    if tool_result_is_failure(output) {
        return false;
    }
    if filesystem_mutation_result_landed(name, output) {
        return true;
    }
    if name != "bash" || !implementation_tool_call_mutates(name, arguments) {
        return false;
    }
    bash_result_likely_succeeded(output)
}

pub(crate) fn implementation_tool_result_landed_substantive_edit(
    name: &str,
    arguments: &str,
    output: &str,
) -> bool {
    if tool_result_is_failure(output) {
        return false;
    }
    if filesystem_substantive_edit_result_landed(name, output) {
        return true;
    }
    if name != "bash" || !implementation_tool_call_substantively_edits(name, arguments) {
        return false;
    }
    bash_result_likely_succeeded(output)
}

fn tool_result_is_failure(output: &str) -> bool {
    let trimmed = output.trim_start();
    trimmed.starts_with("Error:")
        || trimmed.starts_with("⚠ refused:")
        || trimmed.contains(hi_tools::markers::EXIT_CODE_PREFIX)
        || trimmed.contains(hi_tools::markers::TIMED_OUT_PREFIX)
}

fn filesystem_mutation_result_landed(name: &str, output: &str) -> bool {
    filesystem_substantive_edit_result_landed(name, output)
}

fn filesystem_substantive_edit_result_landed(name: &str, output: &str) -> bool {
    let trimmed = output.trim_start();
    let lower = trimmed.to_ascii_lowercase();
    match name {
        "write" => lower.starts_with("wrote ") && lower.contains(" bytes to "),
        "edit" => {
            lower.starts_with("edited ") || lower.starts_with("replaced ") && lower.contains(" in ")
        }
        "multi_edit" => lower.starts_with("applied ") && lower.contains(" edits to "),
        "apply_patch" => trimmed
            .lines()
            .any(|line| matches!(line.trim_start().chars().next(), Some('+' | '-' | '~'))),
        _ => false,
    }
}

fn bash_result_likely_succeeded(output: &str) -> bool {
    !tool_result_is_failure(output)
}

fn simple_shell_words(command: &str) -> Option<Vec<String>> {
    let mut chars = command.trim().chars().peekable();
    let mut words = Vec::new();
    let mut current = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    let mut saw_word = false;
    while let Some(ch) = chars.next() {
        if escaped {
            current.push(ch);
            saw_word = true;
            escaped = false;
            continue;
        }
        if in_single {
            if ch == '\'' {
                in_single = false;
            } else {
                current.push(ch);
            }
            saw_word = true;
            continue;
        }
        if in_double {
            if ch == '"' {
                in_double = false;
            } else if ch == '\\' {
                let next = chars.next()?;
                current.push('\\');
                current.push(next);
            } else if matches!(ch, '$' | '`') {
                return None;
            } else {
                current.push(ch);
            }
            saw_word = true;
            continue;
        }
        match ch {
            '\'' => {
                in_single = true;
                saw_word = true;
            }
            '"' => {
                in_double = true;
                saw_word = true;
            }
            '\\' => {
                escaped = true;
                saw_word = true;
            }
            ch if ch.is_whitespace() => {
                if saw_word {
                    words.push(std::mem::take(&mut current));
                    saw_word = false;
                }
            }
            ';' => {
                if saw_word {
                    words.push(std::mem::take(&mut current));
                    saw_word = false;
                }
                if chars.any(|rest| !rest.is_whitespace()) {
                    return None;
                }
                break;
            }
            '&' | '|' | '<' | '>' | '`' | '$' | '(' | ')' | '{' | '}' => return None,
            _ => {
                current.push(ch);
                saw_word = true;
            }
        }
    }
    if escaped || in_single || in_double {
        return None;
    }
    if saw_word {
        words.push(current);
    }
    if words.is_empty() { None } else { Some(words) }
}

fn strip_echo_options(mut words: &[String]) -> &[String] {
    while let Some((first, rest)) = words.split_first()
        && matches!(first.as_str(), "-n" | "-e" | "-E")
    {
        words = rest;
    }
    words
}

fn control_phrase_signature(words: &[String]) -> Option<&'static str> {
    if words.is_empty() {
        return None;
    }
    let phrase = words.join(" ");
    let mut normalized = phrase.trim().to_ascii_lowercase();
    for suffix in ["\\n", "\\r", "\n", "\r"] {
        while normalized.ends_with(suffix) {
            let new_len = normalized.len().saturating_sub(suffix.len());
            normalized.truncate(new_len);
            normalized = normalized.trim_end().to_string();
        }
    }
    let normalized = normalized.trim_matches(|ch: char| {
        ch.is_ascii_whitespace() || matches!(ch, '.' | '!' | '?' | '"' | '\'')
    });
    match normalized {
        "stop" | "quit" | "exit" | "done" | "all done" | "finish" | "finished" | "complete"
        | "completed" => Some("control-stop"),
        _ => None,
    }
}

pub(crate) fn shell_command_likely_mutates_workspace(command: &str) -> bool {
    let compact = command
        .to_ascii_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    contains_any(
        &compact,
        &[
            "cargo init",
            "npm init",
            "pnpm init",
            "yarn init",
            "bun init",
            "cargo add",
            "npm install",
            "pnpm add",
            "yarn add",
            "bun add",
            "mkdir ",
            "touch ",
            "cat >",
            "tee ",
            "sed -i",
            "apply_patch",
            "patch -p",
        ],
    )
}

pub(crate) fn shell_command_likely_edits_files(command: &str) -> bool {
    let compact = command
        .to_ascii_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    contains_any(
        &compact,
        &[
            "cat >",
            "cat <<",
            "tee ",
            "sed -i",
            "perl -i",
            "apply_patch",
            "patch -p",
            "python - <<",
            "python3 - <<",
        ],
    )
}

pub(crate) fn shell_command_likely_validates(command: &str) -> bool {
    let compact = command
        .to_ascii_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    contains_any(
        &compact,
        &[
            "cargo test",
            "cargo check",
            "cargo build",
            "cargo clippy",
            "npm test",
            "npm run test",
            "npm run build",
            "npm run check",
            "npm run lint",
            "pnpm test",
            "pnpm build",
            "pnpm check",
            "pnpm lint",
            "yarn test",
            "yarn build",
            "bun test",
            "bun run build",
            "pytest",
            "python -m pytest",
            "go test",
            "make test",
            "make check",
            "make build",
            "just test",
            "just check",
            "just build",
            "timeout 5s cargo run",
            "cargo run --",
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wait_polls_are_detected_by_sleep_or_wait_words() {
        for command in [
            "sleep 300 && du -sh models/GLM-5.2-MLX-mixed-3_6bit/",
            "sleep 5",
            "wait",
            "cd /repo && sleep 60; ls checkpoints | wc -l",
        ] {
            assert!(shell_command_waits(command), "{command:?}");
        }
        for command in [
            "du -sh models/",
            "cargo build --release",
            "echo done",
            "ls -la",
        ] {
            assert!(!shell_command_waits(command), "{command:?}");
        }
        assert!(bash_call_waits(
            r#"{"command":"sleep 300 && du -sh models/"}"#
        ));
        assert!(!bash_call_waits(r#"{"command":"du -sh models/"}"#));
        assert!(!bash_call_waits(r#"{"path":"not-a-bash-call"}"#));
    }

    #[test]
    fn landed_filesystem_edits_are_result_based() {
        assert!(implementation_tool_result_landed_mutation(
            "write",
            r#"{"path":"a.rs","content":"x"}"#,
            "Wrote 1 bytes to a.rs"
        ));
        assert!(implementation_tool_result_landed_substantive_edit(
            "apply_patch",
            r#"{"patch":"..."}"#,
            "~ updated src/lib.rs (2 changes)\n+ added src/new.rs"
        ));
        assert!(!implementation_tool_result_landed_mutation(
            "edit",
            r#"{"path":"a.rs"}"#,
            "Error: editing a.rs: old string not found"
        ));
    }

    #[test]
    fn failed_bash_edit_does_not_count_as_landed_mutation() {
        let args = r#"{"command":"sed -i s/nope/yep/ src/lib.rs"}"#;
        assert!(!implementation_tool_result_landed_mutation(
            "bash",
            args,
            "sed: src/lib.rs: No such file\n[exit code 2]"
        ));
        assert!(!implementation_tool_result_landed_mutation(
            "bash",
            args,
            "⚠ refused: this command cannot be safely checkpointed"
        ));
        assert!(implementation_tool_result_landed_mutation(
            "bash",
            args,
            "[no output]"
        ));
        assert!(implementation_tool_result_landed_mutation(
            "bash",
            args,
            "[no output — command succeeded (exit 0)]"
        ));
        assert!(!implementation_tool_result_landed_mutation(
            "bash",
            args,
            "[timed out — process killed]"
        ));
    }

    #[test]
    fn no_progress_bash_signature_is_narrow() {
        assert_eq!(
            shell_command_no_progress_signature("echo stop"),
            Some("control-stop")
        );
        assert_eq!(
            shell_command_no_progress_signature("echo quit"),
            Some("control-stop")
        );
        assert_eq!(
            shell_command_no_progress_signature("echo exit"),
            Some("control-stop")
        );
        assert_eq!(
            shell_command_no_progress_signature("printf 'done\\n'"),
            Some("control-stop")
        );
        assert_eq!(shell_command_no_progress_signature("true"), Some("noop"));
        assert_eq!(shell_command_no_progress_signature(":"), Some("noop"));

        assert_eq!(shell_command_no_progress_signature("echo hi"), None);
        assert_eq!(shell_command_no_progress_signature("pwd"), None);
        assert_eq!(shell_command_no_progress_signature("cargo test"), None);
        assert_eq!(
            shell_command_no_progress_signature("echo stop && cargo test"),
            None
        );
        assert_eq!(
            shell_command_no_progress_signature("echo stop > marker.txt"),
            None
        );
    }

    #[test]
    fn bash_command_classification_is_conservative() {
        assert_eq!(
            classify_bash_command("echo stop"),
            BashCommandKind::NoProgress
        );
        assert_eq!(classify_bash_command("true"), BashCommandKind::NoProgress);
        assert_eq!(classify_bash_command("pwd"), BashCommandKind::Inspection);
        assert_eq!(
            classify_bash_command("rg TODO src"),
            BashCommandKind::Inspection
        );
        assert_eq!(
            classify_bash_command("cargo test"),
            BashCommandKind::Validation
        );
        assert_eq!(
            classify_bash_command("mkdir src"),
            BashCommandKind::Mutation
        );
        assert_eq!(
            classify_bash_command("echo stop && cargo test"),
            BashCommandKind::Unknown
        );
        assert_eq!(
            classify_bash_command("echo stop > marker.txt"),
            BashCommandKind::Unknown
        );
        assert_eq!(
            classify_bash_command("./scripts/check.sh"),
            BashCommandKind::Unknown
        );
    }
}
