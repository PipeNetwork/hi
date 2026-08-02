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
    if shell_command_has_known_side_effects(&words) {
        return BashCommandKind::Mutation;
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
        "pwd" | "ls" | "find" | "rg" | "grep" | "cat" | "sed" | "nl" | "head" | "tail"
    ) {
        return BashCommandKind::Inspection;
    }
    if cmd == "git" && git_subcommand_is_read_only(&words[1..]) {
        return BashCommandKind::Inspection;
    }
    BashCommandKind::Unknown
}

/// Whether a `git ...` command's subcommand is read-only. `git` itself is
/// ambiguous — `git status`/`git diff`/`git log` only read, but `git add`,
/// `git commit`, `git reset --hard`, `git checkout --`, `git clean -f`,
/// `git push`, `git pull`, `git merge`, `git rebase`, `git stash`, `git rm`,
/// `git mv`, `git config`, `git fetch`, `git apply`, `git cherry-pick`,
/// `git revert`, `git branch -d`, `git tag v1`, `git remote add`, `git gc`,
/// `git prune`, `git submodule`, `git worktree` all mutate the working tree
/// or `.git`. Only the allowlisted unambiguously read-only subcommands (plus
/// bare `git` with no subcommand, which prints help) are safe to treat as
/// inspection; ambiguous ones (`branch`, `tag`, `remote`, `config`) fall
/// through to `Unknown` so the caller's conservative path (serial run with
/// snapshot/checkpoint) applies.
fn git_subcommand_is_read_only(words: &[String]) -> bool {
    // Skip leading global flags. `-C <dir>` consumes its argument; other
    // global flags (`--git-dir=...`, `-c key=val`) carry `=` and are skipped
    // by the `contains('=')` check.
    let mut i = 0;
    while i < words.len() {
        let word = &words[i];
        if matches!(
            word.as_str(),
            "-C" | "-c" | "--git-dir" | "--work-tree" | "--namespace" | "--config-env"
        ) {
            i += 2; // skip the global option and its argument
            continue;
        }
        if word.starts_with('-') || word.contains('=') {
            i += 1;
            continue;
        }
        // First non-flag word is the subcommand.
        return matches!(
            word.as_str(),
            "status"
                | "diff"
                | "log"
                | "show"
                | "ls-files"
                | "rev-parse"
                | "grep"
                | "blame"
                | "describe"
                | "help"
                | "version"
        );
    }
    // Bare `git` (or only flags) prints help — read-only.
    true
}

/// Some commands have an inspection-shaped verb but can still write files.
/// Keep these out of the concurrent read-only batch and classify them as
/// mutations so confirmation/checkpoint policy remains conservative.
fn shell_command_has_known_side_effects(words: &[String]) -> bool {
    let Some(command) = words.first().map(String::as_str) else {
        return false;
    };
    match command {
        "find" => words.iter().any(|word| {
            matches!(
                word.as_str(),
                "-delete" | "-exec" | "-execdir" | "-ok" | "-okdir"
            )
        }),
        "sed" => words.iter().any(|word| {
            word == "-i"
                || (word.starts_with("-i") && word.len() > 2)
                || word == "--in-place"
                || word.starts_with("--in-place=")
        }),
        "git" => git_command_writes_output(&words[1..]),
        _ => false,
    }
}

fn git_command_writes_output(words: &[String]) -> bool {
    let mut i = 0;
    while i < words.len() {
        let word = &words[i];
        if matches!(
            word.as_str(),
            "-C" | "-c" | "--git-dir" | "--work-tree" | "--namespace" | "--config-env"
        ) {
            i += 2;
            continue;
        }
        if word.starts_with('-') || word.contains('=') {
            i += 1;
            continue;
        }
        let subcommand = word.as_str();
        if !matches!(subcommand, "diff" | "show" | "log") {
            return false;
        }
        return words[i + 1..].iter().any(|arg| {
            arg == "-o"
                || arg.starts_with("-o") && arg.len() > 2
                || arg == "--output"
                || arg.starts_with("--output=")
        });
    }
    false
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
        || trimmed.contains("[exit code ")
        || trimmed.contains("[timed out after ")
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
    if let Some(words) = simple_shell_words(command)
        && shell_command_has_known_side_effects(&words)
    {
        return true;
    }
    let compact = command
        .to_ascii_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if compact.starts_with("find ")
        && contains_any(
            &compact,
            &[" -delete", " -exec ", " -execdir ", " -ok ", " -okdir "],
        )
    {
        return true;
    }
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
            // Git subcommands that mutate the working tree or `.git`. `git`
            // itself is ambiguous, so only these explicit subcommands count.
            "git add",
            "git commit",
            "git reset",
            "git checkout",
            "git switch",
            "git restore",
            "git clean",
            "git push",
            "git pull",
            "git merge",
            "git rebase",
            "git stash",
            "git rm",
            "git mv",
            "git config",
            "git fetch",
            "git apply",
            "git cherry-pick",
            "git revert",
            "git branch -d",
            "git branch -D",
            "git tag -d",
            "git tag -a",
            "git tag ",
            "git remote add",
            "git remote remove",
            "git remote set-url",
            "git gc",
            "git prune",
            "git submodule",
            "git worktree",
            "git init",
            "git clone",
            "git am",
            "git update-index",
            "git notes",
            "git replace",
            "git filter-branch",
            "git archive",
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
            // Lightweight fixtures for canned-provider tests: must not contend
            // on the workspace cargo lock the way `cargo test --help` can.
            "true # validate",
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
    fn lightweight_fixture_validation_command_is_recognized() {
        assert!(shell_command_likely_validates("true # validate"));
        assert!(implementation_tool_call_validates(
            "bash",
            r#"{"command":"true # validate"}"#
        ));
        assert_eq!(
            shell_command_no_progress_signature("true # validate"),
            None,
            "comment-tagged true must not collapse to the bare noop signature"
        );
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

    #[test]
    fn git_read_only_subcommands_are_inspection_but_mutating_ones_are_not() {
        for command in [
            "git status",
            "git status --short",
            "git diff",
            "git diff --cached",
            "git log --oneline -5",
            "git show HEAD",
            "git ls-files",
            "git rev-parse HEAD",
            "git grep TODO",
            "git blame src/lib.rs",
            "git describe --tags",
            "git help",
            "git version",
            "git",
            "git -C /repo status",
            "git --git-dir=/repo/.git status",
            "git --work-tree /repo status",
        ] {
            assert_eq!(
                classify_bash_command(command),
                BashCommandKind::Inspection,
                "{command:?} should be Inspection"
            );
        }
        for command in [
            "git add .",
            "git commit -m x",
            "git reset --hard HEAD",
            "git checkout -- src/lib.rs",
            "git switch main",
            "git restore src/lib.rs",
            "git clean -f",
            "git push origin main",
            "git pull",
            "git merge main",
            "git rebase main",
            "git stash",
            "git rm src/lib.rs",
            "git mv a b",
            "git config user.name x",
            "git fetch origin",
            "git apply patch.diff",
            "git cherry-pick abc123",
            "git revert abc123",
            "git branch -d old",
            "git tag v1.0",
            "git remote add origin url",
            "git gc",
            "git prune",
            "git submodule update",
            "git worktree add ../wt",
            "git init",
            "git clone url",
        ] {
            assert_ne!(
                classify_bash_command(command),
                BashCommandKind::Inspection,
                "{command:?} must not be Inspection (mutates working tree or .git)"
            );
        }
        // Ambiguous subcommands fall through to Unknown (conservative serial
        // path with snapshot/checkpoint), never Inspection.
        for command in ["git branch", "git tag", "git remote"] {
            assert_eq!(
                classify_bash_command(command),
                BashCommandKind::Unknown,
                "{command:?} should be Unknown (ambiguous)"
            );
        }
        // `git config` is ambiguous (read vs write) and the substring matcher
        // can't distinguish, so it's conservatively Mutation — safe, just
        // serial with snapshot/checkpoint.
        assert_eq!(
            classify_bash_command("git config --list"),
            BashCommandKind::Mutation
        );
    }

    #[test]
    fn git_mutations_are_detected_by_implementation_tool_call_mutates() {
        for command in [
            "git add .",
            "git commit -m x",
            "git reset --hard HEAD",
            "git checkout -- src/lib.rs",
            "git clean -f",
            "git push origin main",
            "git pull",
            "git merge main",
            "git rebase main",
            "git stash",
            "git rm src/lib.rs",
            "git mv a b",
            "git config user.name x",
            "git fetch origin",
            "git apply patch.diff",
            "git cherry-pick abc123",
            "git revert abc123",
            "git branch -d old",
            "git tag v1.0",
            "git remote add origin url",
            "git gc",
            "git init",
            "git clone url",
        ] {
            assert!(
                implementation_tool_call_mutates("bash", &format!(r#"{{"command":"{command}"}}"#)),
                "{command:?} should be detected as a mutation"
            );
        }
        for command in [
            "git status",
            "git diff",
            "git log --oneline",
            "git show HEAD",
            "git ls-files",
            "git rev-parse HEAD",
            "git grep TODO",
            "git blame src/lib.rs",
            "git describe --tags",
        ] {
            assert!(
                !implementation_tool_call_mutates("bash", &format!(r#"{{"command":"{command}"}}"#)),
                "{command:?} should NOT be detected as a mutation"
            );
        }
    }

    #[test]
    fn inspection_shaped_commands_with_side_effects_are_not_inspection() {
        for command in [
            "find . -delete",
            "find . -exec rm {} \\;",
            "find . -execdir touch {} \\;",
            "sed --in-place s/old/new/ src/lib.rs",
            "git diff --output=patch.txt",
            "git --work-tree /repo diff --output=patch.txt",
            "git show -o patch.txt HEAD",
            "git log --output log.txt",
        ] {
            assert_ne!(
                classify_bash_command(command),
                BashCommandKind::Inspection,
                "{command:?} must not enter the read-only batch"
            );
            assert!(
                implementation_tool_call_mutates(
                    "bash",
                    &serde_json::json!({"command": command}).to_string(),
                ),
                "{command:?} should trigger mutation policy"
            );
        }
    }
}
