const VALIDATION_FAMILIES: &[(&str, &str)] = &[
    ("cargo nextest", "cargo:test"),
    ("cargo test", "cargo:test"),
    ("cargo t", "cargo:test"),
    ("cargo check", "cargo:check"),
    ("cargo build", "cargo:build"),
    ("cargo clippy", "cargo:clippy"),
    ("cargo run", "cargo:run"),
    ("npm run test", "npm:test"),
    ("npm run build", "npm:build"),
    ("npm run check", "npm:check"),
    ("npm run lint", "npm:lint"),
    ("npm test", "npm:test"),
    ("pnpm test", "pnpm:test"),
    ("pnpm build", "pnpm:build"),
    ("pnpm check", "pnpm:check"),
    ("pnpm lint", "pnpm:lint"),
    ("yarn test", "yarn:test"),
    ("yarn build", "yarn:build"),
    ("bun run build", "bun:build"),
    ("bun test", "bun:test"),
    ("python3 -m unittest", "unittest"),
    ("python -m unittest", "unittest"),
    ("python -m pytest", "pytest"),
    ("python3 -m pytest", "pytest"),
    ("python -c", "python:inline"),
    ("python3 -c", "python:inline"),
    ("python -m", "python:module"),
    ("python3 -m", "python:module"),
    ("python", "python:script"),
    ("python3", "python:script"),
    ("pypy", "python:script"),
    ("pypy3", "python:script"),
    ("node -e", "node:inline"),
    ("node --eval", "node:inline"),
    ("node", "node:script"),
    ("ruby", "ruby:script"),
    ("perl", "perl:script"),
    ("php", "php:script"),
    ("pytest", "pytest"),
    ("go test", "go:test"),
    ("go run", "go:run"),
    ("make test", "make:test"),
    ("make check", "make:check"),
    ("make build", "make:build"),
    ("just test", "just:test"),
    ("just check", "just:check"),
    ("just build", "just:build"),
];

/// Whether a call can produce a result tracked by the semantic result guard.
/// Result status and landed effects still decide eligibility after execution.
pub(crate) fn tool_result_hash_guard_applies(name: &str, arguments: &str) -> bool {
    matches!(name, "read" | "list" | "grep" | "glob" | "bash_output")
        || (name == "bash"
            && (super::super::implementation::bash_call_waits(arguments)
                || super::super::implementation::bash_bounded_execution_probe(arguments).is_some()
                || super::super::implementation::bash_inspection_signature(arguments).is_some()
                || bash_validation_scope(arguments).is_some()))
}

/// Canonical validator family plus its own invocation arguments. Shell stages
/// used only to present the result (`grep`, `head`, and friends) are outside
/// the scope; Cargo package, manifest, and target selectors remain inside it.
pub(super) fn bash_validation_scope(arguments: &str) -> Option<String> {
    let command = super::super::implementation::bash_command(arguments)?;
    let (start, phrase_end, family) = validation_match(&command)?;
    let suffix = &command[phrase_end..];
    let end = first_shell_boundary(suffix);
    // Retain setup that changes the validator's meaning, especially `cd` and
    // environment selection. Only presentation stages *after* the validator
    // are ignored. Otherwise identical output from two subprojects could be
    // incorrectly conflated.
    let executable_start = command_token_start(&command, start);
    let context = normalize_validation_context(&command[..executable_start]);
    let executable_prefix = &command[executable_start..start];
    let arguments = normalize_validation_arguments(family, &suffix[..end]);
    Some(format!(
        "{family}@{executable_prefix}|{context}|{arguments}"
    ))
}

/// A successful shell exit is validation evidence only when it reflects the
/// validator rather than a later pipeline/filter command. The call remains a
/// validation observation for convergence even when its green status is not
/// trustworthy.
pub(crate) fn validation_exit_status_is_reliable(name: &str, arguments: &str) -> bool {
    if name != "bash" {
        return true;
    }
    let Some(command) = super::super::implementation::bash_command(arguments) else {
        return false;
    };
    let Some((_, phrase_end, family)) = validation_match(&command) else {
        return false;
    };
    !validation_requests_metadata(family, &command[phrase_end..])
        && shell_command_preserves_exit_status(&command)
}

fn validation_match(command: &str) -> Option<(usize, usize, &'static str)> {
    validation_matches(command)
        .into_iter()
        .min_by_key(|(start, _, _)| *start)
}

pub(super) fn validation_matches(command: &str) -> Vec<(usize, usize, &'static str)> {
    let lower = command.to_ascii_lowercase();
    VALIDATION_FAMILIES
        .iter()
        .filter_map(|(needle, family)| {
            let first_word = needle.split_whitespace().next()?;
            lower.match_indices(first_word).find_map(|(start, _)| {
                let end = validation_phrase_end(&lower, start, needle)?;
                (validation_is_at_command_position(command, start)
                    && validation_family_applies(command, end, family))
                .then_some((start, end, *family))
            })
        })
        .collect()
}

fn validation_family_applies(command: &str, phrase_end: usize, family: &str) -> bool {
    let extensions: &[&str] = match family {
        "python:script" => &[".py", ".pyc"],
        "node:script" => &[".js", ".mjs", ".cjs"],
        "ruby:script" => &[".rb"],
        "perl:script" => &[".pl", ".pm"],
        "php:script" => &[".php"],
        _ => return true,
    };
    let suffix = &command[phrase_end..];
    let words = suffix[..first_shell_boundary(suffix)]
        .split_whitespace()
        .collect::<Vec<_>>();
    words.first().is_some_and(|word| !word.starts_with('-'))
        && words.iter().any(|word| {
            let path = word.trim_matches(|character| matches!(character, ';' | ',' | '\'' | '"'));
            extensions.iter().any(|extension| path.ends_with(extension))
        })
}

fn validation_phrase_end(command: &str, start: usize, phrase: &str) -> Option<usize> {
    let boundary = |byte: u8| byte.is_ascii_whitespace() || b"/;&|()".contains(&byte);
    if start > 0 && !boundary(command.as_bytes()[start - 1]) {
        return None;
    }
    let quoted_executable = quoted_executable_delimiter(command, start);
    let mut cursor = start;
    for (index, word) in phrase.split_whitespace().enumerate() {
        if index > 0 {
            let whitespace = command[cursor..]
                .bytes()
                .take_while(u8::is_ascii_whitespace)
                .count();
            if whitespace == 0 {
                return None;
            }
            cursor += whitespace;
        }
        let end = cursor.checked_add(word.len())?;
        if command.get(cursor..end)? != word {
            return None;
        }
        cursor = end;
        if index == 0
            && let Some(delimiter) = quoted_executable
        {
            if command.as_bytes().get(cursor) != Some(&delimiter) {
                return None;
            }
            cursor += 1;
        }
    }
    (cursor == command.len() || boundary(command.as_bytes()[cursor])).then_some(cursor)
}

fn quoted_executable_delimiter(command: &str, start: usize) -> Option<u8> {
    let token_start = command_token_start(command, start);
    let token_prefix = command.get(token_start..start)?;
    let delimiter = *token_prefix.as_bytes().first()?;
    if !matches!(delimiter, b'\'' | b'"') || token_prefix.as_bytes()[1..].contains(&delimiter) {
        return None;
    }
    Some(delimiter)
}

pub(super) fn first_shell_boundary(source: &str) -> usize {
    let mut single_quoted = false;
    let mut double_quoted = false;
    let mut escaped = false;
    let bytes = source.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if escaped {
            escaped = false;
        } else if byte == b'\\' && !single_quoted {
            escaped = true;
        } else if byte == b'\'' && !double_quoted {
            single_quoted = !single_quoted;
        } else if byte == b'"' && !single_quoted {
            double_quoted = !double_quoted;
        } else if !single_quoted && !double_quoted {
            if byte == b'#' && (index == 0 || bytes[index - 1].is_ascii_whitespace()) {
                return index;
            }
            if matches!(byte, b'|' | b';' | b'\n' | b'&') {
                return index;
            }
        }
        index += 1;
    }
    source.len()
}

pub(super) fn validation_requests_metadata(family: &str, suffix: &str) -> bool {
    suffix[..first_shell_boundary(suffix)]
        .split_whitespace()
        .map(shell_word_before_redirection)
        .map(|word| word.trim_matches(['\'', '"']))
        .any(|word| {
            matches!(word, "-h" | "--help" | "-V" | "--version")
                || (family.starts_with("cargo:") && cargo_metadata_flag_cluster(word))
        })
}

fn cargo_metadata_flag_cluster(word: &str) -> bool {
    let Some(flags) = word
        .strip_prefix('-')
        .filter(|flags| !flags.starts_with('-'))
    else {
        return false;
    };
    flags.len() > 1
        && flags
            .chars()
            .all(|flag| matches!(flag, 'q' | 'v' | 'h' | 'V'))
        && flags.chars().any(|flag| matches!(flag, 'h' | 'V'))
}

fn shell_word_before_redirection(word: &str) -> &str {
    let mut single_quoted = false;
    let mut double_quoted = false;
    let mut escaped = false;
    for (index, byte) in word.bytes().enumerate() {
        if escaped {
            escaped = false;
        } else if byte == b'\\' && !single_quoted {
            escaped = true;
        } else if byte == b'\'' && !double_quoted {
            single_quoted = !single_quoted;
        } else if byte == b'"' && !single_quoted {
            double_quoted = !double_quoted;
        } else if !single_quoted && !double_quoted && matches!(byte, b'<' | b'>') {
            return &word[..index];
        }
    }
    word
}

fn normalize_validation_arguments(family: &str, arguments: &str) -> String {
    let mut words = arguments.split_whitespace().peekable();
    let mut normalized = Vec::new();
    let mut cargo_options = Vec::new();
    while let Some(word) = words.next() {
        if word == "--" {
            normalized.push(word.to_string());
            normalized.extend(words.map(str::to_string));
            break;
        }
        if word.contains(['<', '>']) {
            continue;
        }
        if family.starts_with("cargo:") {
            if matches!(word, "-q" | "--quiet" | "-v" | "--verbose")
                || word.starts_with("--color=")
                || word.starts_with("--message-format=")
            {
                continue;
            }
            if matches!(word, "--color" | "--message-format") {
                let _ = words.next();
                continue;
            }
            if matches!(word, "-p" | "--package") {
                if let Some(value) = words.next() {
                    cargo_options.push(format!("--package={value}"));
                }
                continue;
            }
            if let Some(option) = [
                "--manifest-path",
                "--target",
                "--features",
                "--exclude",
                "--test",
                "--bin",
                "--example",
                "--bench",
                "--profile",
            ]
            .iter()
            .find(|option| **option == word)
            {
                if let Some(value) = words.next() {
                    cargo_options.push(format!("{option}={value}"));
                }
                continue;
            }
            if word.starts_with("-p") && word.len() > 2 {
                cargo_options.push(format!("--package={}", &word[2..]));
                continue;
            }
            if [
                "--package=",
                "--manifest-path=",
                "--target=",
                "--features=",
                "--exclude=",
                "--test=",
                "--bin=",
                "--example=",
                "--bench=",
                "--profile=",
            ]
            .iter()
            .any(|prefix| word.starts_with(prefix))
                || matches!(
                    word,
                    "--workspace"
                        | "--all"
                        | "--lib"
                        | "--bins"
                        | "--tests"
                        | "--benches"
                        | "--all-targets"
                        | "--all-features"
                        | "--no-default-features"
                        | "--release"
                        | "--locked"
                        | "--offline"
                        | "--frozen"
                )
            {
                cargo_options.push(word.to_string());
                continue;
            }
        }
        normalized.push(word.to_string());
    }
    cargo_options.sort_unstable();
    cargo_options.dedup();
    cargo_options.extend(normalized);
    cargo_options.join(" ")
}

fn normalize_validation_context(context: &str) -> String {
    context
        .split([';', '\n'])
        .flat_map(|part| part.split("&&"))
        .map(|part| part.trim().trim_start_matches(['(', '{']).trim())
        .filter(|part| {
            !part.is_empty()
                && (part.starts_with("cd ")
                    || part.starts_with("pushd ")
                    || part.starts_with("export ")
                    || part.starts_with("env ")
                    || part
                        .split_whitespace()
                        .all(|word| !word.is_empty() && word.contains('=')))
        })
        .map(|part| part.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect::<Vec<_>>()
        .join(" && ")
}

fn command_token_start(command: &str, start: usize) -> usize {
    command[..start]
        .rfind(|character: char| character.is_ascii_whitespace() || ";|&()".contains(character))
        .map_or(0, |index| index + 1)
}

fn validation_is_at_command_position(command: &str, start: usize) -> bool {
    let prefix = &command[..start];
    if prefix.contains("$(")
        || prefix.contains('`')
        || (!plain_shell_position(prefix) && quoted_executable_delimiter(command, start).is_none())
    {
        return false;
    }
    let token_start = command_token_start(command, start);
    if prefix[token_start..].contains(['=', '<', '>']) {
        return false;
    }
    let before_token = prefix[..token_start].trim_end_matches([' ', '\t', '\r']);
    if before_token.is_empty()
        || before_token.ends_with("&&")
        || before_token.ends_with(';')
        || before_token.ends_with('\n')
    {
        return true;
    }
    let current_segment = before_token
        .rsplit_once("&&")
        .map_or(before_token, |(_, segment)| segment)
        .trim();
    let mut words = current_segment.split_whitespace();
    let first = words.next();
    first.is_some_and(|word| word == "env" || word.contains('='))
        && words.all(|word| word.contains('='))
}

fn plain_shell_position(prefix: &str) -> bool {
    let mut single_quoted = false;
    let mut double_quoted = false;
    let mut escaped = false;
    let mut comment = false;
    for (index, byte) in prefix.bytes().enumerate() {
        if comment {
            comment = byte != b'\n';
            continue;
        }
        if escaped {
            escaped = false;
        } else if byte == b'\\' && !single_quoted {
            escaped = true;
        } else if byte == b'\'' && !double_quoted {
            single_quoted = !single_quoted;
        } else if byte == b'"' && !single_quoted {
            double_quoted = !double_quoted;
        } else if byte == b'#'
            && !single_quoted
            && !double_quoted
            && (index == 0 || prefix.as_bytes()[index - 1].is_ascii_whitespace())
        {
            comment = true;
        }
    }
    !single_quoted && !double_quoted && !comment
}

fn shell_command_preserves_exit_status(source: &str) -> bool {
    let mut single_quoted = false;
    let mut double_quoted = false;
    let mut escaped = false;
    let bytes = source.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if escaped {
            escaped = false;
            index += 1;
            continue;
        }
        if byte == b'\\' && !single_quoted {
            escaped = true;
        } else if byte == b'\'' && !double_quoted {
            single_quoted = !single_quoted;
        } else if byte == b'"' && !single_quoted {
            double_quoted = !double_quoted;
        } else if !single_quoted && !double_quoted {
            if byte == b'#' && (index == 0 || bytes[index - 1].is_ascii_whitespace()) {
                if bytes[index..].contains(&b'\n') {
                    return false;
                }
                break;
            }
            if byte == b'|' {
                return false;
            }
            if byte == b';' || byte == b'\n' {
                return false;
            }
            if byte == b'&' {
                if bytes.get(index + 1) == Some(&b'&') {
                    index += 1;
                } else if bytes.get(index + 1) != Some(&b'>')
                    && index.checked_sub(1).and_then(|i| bytes.get(i)) != Some(&b'>')
                {
                    return false;
                }
            }
        }
        index += 1;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments(command: &str) -> String {
        serde_json::json!({ "command": command }).to_string()
    }

    #[test]
    fn quoted_inline_programs_keep_their_complete_scope() {
        let foo = arguments(r#"python3 -c "import app; assert app.foo()""#);
        let bar = arguments(r#"python3 -c "import app; assert app.bar()""#);
        let piped_expression = arguments(r#"node -e "assert.equal(actual | 1, expected)""#);

        assert_ne!(bash_validation_scope(&foo), bash_validation_scope(&bar));
        assert!(
            bash_validation_scope(&piped_expression)
                .is_some_and(|scope| scope.contains("actual | 1"))
        );
    }

    #[test]
    fn grouped_cwd_and_executable_identity_remain_in_scope() {
        let api = arguments("(cd crates/api && cargo clippy)");
        let web = arguments("(cd crates/web && cargo clippy)");
        let venv_a = arguments("/venv-a/bin/python check.py");
        let venv_b = arguments("/venv-b/bin/python check.py");

        assert_ne!(bash_validation_scope(&api), bash_validation_scope(&web));
        assert_ne!(
            bash_validation_scope(&venv_a),
            bash_validation_scope(&venv_b)
        );
    }

    #[test]
    fn embedded_paths_and_metadata_commands_are_not_green_validation() {
        for command in [
            "FOO=/usr/bin/cargo clippy",
            "2>/tmp/cargo clippy",
            "cargo clippy --help",
            "cargo clippy --help>/dev/null",
            "cargo test -h>help.txt",
            "cargo test -qh",
            "cargo test --version",
            "pytest --version 2>/dev/null",
            "pytest -h",
            "go test --help",
        ] {
            assert!(
                !validation_exit_status_is_reliable("bash", &arguments(command)),
                "must not accept metadata or an embedded validator path: {command}"
            );
        }
    }

    #[test]
    fn supported_direct_script_runtimes_have_guard_scopes() {
        for command in [
            "pypy check.py",
            "pypy3 check.py",
            "perl check.pl",
            "php check.php",
        ] {
            assert!(
                bash_validation_scope(&arguments(command)).is_some(),
                "missing semantic scope for {command}"
            );
        }
    }

    #[test]
    fn runtime_metadata_does_not_shadow_a_later_validator() {
        let api = arguments("python --version && cargo clippy --package api");
        let web = arguments("python --version && cargo clippy --package web");
        let npm = arguments("node --version && npm test");

        assert_ne!(bash_validation_scope(&api), bash_validation_scope(&web));
        assert!(bash_validation_scope(&api).is_some_and(|scope| scope.contains("cargo:clippy")));
        assert!(bash_validation_scope(&npm).is_some_and(|scope| scope.contains("npm:test")));
    }

    #[test]
    fn quoted_executable_paths_are_validation_scopes() {
        let first =
            arguments(r#""/opt/toolchains/nightly/bin/cargo" clippy 2>&1 | grep line-1605"#);
        let second =
            arguments(r#""/opt/toolchains/nightly/bin/cargo" clippy 2>&1 | grep line-1606"#);
        let python = arguments(r#""/opt/venv/bin/python" check.py"#);

        assert_eq!(
            bash_validation_scope(&first),
            bash_validation_scope(&second)
        );
        assert!(bash_validation_scope(&first).is_some_and(|scope| scope.contains("cargo:clippy")));
        assert!(
            bash_validation_scope(&python).is_some_and(|scope| scope.contains("python:script"))
        );
    }
}

pub(crate) fn is_validation_command(command: &str) -> bool {
    validation_match(command).is_some()
}

/// Tests are a distinct obligation; compilation alone does not satisfy it.
pub(crate) fn command_runs_tests(arguments: &str) -> bool {
    let Some(command) = super::super::implementation::bash_command(arguments) else {
        return false;
    };
    let command = strip_process_wrappers(&command);
    validation_matches(command)
        .into_iter()
        .any(|(_, end, family)| {
            let suffix = &command[end..];
            let flags = &suffix[..first_shell_boundary(suffix)];
            if validation_requests_metadata(family, suffix)
                || flags.split_whitespace().any(|word| {
                    matches!(
                        word.trim_matches(['\'', '"']),
                        "--no-run" | "--list" | "--collect-only" | "--collectonly" | "--listTests"
                    )
                })
                || !shell_command_preserves_exit_status(&command)
            {
                return false;
            }
            is_test_family(family)
        })
}

pub(super) fn is_test_family(family: &str) -> bool {
    matches!(
        family,
        "cargo:test"
            | "npm:test"
            | "pnpm:test"
            | "yarn:test"
            | "bun:test"
            | "pytest"
            | "unittest"
            | "go:test"
            | "make:test"
            | "just:test"
    )
}

pub(super) fn strip_process_wrappers(command: &str) -> &str {
    let mut rest = command.trim_start();
    for _ in 0..4 {
        let Some(first) = rest.split_whitespace().next() else {
            return rest;
        };
        let name = first.trim_matches(['\'', '"']);
        let after = rest[first.len()..].trim_start();
        rest = match name.to_ascii_lowercase().as_str() {
            "nice" | "nohup" | "command" | "time" => after,
            "timeout" | "gtimeout" => skip_timeout_args(after),
            _ => return rest,
        };
    }
    rest
}

fn skip_timeout_args(mut rest: &str) -> &str {
    loop {
        let Some(word) = rest.split_whitespace().next() else {
            return rest;
        };
        if word.starts_with('-') {
            rest = rest[word.len()..].trim_start();
            if matches!(word, "-k" | "--kill-after" | "-s" | "--signal") {
                if let Some(arg) = rest.split_whitespace().next() {
                    rest = rest[arg.len()..].trim_start();
                }
            }
            continue;
        }
        let duration = word.trim_end_matches(['s', 'm', 'h', 'd', 'S', 'M', 'H']);
        if !duration.is_empty() && duration.chars().all(|c| c.is_ascii_digit() || c == '.') {
            return rest[word.len()..].trim_start();
        }
        return rest;
    }
}

#[cfg(test)]
mod test_obligation_tests {
    use super::*;
    #[test]
    fn tests_must_execute_and_propagate_failure() {
        for command in [
            "cargo test",
            "cargo check && cargo test --quiet",
            "python3 -m unittest test_answer",
            "npm test",
        ] {
            assert!(
                command_runs_tests(&serde_json::json!({"command":command}).to_string()),
                "{command}"
            );
        }
        for command in [
            "cargo check",
            "true # validate",
            "echo cargo test",
            "echo 'cargo test'",
            "cargo test --no-run",
            "cargo test -- --list",
            "pytest --collect-only",
            "cargo test || true",
            "cargo test | head",
            "cargo test --help",
        ] {
            assert!(
                !command_runs_tests(&serde_json::json!({"command":command}).to_string()),
                "{command}"
            );
        }
        assert!(!is_validation_command("true # validate"));
        assert!(!is_validation_command("echo cargo test"));
    }
}
