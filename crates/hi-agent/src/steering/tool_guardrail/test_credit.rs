//! Credit a requested-validation gate from test-runner output when the shell
//! exit status is untrustworthy (`cargo test | tail`) or wrapped (`timeout`).

use super::validation::{
    command_runs_tests, first_shell_boundary, is_test_family, strip_process_wrappers,
    validation_matches, validation_requests_metadata,
};

/// True when the shell line invokes a test runner, even if a later `| tail`
/// makes the process exit status untrustworthy.
pub(crate) fn command_invokes_tests(arguments: &str) -> bool {
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
            {
                return false;
            }
            is_test_family(family)
        })
}

/// Cargo/pytest/nextest summaries that prove the suite ran and passed.
pub(crate) fn test_output_shows_success(output: &str) -> bool {
    if nonzero_failed_count(output) {
        return false;
    }
    let mut saw_ok = false;
    for line in output.lines() {
        let lower = line.trim().to_ascii_lowercase();
        if lower.starts_with("test result: failed") {
            return false;
        }
        if lower.starts_with("test result: ok.") {
            saw_ok = true;
        }
    }
    if saw_ok {
        return true;
    }
    let lower = output.to_ascii_lowercase();
    lower.contains(" passed in ")
        || lower.contains("tests run:")
        || ((lower.contains("test suites:")
            || lower.contains("\ntests:")
            || lower.starts_with("tests:"))
            && lower.contains("passed"))
}

/// `1 failed` / `2 failed` is a real failure; `0 failed` is not.
fn nonzero_failed_count(output: &str) -> bool {
    let lower = output.to_ascii_lowercase();
    let mut rest = lower.as_str();
    while let Some(idx) = rest.find(" failed") {
        let prefix = rest[..idx]
            .rsplit(|c: char| !c.is_ascii_digit())
            .next()
            .unwrap_or("");
        if prefix.parse::<u32>().ok().is_some_and(|n| n > 0) {
            return true;
        }
        rest = &rest[idx + 7..];
    }
    false
}

/// Credit a requested-validation gate from either a reliable test-command exit
/// or an unambiguous passing summary in the tool output.
pub(crate) fn tool_result_shows_passing_tests(
    name: &str,
    arguments: &str,
    output: &str,
    validation_observed: bool,
) -> bool {
    if validation_observed && command_runs_tests(arguments) {
        return true;
    }
    name == "bash" && command_invokes_tests(arguments) && test_output_shows_success(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn piped_cargo_test_is_credited_from_a_green_summary() {
        let args = serde_json::json!({"command": "cargo test --quiet 2>&1 | tail -40"}).to_string();
        assert!(command_invokes_tests(&args));
        assert!(!command_runs_tests(&args));
        let green = "running 33 tests\n.................................\ntest result: ok. 33 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.67s\n\nrunning 2 tests\n..\ntest result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.19s\n";
        assert!(test_output_shows_success(green));
        assert!(tool_result_shows_passing_tests("bash", &args, green, false));
        let failed = "test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured\n";
        assert!(!test_output_shows_success(failed));
        assert!(!tool_result_shows_passing_tests(
            "bash", &args, failed, false
        ));
        assert!(!tool_result_shows_passing_tests(
            "bash",
            &args,
            "only the last 40 log lines, no cargo summary",
            false
        ));
        assert!(!tool_result_shows_passing_tests(
            "bash",
            &serde_json::json!({"command": "echo hi"}).to_string(),
            green,
            false
        ));
    }

    #[test]
    fn wrappers_aliases_and_nextest_count_as_tests() {
        for command in [
            "timeout 60 cargo test --quiet",
            "timeout --preserve-status 120s cargo test",
            "nice cargo t --offline",
            "cargo nextest run",
        ] {
            let args = serde_json::json!({"command": command}).to_string();
            assert!(command_invokes_tests(&args), "{command}");
        }
        assert!(command_runs_tests(
            &serde_json::json!({"command": "timeout 60 cargo test"}).to_string()
        ));
        assert!(test_output_shows_success(
            "Summary [ 1.2s] 35 tests run: 35 passed, 0 skipped\n"
        ));
        assert!(!test_output_shows_success(
            "Summary [ 1.2s] 35 tests run: 34 passed, 1 failed, 0 skipped\n"
        ));
        assert!(test_output_shows_success(
            "test result: ok. 33 passed; 0 failed\n"
        ));
    }
}
