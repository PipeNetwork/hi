//! Fail-closed `git` arm of Auto-mode routine bash: unrecognized shapes stay
//! confirm-only. Callers pass wrapper-peeled words with `words[0]` `git` (or a
//! path ending in `/git`).

use std::path::Path;

use super::implementation::{git_subcommand_is_read_only, simple_shell_words, skip_git_globals};

/// Auto may run this shell command without a confirm. Destructive restores
/// (`checkout --`, `reset --hard`, `clean -f`, `restore`) stay confirm-only.
pub fn git_command_is_routine(command: &str) -> bool {
    git_command_is_routine_in(command, None)
}

pub fn git_command_is_routine_in(command: &str, cwd: Option<&Path>) -> bool {
    let Some(words) = simple_shell_words(command) else {
        return false;
    };
    git_words_are_routine(&words, cwd)
}

pub(crate) fn git_words_are_routine(words: &[String], cwd: Option<&Path>) -> bool {
    let Some(cmd) = words.first().map(String::as_str) else {
        return false;
    };
    if cmd != "git" && !cmd.ends_with("/git") {
        return false;
    }
    let after = skip_git_globals(&words[1..]);
    if git_subcommand_is_read_only(after) {
        return true;
    }
    let Some(verb) = after.first().map(String::as_str) else {
        return false;
    };
    let args = &after[1..];
    match verb {
        "add" | "commit" | "pull" | "fetch" => true,
        "worktree" => args.first().map(String::as_str) == Some("list"),
        "checkout" | "switch" => is_routine_branch_switch(args, cwd),
        "stash" => is_routine_stash(args),
        _ => false,
    }
}

/// Exact spellings only, so clusters (`-qf`) and abbreviations (`--fo`) fail closed.
const BRANCH_SWITCH_BENIGN_FLAGS: &[&str] = &[
    "-q",
    "--quiet",
    "-d",
    "--detach",
    "-t",
    "--track",
    "--no-track",
    "--guess",
    "--no-guess",
    "--progress",
    "--no-progress",
    "--recurse-submodules",
    "--no-recurse-submodules",
];

fn is_routine_branch_switch(args: &[String], cwd: Option<&Path>) -> bool {
    let mut operands = 0;
    let mut it = args.iter().map(String::as_str);
    while let Some(word) = it.next() {
        match word {
            "-b" | "-B" | "-c" | "-C" | "--orphan" => {
                it.next();
            }
            "-" => operands += 1,
            _ if BRANCH_SWITCH_BENIGN_FLAGS.contains(&word) => {}
            _ if word.starts_with('-') => return false,
            _ => {
                if operand_reads_as_path(word, cwd) {
                    return false;
                }
                operands += 1;
            }
        }
    }
    operands <= 1
}

fn is_routine_stash(args: &[String]) -> bool {
    let mut it = args.iter().map(String::as_str);
    let subcommand = loop {
        match it.next() {
            Some("-m" | "--message") => {
                it.next();
            }
            Some("-q" | "--quiet") => {}
            Some(word) if word.starts_with('-') => return false,
            other => break other,
        }
    };
    matches!(
        subcommand,
        None | Some("push" | "save" | "pop" | "apply" | "list" | "show" | "branch")
    )
}

fn operand_reads_as_path(op: &str, cwd: Option<&Path>) -> bool {
    if op.contains('/')
        || op
            .split('/')
            .any(|c| c.starts_with('.') || c.ends_with(".lock"))
        || op.ends_with('/')
        || op.ends_with('.')
        || op.contains("..")
        || op.contains("@{")
        || op.contains(['~', '^', ':', '\\'])
        || op.contains(char::is_whitespace)
    {
        return true;
    }
    if op.starts_with('/') || op.contains(['*', '?', '[']) {
        return true;
    }
    if cwd.is_some_and(|dir| dir.join(op).exists()) {
        return true;
    }
    let Some((stem, ext)) = op.rsplit_once('.') else {
        return false;
    };
    let stem_leaf = stem.rsplit('/').next().unwrap_or(stem);
    if ext == "x"
        && stem_leaf.bytes().any(|b| b.is_ascii_digit())
        && stem_leaf.bytes().all(|b| b.is_ascii_digit() || b == b'.')
    {
        return false;
    }
    (1..=5).contains(&ext.len())
        && ext.chars().all(|c| c.is_ascii_alphanumeric())
        && ext.chars().any(|c| c.is_ascii_alphabetic())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::steering::is_destructive_git_restore;

    fn routine(command: &str) -> bool {
        git_command_is_routine(command) && !is_destructive_git_restore(command)
    }

    #[test]
    fn allowlisted_git_is_routine() {
        for command in [
            "git status",
            "git diff",
            "git add -A",
            "git commit -m 'wip'",
            "git pull",
            "git fetch origin",
            "git worktree list",
            "git checkout main",
            "git switch -c topic",
            "git stash",
            "git stash pop",
        ] {
            assert!(routine(command), "{command}");
        }
    }

    #[test]
    fn discards_and_unknown_shapes_fail_closed() {
        for command in [
            "git checkout -- src/lib.rs",
            "git restore src/lib.rs",
            "git reset --hard",
            "git clean -f",
            "git push",
            "git rebase main",
            "git stash drop",
            "git checkout -qf main",
            "rm -rf src",
        ] {
            assert!(!routine(command), "{command}");
        }
        assert!(git_command_is_routine("git"), "bare git is help, read-only");
        assert!(
            !routine("git checkout src/foo"),
            "slash operands are path restores"
        );
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Makefile"), "all:\n").unwrap();
        assert!(
            !git_command_is_routine_in("git checkout Makefile", Some(dir.path())),
            "a worktree file is a path restore"
        );
        assert!(git_command_is_routine_in(
            "git checkout main",
            Some(dir.path())
        ));
    }
}
