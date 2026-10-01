//! Which files a `/review` audit turn actually read.
//!
//! A live audit of 62 uncommitted files read 11 of them and reported
//! "finding: none" with a clean exit; nothing in the output said so. The
//! turn loop feeds every tool call of an audit turn through
//! [`files_read`] and the drive compares the result with the files in
//! scope, so the report can say "read 11/62" and ask for the rest.
//!
//! Counted: `read` (`path`, `paths`, or a `workspace://` URI), `grep` on a
//! single file, a shell command whose program prints file contents (`cat`,
//! `head`, `sed`, `rg`, …) and names a file, and the files whose diff
//! headers appear in a shell command's output (`git diff`, `git show`) or
//! the `diff` tool's. Naming a file is not enough: `echo src/a.rs` and
//! `git diff -- a.rs b.rs` whose output was truncated before `b.rs`'s
//! header did not show that file. Directory listings, `glob`, and
//! `repo_map` are not reads.

use std::path::{Component, Path};

/// Programs whose file arguments are the contents the model was shown.
/// Anything else (`echo`, `git`, `python`) is not a read from its
/// arguments; a diff counts only through headers in the output.
const CONTENT_PROGRAMS: &[&str] = &[
    "bat", "cat", "grep", "head", "less", "more", "nl", "rg", "sed", "tail",
];

/// Words that run the rest of the segment as the program.
const COMMAND_WRAPPERS: &[&str] = &[
    "command", "env", "exec", "nice", "nohup", "sudo", "time", "xargs",
];

/// Workspace-relative paths (`/` separators) of files under `root` that the
/// call `name(arguments)` read, judged from the arguments and, for shell
/// commands, the diff headers in `content` (the output as the model saw
/// it, so truncated hunks do not count). `complete` is false when that
/// output was clipped: file arguments are then ignored, because a truncated
/// `cat a.rs b.rs` may have shown only `a.rs`, while diff headers still
/// name exactly the files present in what was shown. Deduplicated, in call
/// order.
pub fn files_read(
    root: &Path,
    name: &str,
    arguments: &str,
    content: &str,
    complete: bool,
) -> Vec<String> {
    let args: serde_json::Value = serde_json::from_str(arguments).unwrap_or_default();
    let mut out = Vec::new();
    let mut note = |candidate: &str| {
        if let Some(path) = workspace_file(root, candidate)
            && !out.contains(&path)
        {
            out.push(path);
        }
    };
    match name {
        "read" => {
            if let Some(path) = args.get("path").and_then(|v| v.as_str()) {
                note(path);
            }
            if let Some(paths) = args.get("paths").and_then(|v| v.as_array()) {
                for path in paths.iter().filter_map(|v| v.as_str()) {
                    note(path);
                }
            }
            if let Some(uri) = args.get("uri").and_then(|v| v.as_str())
                && let Some(path) = uri.strip_prefix("workspace://")
            {
                note(path);
            }
        }
        "grep" => {
            if let Some(path) = args.get("path").and_then(|v| v.as_str()) {
                note(path);
            }
        }
        "bash" => {
            if complete && let Some(command) = args.get("command").and_then(|v| v.as_str()) {
                for token in command_tokens(command) {
                    note(token);
                }
            }
            for path in diff_header_paths(content) {
                note(path);
            }
        }
        "diff" => {
            for path in diff_header_paths(content) {
                note(path);
            }
        }
        _ => {}
    }
    out
}

/// `candidate` as a workspace-relative path when it names an existing file
/// under `root`: absolute paths inside the workspace are made relative,
/// `./` and quotes are stripped, a trailing `:line[:col]` is dropped, and
/// anything escaping the root (`..`) is rejected.
pub fn workspace_file(root: &Path, candidate: &str) -> Option<String> {
    let raw = candidate.trim().trim_matches(['`', '"', '\'', ',', ';']);
    let raw = strip_line_suffix(raw);
    if raw.is_empty() || raw == "-" {
        return None;
    }
    let path = Path::new(raw);
    let relative = if path.is_absolute() {
        path.strip_prefix(root).ok()?.to_path_buf()
    } else {
        path.to_path_buf()
    };
    if relative
        .components()
        .any(|part| !matches!(part, Component::Normal(_) | Component::CurDir))
    {
        return None;
    }
    let normalized: Vec<String> = relative
        .components()
        .filter_map(|part| match part {
            Component::Normal(name) => Some(name.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect();
    if normalized.is_empty() {
        return None;
    }
    let joined = normalized.join("/");
    root.join(&joined).is_file().then_some(joined)
}

/// `src/a.rs:12:5` -> `src/a.rs`; a Windows drive letter is not a line.
fn strip_line_suffix(raw: &str) -> &str {
    let mut path = raw;
    while let Some((head, tail)) = path.rsplit_once(':') {
        if !tail.is_empty() && tail.chars().all(|ch| ch.is_ascii_digit()) && !head.is_empty() {
            path = head;
        } else {
            break;
        }
    }
    path
}

/// File arguments of each segment whose program prints file contents.
/// Option flags are dropped. `git` is omitted: a diff counts through the
/// headers in the output, so a truncated `git diff -- a.rs b.rs` does not
/// mark `b.rs` read.
fn command_tokens(command: &str) -> impl Iterator<Item = &str> {
    command
        .split(['|', ';', '&', '(', ')', '<', '>', '\n'])
        .flat_map(segment_arguments)
}

fn segment_arguments(segment: &str) -> Vec<&str> {
    let mut words = segment
        .split_whitespace()
        .map(|word| word.trim_matches(['"', '\'', '`']))
        .filter(|word| !word.is_empty())
        .skip_while(|word| COMMAND_WRAPPERS.contains(word) || is_assignment(word));
    let Some(program) = words.next() else {
        return Vec::new();
    };
    let program = program.rsplit('/').next().unwrap_or(program);
    if !CONTENT_PROGRAMS.contains(&program) {
        return Vec::new();
    }
    words.filter(|word| !word.starts_with('-')).collect()
}

/// `NAME=value` before a program.
fn is_assignment(word: &str) -> bool {
    word.split_once('=').is_some_and(|(name, _)| {
        !name.is_empty()
            && name
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
    })
}

/// Paths named by unified-diff headers in `content`: `diff --git a/X b/X`
/// (the `b/` side, so a rename counts as its new name) and `+++ b/X`.
fn diff_header_paths(content: &str) -> impl Iterator<Item = &str> {
    content.lines().filter_map(|line| {
        if let Some(rest) = line.strip_prefix("diff --git ") {
            return rest.rsplit_once(" b/").map(|(_, path)| path.trim());
        }
        if let Some(rest) = line.strip_prefix("+++ b/") {
            return Some(rest.split('\t').next().unwrap_or(rest).trim());
        }
        None
    })
}

/// `scope` files not in `read`, in scope order.
pub fn unread_files(scope: &[String], read: &[String]) -> Vec<String> {
    scope
        .iter()
        .filter(|file| !read.contains(file))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/a.rs"), "fn a() {}\n").unwrap();
        std::fs::write(dir.path().join("src/b.rs"), "fn b() {}\n").unwrap();
        std::fs::write(dir.path().join("README.md"), "# x\n").unwrap();
        dir
    }

    fn json(value: serde_json::Value) -> String {
        value.to_string()
    }

    #[test]
    fn read_tool_paths_and_uris_count_missing_and_escaping_paths_do_not() {
        let dir = root();
        let root = dir.path();
        assert_eq!(
            files_read(
                root,
                "read",
                &json(serde_json::json!({"path": "./src/a.rs"})),
                "",
                true,
            ),
            vec!["src/a.rs"]
        );
        assert_eq!(
            files_read(
                root,
                "read",
                &json(
                    serde_json::json!({"paths": ["src/a.rs", "src/b.rs", "src/a.rs", "nope.rs"]})
                ),
                "",
                true,
            ),
            vec!["src/a.rs", "src/b.rs"]
        );
        assert_eq!(
            files_read(
                root,
                "read",
                &json(serde_json::json!({"uri": "workspace://README.md"})),
                "",
                true,
            ),
            vec!["README.md"]
        );
        let absolute = root.join("src/b.rs").display().to_string();
        assert_eq!(
            files_read(
                root,
                "read",
                &json(serde_json::json!({"path": absolute})),
                "",
                true,
            ),
            vec!["src/b.rs"]
        );
        assert!(
            files_read(
                root,
                "read",
                &json(serde_json::json!({"path": "../etc/passwd"})),
                "",
                true,
            )
            .is_empty()
        );
        assert!(
            files_read(
                root,
                "read",
                &json(serde_json::json!({"path": "src"})),
                "",
                true
            )
            .is_empty()
        );
        assert!(files_read(root, "read", "not json", "", true).is_empty());
    }

    #[test]
    fn grep_on_a_file_counts_directory_searches_and_listings_do_not() {
        let dir = root();
        let root = dir.path();
        assert_eq!(
            files_read(
                root,
                "grep",
                &json(serde_json::json!({"pattern": "fn", "path": "src/a.rs"})),
                "",
                true,
            ),
            vec!["src/a.rs"]
        );
        assert!(
            files_read(
                root,
                "grep",
                &json(serde_json::json!({"pattern": "fn", "path": "src"})),
                "",
                true,
            )
            .is_empty()
        );
        for name in ["glob", "list", "repo_map", "write", "edit"] {
            assert!(
                files_read(
                    root,
                    name,
                    &json(serde_json::json!({"path": "src/a.rs"})),
                    "",
                    true,
                )
                .is_empty(),
                "{name}"
            );
        }
    }

    #[test]
    fn shell_commands_count_named_files_and_diff_headers_in_the_output() {
        let dir = root();
        let root = dir.path();
        let bash = |command: &str, content: &str| {
            files_read(
                root,
                "bash",
                &json(serde_json::json!({"command": command})),
                content,
                true,
            )
        };
        assert_eq!(
            bash("sed -n '1,40p' src/a.rs | head -20", ""),
            vec!["src/a.rs"]
        );
        assert_eq!(
            bash("cat \"src/b.rs\"; rg fn ./src/a.rs:3", ""),
            vec!["src/b.rs", "src/a.rs"]
        );
        assert!(
            bash("git diff HEAD -- src/a.rs", "").is_empty(),
            "naming a path is not a read when no diff for it was shown"
        );
        assert_eq!(
            bash(
                "git diff HEAD -- src/a.rs src/b.rs",
                "diff --git a/src/a.rs b/src/a.rs\n"
            ),
            vec!["src/a.rs"],
            "a truncated diff counts only the file whose header was shown"
        );
        assert!(bash("echo src/a.rs src/b.rs", "").is_empty());
        assert!(
            files_read(
                root,
                "bash",
                &json(serde_json::json!({"command": "cat src/a.rs src/b.rs"})),
                "fn a() {}\n… output truncated",
                false,
            )
            .is_empty(),
            "a clipped cat of several files did not show every one"
        );
        assert_eq!(
            files_read(
                root,
                "bash",
                &json(serde_json::json!({"command": "git diff HEAD -- src/a.rs src/b.rs"})),
                "diff --git a/src/a.rs b/src/a.rs\n… output truncated",
                false,
            ),
            vec!["src/a.rs"],
            "a header in the clipped output still counts"
        );
        assert!(bash("git status --short", "").is_empty());
        let diff = "diff --git a/src/b.rs b/src/b.rs\nindex 1..2 100644\n--- a/src/b.rs\n+++ b/src/b.rs\n@@ -1 +1 @@\n-x\n+y\ndiff --git a/gone.rs b/gone.rs\n";
        assert_eq!(bash("git diff HEAD", diff), vec!["src/b.rs"]);
        assert_eq!(
            bash("git show HEAD", "+++ b/README.md\t\n"),
            vec!["README.md"]
        );
        let tool = format!(
            "{diff}\nnew (untracked) files and bounded contents:\n--- /dev/null\n+++ b/README.md\n+# App\n\
summarized binary/generated/vendor/oversized files:\n  src/a.rs\n"
        );
        assert_eq!(
            files_read(root, "diff", "{}", &tool, true),
            vec!["src/b.rs", "README.md"],
            "the `diff` tool's patches are reads; a file it only names is not"
        );
    }

    #[test]
    fn shell_commands_that_only_list_or_count_files_read_nothing() {
        let dir = root();
        let root = dir.path();
        let bash = |command: &str, content: &str| {
            files_read(
                root,
                "bash",
                &json(serde_json::json!({"command": command})),
                content,
                true,
            )
        };
        for command in [
            "wc -l src/a.rs src/b.rs",
            "ls -la src/a.rs README.md",
            "stat src/a.rs",
            "git log --oneline -- src/a.rs",
            "git --no-pager status --short src/b.rs",
            "time wc -l src/a.rs",
            "git diff --name-only | xargs wc -l src/b.rs",
        ] {
            assert!(bash(command, "").is_empty(), "{command}");
        }
        assert_eq!(
            bash("head -40 src/a.rs && wc -l src/b.rs", ""),
            vec!["src/a.rs"]
        );
        assert_eq!(bash("LC_ALL=C /usr/bin/cat src/b.rs", ""), vec!["src/b.rs"]);
        assert_eq!(
            bash(
                "git log -p -- src/b.rs",
                "diff --git a/src/b.rs b/src/b.rs\n"
            ),
            vec!["src/b.rs"],
            "the diff it printed is a read"
        );
    }

    #[test]
    fn unread_keeps_scope_order() {
        let scope = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        assert_eq!(
            unread_files(&scope, &["c".to_string(), "x".to_string()]),
            vec!["a", "b"]
        );
        assert!(unread_files(&scope, &scope).is_empty());
    }
}
