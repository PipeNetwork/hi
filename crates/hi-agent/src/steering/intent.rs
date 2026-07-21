//! Intent classification and prompt builders: [`classify_read_only_intent`],
//! [`classify_implementation_intent`], evidence-kind detection, and search-hit
//! scoring. Uses types from [`types`](super::types).

use super::types::{EvidenceKind, ImplementationIntent, ReviewIntent, SecuritySearchFamilies};
pub(crate) fn compact_search_hit_line(line: &str) -> String {
    let trimmed = line.trim();
    if trimmed.is_empty()
        || trimmed.starts_with("no matches")
        || trimmed.starts_with("Error:")
        || trimmed.starts_with("[preflight ")
    {
        return String::new();
    }
    let mut parts = trimmed.splitn(3, ':');
    let Some(path) = parts.next().map(str::trim).filter(|path| !path.is_empty()) else {
        return String::new();
    };
    let rest = parts.collect::<Vec<_>>().join(":");
    if rest.trim().is_empty() || !std::path::Path::new(path).is_file() {
        return String::new();
    }
    let rest = rest
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(140)
        .collect::<String>();
    format!("{path}: {rest}")
}

pub(crate) fn search_hit_score(snippet: &str) -> u8 {
    let lower = snippet.to_ascii_lowercase();
    let mut score = 0u8;
    if contains_any(
        &lower,
        &[
            "unsafe", "unwrap(", ".unwrap", "expect(", ".expect", "panic!",
        ],
    ) {
        score = score.saturating_add(5);
    }
    if contains_any(
        &lower,
        &[
            "command::new",
            "process::command",
            "std::process",
            ".spawn(",
            "shell",
            "exec",
        ],
    ) {
        score = score.saturating_add(4);
    }
    if contains_any(
        &lower,
        &[
            "api_key",
            "apikey",
            "api-key",
            "secret",
            "password",
            "bearer",
            "authorization",
            "credential",
        ],
    ) {
        score = score.saturating_add(4);
    }
    if contains_any(
        &lower,
        &[
            "std::env",
            "env::var",
            "std::fs",
            "fs::write",
            "read_to_string",
            "remove_file",
            "set_permissions",
            "0o600",
            "0o700",
        ],
    ) {
        score = score.saturating_add(3);
    }
    if contains_any(&lower, &["token", "auth"]) {
        score = score.saturating_add(1);
    }
    score
}

pub(crate) fn grep_match_line_count(output: &str) -> u32 {
    let trimmed = output.trim();
    if trimmed.is_empty() || trimmed.starts_with("no matches for ") {
        return 0;
    }
    output
        .lines()
        .filter(|line| !line.trim().is_empty())
        .count() as u32
}

pub(crate) fn evidence_kind_for_tool(name: &str, arguments: &str) -> Option<EvidenceKind> {
    match name {
        "read" => Some(EvidenceKind::FileRead),
        "grep" | "glob" => Some(EvidenceKind::TargetedSearch),
        "list" | "diff" | "status" => Some(EvidenceKind::Listing),
        "bash" => evidence_kind_for_bash(arguments),
        _ => None,
    }
}

pub(crate) fn evidence_kind_for_bash(arguments: &str) -> Option<EvidenceKind> {
    let value = serde_json::from_str::<serde_json::Value>(arguments).ok()?;
    let command = value
        .get("command")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_ascii_lowercase();
    if command
        .split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '-')
        .any(|word| matches!(word, "cat" | "sed" | "nl" | "head" | "tail"))
    {
        return Some(EvidenceKind::FileRead);
    }
    if command
        .split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '-')
        .any(|word| matches!(word, "rg" | "grep" | "git"))
    {
        return Some(EvidenceKind::TargetedSearch);
    }
    if command
        .split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '-')
        .any(|word| matches!(word, "ls" | "find"))
    {
        return Some(EvidenceKind::Listing);
    }
    None
}

pub(crate) fn security_search_families_for_tool(
    name: &str,
    arguments: &str,
) -> SecuritySearchFamilies {
    let Some(search_text) = security_search_text_for_tool(name, arguments) else {
        return SecuritySearchFamilies::default();
    };
    security_search_families(&search_text)
}

pub(crate) fn security_search_text_for_tool(name: &str, arguments: &str) -> Option<String> {
    let value = serde_json::from_str::<serde_json::Value>(arguments).ok()?;
    match name {
        "grep" => value
            .get("pattern")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string),
        "glob" => {
            let mut parts = Vec::new();
            for key in ["pattern", "path"] {
                if let Some(text) = value.get(key).and_then(serde_json::Value::as_str)
                    && !text.is_empty()
                {
                    parts.push(text);
                }
            }
            (!parts.is_empty()).then(|| parts.join(" "))
        }
        "bash" => value
            .get("command")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string),
        _ => None,
    }
}

pub(crate) fn security_search_families(search_text: &str) -> SecuritySearchFamilies {
    let lower = search_text.to_ascii_lowercase();
    let tokens = lower
        .split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();
    let has_token = |needles: &[&str]| -> bool {
        tokens
            .iter()
            .any(|token| needles.iter().any(|needle| token == needle))
    };
    SecuritySearchFamilies {
        unsafe_or_panic: contains_any(&lower, &["unsafe", "unwrap", "expect", "panic"]),
        execution_or_fs_env: contains_any(
            &lower,
            &[
                "command",
                "std::process",
                "process::",
                "shell",
                "exec",
                "spawn",
                "filesystem",
                "std::fs",
                "fs::",
                "read_to_string",
                "remove_file",
                "std::env",
                "env::",
            ],
        ) || has_token(&["process", "fs", "file", "write", "env"]),
        secret_or_auth: contains_any(
            &lower,
            &[
                "secret",
                "token",
                "auth",
                "api_key",
                "apikey",
                "password",
                "credential",
                "bearer",
            ],
        ),
    }
}

pub(crate) fn classify_read_only_intent(input: &str) -> Option<ReviewIntent> {
    let normalized = normalize_intent_text(input);
    if normalized.trim().is_empty() {
        return None;
    }
    if let Some(intent) = expanded_read_only_macro_intent(&normalized) {
        return Some(intent);
    }
    explicit_no_mutation_request(&normalized).then(|| no_mutation_review_intent(&normalized))
}

pub(crate) fn normalize_intent_text(input: &str) -> String {
    let lower = input.to_ascii_lowercase();
    let fixed = lower
        .replace("disucss", "discuss")
        .replace("implimenting", "implementing")
        .replace("implimentation", "implementation")
        .replace("impliment", "implement")
        .replace("whats its", "whats")
        .replace("what's its", "whats");
    fixed
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub(crate) fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

fn without_scoped_no_edit_constraints(normalized: &str) -> String {
    let mut text = normalized.to_string();
    for phrase in [
        "do not change the existing tests",
        "do not change existing tests",
        "do not change any tests",
        "do not change the tests",
        "do not change tests",
        "do not change the test",
        "do not change test",
        "do not edit the existing tests",
        "do not edit existing tests",
        "do not edit any tests",
        "do not edit the tests",
        "do not edit tests",
        "do not edit the test",
        "do not edit test",
        "do not modify the existing tests",
        "do not modify existing tests",
        "do not modify any tests",
        "do not modify the tests",
        "do not modify tests",
        "do not modify the test",
        "do not modify test",
        "don t change the existing tests",
        "don t change existing tests",
        "don t change any tests",
        "don t change the tests",
        "don t change tests",
        "don t change the test",
        "don t change test",
        "don t edit the existing tests",
        "don t edit existing tests",
        "don t edit any tests",
        "don t edit the tests",
        "don t edit tests",
        "don t edit the test",
        "don t edit test",
        "don t modify the existing tests",
        "don t modify existing tests",
        "don t modify any tests",
        "don t modify the tests",
        "don t modify tests",
        "don t modify the test",
        "don t modify test",
    ] {
        text = text.replace(phrase, "");
    }
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub(crate) fn default_read_only_inspection_cap(intent: ReviewIntent) -> u32 {
    match intent {
        ReviewIntent::Review => super::constants::REVIEW_INSPECTION_CAP,
        ReviewIntent::Status => super::constants::STATUS_INSPECTION_CAP,
        ReviewIntent::Roadmap => super::constants::ROADMAP_INSPECTION_CAP,
        ReviewIntent::Gaps => super::constants::GAPS_INSPECTION_CAP,
        ReviewIntent::Security => super::constants::SECURITY_INSPECTION_CAP,
    }
}

/// The task-scaled, project-size-ceilinged base cap (no soft-cap extension).
/// `indexed_file_count` is the number of source files the repo-intelligence
/// indexer found (0 when unavailable). The effective cap is:
///
///   min(base * task_multiplier, project_size_ceiling)
///
/// rounded to the nearest integer. When the user gives an explicit cap via
/// prompt text ("at most N file inspections"), that cap is respected as-is —
/// no multiplier or ceiling is applied, because the user's explicit limit is
/// authoritative. The soft-cap extension is added later by
/// [`EvidenceTracker::effective_cap_with_extensions`].
pub(crate) fn scaled_inspection_cap(
    input: &str,
    intent: ReviewIntent,
    indexed_file_count: u32,
) -> u32 {
    // An explicit user-specified cap is authoritative — don't scale it.
    if let Some(explicit) = explicit_read_only_inspection_cap(input) {
        return explicit;
    }
    let base = default_read_only_inspection_cap(intent);
    let multiplier = super::constants::inspection_cap_multiplier(intent);
    let scaled = (base as f64 * multiplier).round() as u32;
    let ceiling = super::constants::inspection_cap_project_ceiling(indexed_file_count);
    scaled.min(ceiling)
}

pub(crate) fn active_read_only_inspection_cap(input: &str, intent: ReviewIntent) -> u32 {
    let default = default_read_only_inspection_cap(intent);
    explicit_read_only_inspection_cap(input).map_or(default, |cap| default.min(cap))
}

/// Quick count of source files in the workspace for project-size-aware cap
/// scaling. Walks the root directory one level deep, counting files with
/// recognized source extensions. Deliberately shallow and fast — this runs at
/// turn setup, not in a hot loop. Returns 0 if the root can't be read.
pub(crate) fn workspace_source_file_count(root: &std::path::Path) -> u32 {
    const SOURCE_EXTENSIONS: &[&str] = &[
        "rs", "py", "go", "js", "ts", "jsx", "tsx", "java", "kt", "rb", "c", "cpp", "cc", "h",
        "hpp", "cs", "swift", "m", "mm", "scala", "clj", "ex", "exs", "erl", "hs", "ml", "fs",
        "nim", "zig", "v", "odin", "lua", "php", "pl", "sh", "bash", "zsh", "fish",
    ];
    let mut count = 0u32;
    let mut stack = vec![root.to_path_buf()];
    let mut depth = 0u32;
    while let Some(dir) = stack.pop() {
        depth = depth.saturating_add(1);
        // Cap the walk so it stays fast on huge repos.
        if depth > 3 || count > 5000 {
            break;
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            // Skip hidden dirs and common non-source directories.
            if path.is_dir() {
                if name_str.starts_with('.')
                    || matches!(
                        name_str.as_ref(),
                        "node_modules" | "target" | "vendor" | ".git" | "dist" | "build"
                            | "__pycache__" | ".venv" | "venv" | "env"
                    )
                {
                    continue;
                }
                stack.push(path);
            } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if SOURCE_EXTENSIONS.contains(&ext) {
                    count = count.saturating_add(1);
                }
            }
        }
    }
    count
}

pub(crate) fn explicit_read_only_inspection_cap(input: &str) -> Option<u32> {
    let normalized = normalize_intent_text(input);
    let words = normalized.split_whitespace().collect::<Vec<_>>();
    let mut cap: Option<u32> = None;
    for index in 0..words.len() {
        let parsed = parse_inspection_cap_at(&words, index);
        cap = match (cap, parsed) {
            (Some(existing), Some(parsed)) => Some(existing.min(parsed)),
            (None, Some(parsed)) => Some(parsed),
            (existing, None) => existing,
        };
    }
    cap
}

fn parse_inspection_cap_at(words: &[&str], index: usize) -> Option<u32> {
    if words.get(index..index + 2) == Some(&["at", "most"]) {
        return parse_cap_after_number(words, index + 2);
    }
    if words.get(index..index + 4) == Some(&["use", "no", "more", "than"])
        || words.get(index..index + 3) == Some(&["no", "more", "than"])
    {
        let number_index = if words.get(index) == Some(&"use") {
            index + 4
        } else {
            index + 3
        };
        return parse_cap_after_number(words, number_index);
    }
    if matches!(words.get(index), Some(&"max" | &"maximum")) {
        return parse_cap_after_number(words, index + 1);
    }
    None
}

fn parse_cap_after_number(words: &[&str], number_index: usize) -> Option<u32> {
    let number = words.get(number_index)?.parse::<u32>().ok()?;
    if number == 0 {
        return None;
    }
    let mut noun_index = number_index + 1;
    if matches!(words.get(noun_index), Some(&"file" | &"tool")) {
        noun_index += 1;
    }
    matches!(
        words.get(noun_index),
        Some(&"inspection" | &"inspections" | &"read" | &"reads")
    )
    .then_some(number)
}

fn expanded_read_only_macro_intent(normalized: &str) -> Option<ReviewIntent> {
    if normalized.starts_with("read only security request for") {
        Some(ReviewIntent::Security)
    } else if normalized.starts_with("read only status request for") {
        Some(ReviewIntent::Status)
    } else if normalized.starts_with("read only roadmap request for") {
        Some(ReviewIntent::Roadmap)
    } else if normalized.starts_with("read only gaps request for") {
        Some(ReviewIntent::Gaps)
    } else if normalized.starts_with("read only review request for") {
        Some(ReviewIntent::Review)
    } else {
        None
    }
}

fn no_mutation_review_intent(normalized: &str) -> ReviewIntent {
    if explicit_security_review_request(normalized) {
        ReviewIntent::Security
    } else if explicit_gap_review_request(normalized) {
        ReviewIntent::Gaps
    } else if explicit_roadmap_review_request(normalized) {
        ReviewIntent::Roadmap
    } else if explicit_status_review_request(normalized) {
        ReviewIntent::Status
    } else {
        // Both an explicit code-review request and the default (no recognized
        // review kind) map to a plain code review.
        ReviewIntent::Review
    }
}

pub(crate) fn explicit_security_review_request(normalized: &str) -> bool {
    contains_any(
        normalized,
        &[
            "security review",
            "security audit",
            "security issue",
            "security issues",
            "review for security",
            "audit for security",
            "unsafe unwrap",
            "unsafe unwraps",
            "secret leak",
            "secret leaks",
            "token leak",
            "token leaks",
            "auth leak",
            "auth leaks",
        ],
    )
}

pub(crate) fn explicit_gap_review_request(normalized: &str) -> bool {
    contains_any(
        normalized,
        &[
            "what is missing",
            "whats missing",
            "what missing",
            "missing gaps",
            "gap review",
            "review gaps",
            "review for gaps",
            "audit gaps",
        ],
    )
}

pub(crate) fn explicit_roadmap_review_request(normalized: &str) -> bool {
    contains_any(
        normalized,
        &[
            "roadmap",
            "build next",
            "what should build",
            "what should we build",
            "what should i build",
            "what should we do next",
            "what should i do next",
            "consider building",
        ],
    )
}

pub(crate) fn explicit_status_review_request(normalized: &str) -> bool {
    matches!(normalized, "status" | "state")
        || contains_any(
            normalized,
            &[
                "current status",
                "project status",
                "repo status",
                "repository status",
                "codebase status",
                "workspace status",
                "current state",
                "state of",
                "status of",
                "where are we",
            ],
        )
}

pub(crate) fn explicit_no_mutation_request(normalized: &str) -> bool {
    let unscoped = without_scoped_no_edit_constraints(normalized);
    if contains_any(
        &unscoped,
        &[
            "do not treat this as a read only",
            "do not treat this as read only",
            "don t treat this as a read only",
            "don t treat this as read only",
            "not a read only review",
            "not read only",
        ],
    ) && !explicit_no_edit_instruction(&unscoped)
    {
        return false;
    }

    contains_any(
        &unscoped,
        &[
            "read only",
            "discuss only",
            "do not write",
            "do not edit",
            "do not modify",
            "do not change",
            "don t write",
            "don t edit",
            "don t modify",
            "don t change",
            "without modifying",
            "without changing",
            "no file changes",
            "no changes",
        ],
    )
}

pub(crate) fn read_only_turn_prompt(input: &str, intent: ReviewIntent) -> String {
    let cap = active_read_only_inspection_cap(input, intent);
    let recipe = match intent {
        ReviewIntent::Security => {
            "Search for unsafe, unwrap, expect, panic!, command execution, filesystem/env access, and secret/token/auth patterns. Then read the most relevant matching files."
        }
        ReviewIntent::Status => {
            "Inspect git status or diff summary, workspace manifests, README/docs if present, main crate or module entrypoints, and tests."
        }
        ReviewIntent::Roadmap => {
            "Inspect manifests, owning modules, tests, and TODO/FIXME or missing-coverage search results before naming build-next work."
        }
        ReviewIntent::Gaps => {
            "Inspect manifests, owning modules, tests, and TODO/FIXME or missing-coverage search results before naming gaps."
        }
        ReviewIntent::Review => {
            "Inspect relevant files or targeted search results before giving findings."
        }
    };
    format!(
        "{input}\n\nRead-only review guard: do not write, edit, apply patches, run mutating shell commands, or change files. Use read-only inspection before the final answer. Active inspection cap: at most {cap} file reads/searches for this turn; listings and diffs may provide context but do not raise the cap. Context-efficient tools (explore, repo_map, find_symbol) cost less against the cap — prefer them to cover more ground. Once the cap is reached, answer from gathered evidence instead of inspecting more. {recipe} If only a directory listing is available, keep inspecting before making file-specific findings."
    )
}

pub(crate) fn classify_implementation_intent(input: &str) -> Option<ImplementationIntent> {
    let normalized = normalize_intent_text(input);
    if normalized.trim().is_empty()
        || !(expanded_build_macro_request(&normalized)
            || explicit_implementation_task_request(&normalized)
            || natural_implementation_continuation(&normalized))
    {
        return None;
    }
    Some(ImplementationIntent {
        tui: implementation_mentions_tui(&normalized),
    })
}

fn natural_implementation_continuation(normalized: &str) -> bool {
    contains_any(
        normalized,
        &[
            "keep building",
            "continue building",
            "keep implementing",
            "continue implementing",
        ],
    ) && !explicit_no_edit_instruction(normalized)
}

fn expanded_build_macro_request(normalized: &str) -> bool {
    normalized.starts_with("build ")
        && normalized.contains("implementation requirements inspect the workspace")
}

fn explicit_implementation_task_request(normalized: &str) -> bool {
    (normalized.starts_with("implementation task")
        || normalized.contains(" implementation task ")
        || normalized.starts_with("benchmark implementation task")
        || normalized.contains(" disposable benchmark workspace "))
        && contains_any(
            normalized,
            &[
                "expected to edit",
                "allowed to edit",
                "edit files",
                "apply patches",
                "change files",
                "run the verification command",
                "implement ",
            ],
        )
        && !explicit_no_edit_instruction(normalized)
}

fn explicit_no_edit_instruction(normalized: &str) -> bool {
    let unscoped = without_scoped_no_edit_constraints(normalized);
    contains_any(
        &unscoped,
        &[
            "do not write",
            "do not edit",
            "do not modify",
            "do not change",
            "don t write",
            "don t edit",
            "don t modify",
            "don t change",
            "without modifying",
            "without changing",
            "no file changes",
            "no changes",
        ],
    )
}

pub(crate) fn implementation_mentions_tui(normalized: &str) -> bool {
    contains_any(
        normalized,
        &["tui", "terminal ui", "text ui", "ratatui", "crossterm"],
    )
}

pub(crate) fn implementation_turn_prompt(input: &str, intent: ImplementationIntent) -> String {
    let mut rules = vec![
        "Implementation guard: inspect the workspace before choosing files or stack.".to_string(),
        "Choose the existing local stack from manifests and entrypoints. If the workspace is empty or has no manifest, create the minimal project in the current directory rather than a nested sub-project.".to_string(),
        "Make concrete file changes; do not stop at a plan, explanation, or scaffold.".to_string(),
        "Prefer a compact working vertical slice and small valid tool calls over one huge all-at-once source write.".to_string(),
        "Run a noninteractive validation command after the last file change, such as cargo test/check/build, npm test/build, python -m pytest, go test, make test, or an equivalent local command.".to_string(),
        "The final recap must name changed files and exact validation command(s).".to_string(),
        "Do not install packages globally or with host package managers. Use project manifests, project-local installs, or a virtual environment when dependencies are necessary.".to_string(),
    ];
    if intent.tui {
        rules.push("For a TUI with no clear existing stack, default to Rust with Ratatui and Crossterm. In an empty directory, prefer `cargo init --bin .` before editing so Cargo.toml already has a valid target. Do not run a foreground TUI directly; validate with unit tests, cargo build/check/test, or a bounded smoke command such as `timeout 5s cargo run`.".to_string());
    }
    format!("{input}\n\n{}", rules.join("\n"))
}



#[cfg(test)]
mod golden_table {
    use super::*;
    use crate::steering::types::ReviewIntent;

    /// Frozen prompt → intent pairs. Prefer `/macro` expansions and phrases already
    /// proven in `tests/steering.rs` so this table tracks real classifier gates.
    #[test]
    fn read_only_intent_golden_table() {
        let cases: &[(&str, Option<ReviewIntent>)] = &[
            ("status", None),
            ("fix the unsafe unwraps", None),
            ("review codebase and discuss status and state", None),
            (
                "review this code for auth leaks but do not edit",
                Some(ReviewIntent::Security),
            ),
            (
                "Review this codebase for issues related to ipop/coder-balanced API routing or latency. Use at most 4 file inspections. Do not modify files. Return concise findings only.",
                Some(ReviewIntent::Review),
            ),
        ];
        for (prompt, want) in cases {
            assert_eq!(
                classify_read_only_intent(prompt),
                *want,
                "read-only classify failed for {prompt:?}"
            );
        }
    }

    #[test]
    fn implementation_intent_golden_table() {
        let build_macro = "Build a small helper.

Implementation requirements
Inspect the workspace before editing.
Expected to edit files and run verification.";
        // Expanded /build macro shape (see expanded_build_macro_request).
        let expanded = "build foo implementation requirements inspect the workspace before you edit files";
        assert!(
            classify_implementation_intent(expanded).is_some()
                || classify_implementation_intent(build_macro).is_some()
                || classify_implementation_intent(
                    "Implementation task: expected to edit files and run the verification command"
                )
                .is_some(),
            "at least one known implementation shape should classify"
        );
        assert!(
            classify_implementation_intent("keep building the feature").is_some(),
            "natural continuation should classify"
        );
        for prompt in [
            "what is the status?",
            "review only, do not change code",
            "discuss the architecture",
            "status",
        ] {
            assert_eq!(
                classify_implementation_intent(prompt),
                None,
                "expected no implementation intent for {prompt:?}"
            );
        }
    }
}
