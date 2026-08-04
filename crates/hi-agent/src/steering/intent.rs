//! Intent classification and prompt builders: [`classify_read_only_intent`],
//! [`classify_implementation_intent`], evidence-kind detection, and search-hit
//! scoring. Uses types from [`types`](super::types).

use std::collections::BTreeSet;

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

/// Recognize the common bare review form after the task contract has already
/// established that the turn is read-only.  `classify_read_only_intent` stays
/// intentionally conservative because it is also used to distinguish an
/// explicit "do not edit" constraint from an implementation request.  That
/// made a plain `review codebase` inconsistent, though: tool admission removed
/// mutation tools while prompt steering, preflight, and review caps were not
/// enabled.  Keep this second-stage classifier scoped to review verbs and only
/// call it with the contract's read-only result.
pub(crate) fn implicit_read_only_review_intent(
    input: &str,
    task_is_read_only: bool,
) -> Option<ReviewIntent> {
    if !task_is_read_only || !is_bare_codebase_review(input) {
        return None;
    }
    let normalized = normalize_intent_text(input);
    Some(no_mutation_review_intent(&normalized))
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
    // Token-structural, not an enumerated phrase list: a negation head
    // ("do not" / "don t" / "never" / "without" / "avoid") followed by an
    // edit verb whose object within a short window is *scoped* (tests, docs,
    // comments, …) is a carve-out on an implementation task — "fix the bug;
    // do not modify any existing test files" — not a read-only request. The
    // clause is removed before the no-mutation phrases are looked for.
    // Negated edits with global or absent objects ("do not modify anything",
    // bare "do not edit") are kept: those are genuine no-mutation requests.
    const EDIT_VERBS: &[&str] = &[
        "change",
        "edit",
        "modify",
        "touch",
        "alter",
        "update",
        "rewrite",
        "changing",
        "editing",
        "modifying",
        "touching",
        "altering",
        "updating",
        "rewriting",
    ];
    const SCOPED_OBJECTS: &[&str] = &[
        "test",
        "tests",
        "spec",
        "specs",
        "doc",
        "docs",
        "documentation",
        "comment",
        "comments",
        "changelog",
        "readme",
    ];
    /// Determiners/adjectives allowed between verb and object ("any existing test").
    const OBJECT_WINDOW: usize = 4;
    let tokens: Vec<&str> = normalized.split_whitespace().collect();
    let mut keep = vec![true; tokens.len()];
    let mut i = 0;
    while i < tokens.len() {
        let head_len = match tokens[i] {
            "do" | "don" if matches!(tokens.get(i + 1).copied(), Some("not") | Some("t")) => 2,
            "dont" | "never" | "without" | "avoid" => 1,
            _ => 0,
        };
        if head_len == 0 {
            i += 1;
            continue;
        }
        let verb_index = i + head_len;
        if !tokens
            .get(verb_index)
            .is_some_and(|verb| EDIT_VERBS.contains(verb))
        {
            i += 1;
            continue;
        }
        let mut matched = false;
        for j in verb_index + 1..=(verb_index + OBJECT_WINDOW).min(tokens.len().saturating_sub(1)) {
            if SCOPED_OBJECTS.contains(&tokens[j]) {
                let mut clause_end = j;
                if matches!(tokens.get(j + 1).copied(), Some("file") | Some("files")) {
                    clause_end = j + 1;
                }
                for slot in keep.iter_mut().take(clause_end + 1).skip(i) {
                    *slot = false;
                }
                i = clause_end + 1;
                matched = true;
                break;
            }
        }
        if !matched {
            i += 1;
        }
    }
    tokens
        .iter()
        .zip(&keep)
        .filter_map(|(token, keep)| keep.then_some(*token))
        .collect::<Vec<_>>()
        .join(" ")
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
    if matches!(intent, ReviewIntent::Review) && is_bounded_file_review(input, false) {
        return super::constants::BOUNDED_FILE_REVIEW_INSPECTION_CAP;
    }
    let base = default_read_only_inspection_cap(intent);
    let multiplier = super::constants::inspection_cap_multiplier(intent);
    let scaled = (base as f64 * multiplier).round() as u32;
    let ceiling = super::constants::inspection_cap_project_ceiling(indexed_file_count);
    scaled.min(ceiling)
}

pub(crate) fn active_read_only_inspection_cap(input: &str, intent: ReviewIntent) -> u32 {
    let default = default_read_only_inspection_cap(intent);
    if let Some(explicit) = explicit_read_only_inspection_cap(input) {
        return default.min(explicit);
    }
    if matches!(intent, ReviewIntent::Review) && is_bounded_file_review(input, false) {
        return super::constants::BOUNDED_FILE_REVIEW_INSPECTION_CAP;
    }
    default
}

/// Quick count of source files in the workspace for project-size-aware cap
/// scaling. Walks the root directory through depth three, counting files with
/// recognized source extensions. Deliberately shallow and fast — this runs at
/// turn setup, not in a hot loop. Returns 0 if the root can't be read.
pub(crate) fn workspace_source_file_count(root: &std::path::Path) -> u32 {
    const SOURCE_EXTENSIONS: &[&str] = &[
        "rs", "py", "go", "js", "ts", "jsx", "tsx", "java", "kt", "rb", "c", "cpp", "cc", "h",
        "hpp", "cs", "swift", "m", "mm", "scala", "clj", "ex", "exs", "erl", "hs", "ml", "fs",
        "nim", "zig", "v", "odin", "lua", "php", "pl", "sh", "bash", "zsh", "fish",
    ];
    let mut count = 0u32;
    let mut stack = vec![(root.to_path_buf(), 0u32)];
    while let Some((dir, depth)) = stack.pop() {
        // Cap the walk so it stays fast on huge repos while still counting
        // every directory up to the intended depth. The old global counter
        // stopped after visiting only three directories total, making the
        // project-size cap depend on directory iteration order.
        if depth > 3 {
            continue;
        }
        if count >= 5000 {
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
            if entry.file_type().is_ok_and(|file_type| file_type.is_dir()) {
                // A top-level `models/` directory is the documented local
                // model-cache location. It can contain hundreds of gigabytes
                // and should not affect the review-cap estimate. Keep nested
                // `src/models` directories eligible because those can be real
                // source packages.
                let root_model_cache = depth == 0 && name_str == "models";
                // `terminal-bench/jobs` contains timestamped run artifacts,
                // including source-looking files. It is not the project's
                // source tree, and walking it once per review turn can be
                // surprisingly expensive on a large benchmark checkout.
                let terminal_bench_jobs = depth == 2
                    && name_str == "jobs"
                    && dir.strip_prefix(root).is_ok_and(|relative| {
                        relative == std::path::Path::new("bench/terminal-bench")
                    });
                if root_model_cache
                    || terminal_bench_jobs
                    || name_str.starts_with('.')
                    || matches!(
                        name_str.as_ref(),
                        "node_modules"
                            | "target"
                            | "vendor"
                            | ".git"
                            | "dist"
                            | "build"
                            | "__pycache__"
                            | ".venv"
                            | "venv"
                            | "env"
                            | "hi-test-scratch"
                    )
                {
                    continue;
                }
                stack.push((path, depth.saturating_add(1)));
            } else if let Some(ext) = path.extension().and_then(|e| e.to_str())
                && count < 5000
                && SOURCE_EXTENSIONS.contains(&ext)
            {
                count = count.saturating_add(1);
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
        ],
    ) || no_changes_is_a_request(&unscoped)
}

/// "no changes" / "no file changes" counts as a no-mutation request only when
/// it isn't a *description* — real issue reports say things like "I made no
/// changes to it and it still fails", which is evidence about the past, not
/// an instruction for this turn (found via the SWE-bench prompt corpus).
fn no_changes_is_a_request(unscoped: &str) -> bool {
    const DESCRIPTIVE_PRECEDERS: &[&str] = &[
        "made",
        "was",
        "were",
        "has",
        "had",
        "have",
        "saw",
        "seen",
        "shows",
        "showed",
        "showing",
        "observed",
        "noticed",
        "reproduced",
        "reports",
        "reported",
    ];
    let tokens: Vec<&str> = unscoped.split_whitespace().collect();
    for (i, token) in tokens.iter().enumerate() {
        if *token != "no" {
            continue;
        }
        let object = (tokens.get(i + 1).copied(), tokens.get(i + 2).copied());
        if !matches!(
            object,
            (Some("changes" | "change"), _) | (Some("file"), Some("changes" | "change"))
        ) {
            continue;
        }
        let descriptive = i
            .checked_sub(1)
            .and_then(|j| tokens.get(j))
            .is_some_and(|prev| DESCRIPTIVE_PRECEDERS.contains(prev));
        if !descriptive {
            return true;
        }
    }
    false
}

pub(crate) fn read_only_turn_prompt(input: &str, intent: ReviewIntent) -> String {
    let cap = active_read_only_inspection_cap(input, intent);
    let bounded_exact_review =
        matches!(intent, ReviewIntent::Review) && is_bounded_file_review(input, false);
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
            "Treat this as a bounded static review: make one orientation pass, inspect the highest-risk relevant files or targeted search results, then give concrete findings. Do not repeatedly relist the workspace, narrate planning, or spawn subagents unless the user explicitly asks for parallel investigations."
        }
    };
    let bounded_guidance = if bounded_exact_review {
        " This is a bounded exact-file review: start with one batched `read` call using the `paths` array for all named files when there is more than one. Prefer one read pass per named file and a targeted `grep` only when it tests a concrete candidate. Do not keep paging through a large file for completeness. After the first useful pass, stop inspecting and answer from the evidence unless one targeted read is required to verify a specific finding. Do not reread the same content; make a best-effort finding from the first useful pass and state the inspection limit instead."
    } else {
        ""
    };
    format!(
        "{input}\n\nRead-only review guard: use only the currently advertised read-only inspection tools; never invent tool names or handles remembered from earlier turns. Do not write, edit, apply patches, or change files. Respect explicit user exclusions: never invoke a tool or inspect an artifact the user explicitly forbids, even if this review recipe mentions it. Do not narrate tool availability, stale handles, polling recovery, or internal steering to the user; inspect with the available tools and give the review directly. Use read-only inspection before the final answer. Active inspection cap: at most {cap} file reads/searches for this turn; listings and diffs may provide context but do not raise the cap. Context-efficient tools (explore, repo_map, find_symbol) cost less against the cap — prefer them to cover more ground. Once the cap is reached, answer from gathered evidence instead of inspecting more. {recipe}{bounded_guidance} If only a directory listing is available, keep inspecting before making file-specific findings."
    )
}

/// Whether a read-only review names a small, closed set of files. This shared
/// shape keeps prompt steering and tool advertisement aligned: exact-file
/// reviews get a cheap targeted catalog, while broad reviews retain discovery
/// tools so they do not lose workspace coverage.
pub(crate) fn is_bounded_file_review(input: &str, mutating: bool) -> bool {
    if mutating {
        return false;
    }
    let lower = input.to_ascii_lowercase();
    if !["review", "audit", "inspect"]
        .iter()
        .any(|marker| lower.contains(marker))
    {
        return false;
    }
    if [
        "across",
        "directory",
        "folder",
        "codebase",
        "whole repo",
        "entire repo",
        "all files",
        "related",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
    {
        return false;
    }
    let file_mentions = lower
        .split(|character: char| {
            !(character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-' | '/'))
        })
        .filter(|token| is_file_reference(token))
        .collect::<BTreeSet<_>>();
    (1..=3).contains(&file_mentions.len())
}

/// Whether a broad review is the short, underspecified form for which the
/// deterministic preflight plus two model-directed inspection passes is enough.
/// Detailed reviews keep the normal discovery loop; explicit parallel review
/// requests are also left alone so the user retains control over that cost.
pub(crate) fn is_bare_codebase_review(input: &str) -> bool {
    let normalized = normalize_intent_text(input);
    let words = normalized.split_whitespace().collect::<Vec<_>>();
    if words.len() > 7
        || !matches!(words.first().copied(), Some("review" | "audit"))
        || words
            .iter()
            .any(|word| matches!(*word, "parallel" | "subagent" | "delegate"))
    {
        return false;
    }
    words
        .iter()
        .any(|word| matches!(*word, "codebase" | "repository" | "repo"))
}

pub(crate) fn is_file_reference(token: &str) -> bool {
    let token = token.trim_matches(['.', '/', '-', '_']);
    token == "readme"
        || token == "license"
        || token == "cargo.toml"
        || token == "package.json"
        || [
            ".md", ".txt", ".toml", ".json", ".yaml", ".yml", ".rs", ".py", ".js", ".ts", ".tsx",
            ".go", ".java", ".c", ".cpp", ".h",
        ]
        .iter()
        .any(|suffix| token.ends_with(suffix))
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
    fn implicit_review_classifier_aligns_bare_review_with_read_only_contract() {
        assert_eq!(
            implicit_read_only_review_intent("review codebase", true),
            Some(ReviewIntent::Review)
        );
        assert_eq!(
            implicit_read_only_review_intent("review codebase and discuss status", true),
            Some(ReviewIntent::Review)
        );
        assert_eq!(
            implicit_read_only_review_intent("review codebase and fix the bug", false),
            None
        );
    }

    #[test]
    fn bare_codebase_review_is_distinguished_from_deep_or_parallel_review() {
        assert!(is_bare_codebase_review("review codebase"));
        assert!(is_bare_codebase_review(
            "review the repository for major issues"
        ));
        assert!(!is_bare_codebase_review(
            "review codebase using parallel independent investigations"
        ));
        assert!(!is_bare_codebase_review(
            "review crates/hi-agent/src/lib.rs and trace the full request lifecycle"
        ));
    }

    /// Corpus harness against real-world issue reports (every SWE-bench-style
    /// problem statement is an implementation request by construction, so any
    /// read-only classification is a false positive). Reporting-only:
    /// `HI_INTENT_CORPUS=<prompts.jsonl> cargo test -p hi-agent --lib \
    ///  intent_corpus -- --ignored --nocapture`
    #[test]
    #[ignore = "set HI_INTENT_CORPUS to a jsonl of {\"prompt\": …} lines"]
    fn intent_corpus_read_only_false_positives() {
        let Some(path) = std::env::var_os("HI_INTENT_CORPUS") else {
            return;
        };
        let text = std::fs::read_to_string(path).expect("corpus file");
        let mut total = 0usize;
        let mut false_positives = Vec::new();
        for line in text.lines() {
            let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            let Some(prompt) = value.get("prompt").and_then(|p| p.as_str()) else {
                continue;
            };
            total += 1;
            if let Some(intent) = classify_read_only_intent(prompt) {
                let head: String = prompt.chars().take(120).collect();
                false_positives.push(format!("{intent:?}: {head}"));
            }
        }
        println!(
            "intent corpus: {}/{total} implementation prompts misread as read-only",
            false_positives.len()
        );
        for fp in false_positives.iter().take(10) {
            println!("  FP {fp}");
        }
    }

    #[test]
    fn read_only_prompt_avoids_internal_tool_availability_wording() {
        let prompt = read_only_turn_prompt(
            "review this code for auth leaks but do not edit",
            ReviewIntent::Security,
        );

        assert!(prompt.contains("currently advertised read-only inspection tools"));
        assert!(prompt.contains("advertised read-only inspection tools"));
        assert!(prompt.contains("invent tool names or handles remembered from earlier turns"));
        assert!(
            !prompt.contains("shell execution (`bash`)")
                && !prompt.contains("unavailable for this review")
        );
        assert!(!prompt.contains("run mutating shell commands"));
    }

    #[test]
    fn bounded_exact_review_prompt_prefers_a_best_effort_targeted_pass() {
        let prompt = read_only_turn_prompt(
            "Review only crates/hi-ai/src/openai/request.rs and crates/hi-ai/src/openai/stream.rs for one concrete bug. Use targeted read or grep within those two files only and do not edit files.",
            ReviewIntent::Review,
        );

        assert!(prompt.contains("bounded exact-file review"));
        assert!(prompt.contains("one batched `read` call"));
        assert!(prompt.contains("Do not reread the same content"));
        assert!(prompt.contains("best-effort finding"));
        assert!(prompt.contains("Do not keep paging through a large file"));
    }

    #[test]
    fn broad_review_prompt_keeps_general_inspection_guidance() {
        let prompt =
            read_only_turn_prompt("Review the codebase for major issues", ReviewIntent::Review);

        assert!(!prompt.contains("bounded exact-file review"));
        assert!(prompt.contains("bounded static review"));
        assert!(prompt.contains("Do not repeatedly relist the workspace"));
    }

    #[test]
    fn source_count_does_not_stop_at_a_deep_sibling() {
        let root = std::env::temp_dir().join(format!(
            "hi-intent-source-count-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock is after the epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(root.join("deep/a/b/c")).expect("create deep tree");
        std::fs::create_dir_all(root.join("sibling")).expect("create sibling");
        std::fs::write(root.join("deep/a/b/c/ignored.rs"), "fn ignored() {}\n")
            .expect("write deep file");
        std::fs::write(root.join("sibling/kept.rs"), "fn kept() {}\n").expect("write sibling file");
        std::fs::create_dir_all(root.join("models")).expect("create model cache");
        std::fs::write(root.join("models/generated.py"), "def generated(): pass\n")
            .expect("write model cache file");
        std::fs::create_dir_all(root.join("bench/terminal-bench/jobs"))
            .expect("create benchmark jobs");
        std::fs::write(
            root.join("bench/terminal-bench/jobs/generated.py"),
            "def generated_job(): pass\n",
        )
        .expect("write benchmark job file");
        std::fs::create_dir_all(root.join("crates/hi-agent/hi-test-scratch"))
            .expect("create test scratch");
        std::fs::write(
            root.join("crates/hi-agent/hi-test-scratch/generated.rs"),
            "fn generated_test_artifact() {}\n",
        )
        .expect("write test scratch file");

        assert_eq!(workspace_source_file_count(&root), 1);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn implementation_intent_golden_table() {
        let build_macro = "Build a small helper.

Implementation requirements
Inspect the workspace before editing.
Expected to edit files and run verification.";
        // Expanded /build macro shape (see expanded_build_macro_request).
        let expanded =
            "build foo implementation requirements inspect the workspace before you edit files";
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
