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
        "grep" | "glob" | "diff" | "status" => Some(EvidenceKind::TargetedSearch),
        "list" => Some(EvidenceKind::Listing),
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
    if explicitly_mutating_request(&normalized) && !read_only_language_present(&normalized) {
        return None;
    }
    if contains_any(
        &normalized,
        &[
            "security", "unsafe", "unwrap", "expect", "panic", "secret", "token", "auth",
        ],
    ) {
        return Some(ReviewIntent::Security);
    }
    if contains_any(
        &normalized,
        &[
            "missing",
            "gap",
            "gaps",
            "lacks",
            "whats missing",
            "what is missing",
        ],
    ) {
        return Some(ReviewIntent::Gaps);
    }
    if contains_any(
        &normalized,
        &[
            "roadmap",
            "build next",
            "what should build",
            "what should we build",
            "consider building",
        ],
    ) {
        return Some(ReviewIntent::Roadmap);
    }
    if contains_any(
        &normalized,
        &["status", "state", "current state", "discuss state"],
    ) {
        return Some(ReviewIntent::Status);
    }
    if contains_any(
        &normalized,
        &[
            "review codebase",
            "code review",
            "review repo",
            "review repository",
            "audit codebase",
        ],
    ) {
        return Some(ReviewIntent::Review);
    }
    None
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

pub(crate) fn explicitly_mutating_request(normalized: &str) -> bool {
    let words: Vec<&str> = normalized.split_whitespace().collect();
    words.iter().any(|word| {
        matches!(
            *word,
            "fix"
                | "change"
                | "update"
                | "write"
                | "create"
                | "delete"
                | "remove"
                | "refactor"
                | "patch"
                | "commit"
        )
    }) || (words
        .iter()
        .any(|word| matches!(*word, "implement" | "build"))
        && !contains_any(
            normalized,
            &["consider", "should", "what should", "discuss"],
        ))
}

pub(crate) fn read_only_language_present(normalized: &str) -> bool {
    contains_any(
        normalized,
        &[
            "read only",
            "discuss only",
            "discuss",
            "review",
            "audit",
            "status",
            "state",
            "what should",
            "consider",
        ],
    )
}

pub(crate) fn read_only_turn_prompt(input: &str, intent: ReviewIntent) -> String {
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
        "{input}\n\nRead-only review guard: do not write, edit, apply patches, run mutating shell commands, or change files. Use read-only inspection before the final answer. {recipe} If only a directory listing is available, keep inspecting or explicitly say the evidence is insufficient instead of making file-specific findings."
    )
}

pub(crate) fn classify_implementation_intent(input: &str) -> Option<ImplementationIntent> {
    let normalized = normalize_intent_text(input);
    if normalized.trim().is_empty()
        || contains_any(
            &normalized,
            &[
                "read only",
                "read only review guard",
                "do not write",
                "use read only inspection",
                "what should",
                "should we",
                "consider",
                "roadmap",
                "discuss",
                "analyze",
                "assess",
                "evaluate",
            ],
        )
    {
        return None;
    }
    let words: Vec<&str> = normalized.split_whitespace().collect();
    let has_direct_action = words
        .iter()
        .any(|word| matches!(*word, "build" | "create" | "make" | "develop" | "scaffold"));
    // "implementation"/"implementing" as a noun ("finish the av1
    // implementation") is both the action and the artifact — the word itself
    // names the thing being built, so it satisfies `has_artifact` below even
    // when no explicit artifact word ("app", "tui", …) appears.
    let has_implementation_noun = words
        .iter()
        .any(|word| matches!(*word, "implementation" | "implementing"));
    let has_generic_action = words.iter().any(|word| {
        matches!(
            *word,
            "implement" | "implementing" | "implementation" | "write"
        )
    });
    if !has_direct_action && !has_generic_action {
        return None;
    }
    let has_artifact = has_implementation_noun
        || words.iter().any(|word| {
            matches!(
                *word,
                "app"
                    | "application"
                    | "tool"
                    | "tui"
                    | "cli"
                    | "calculator"
                    | "estimator"
                    | "dashboard"
                    | "program"
                    | "utility"
                    | "service"
                    | "game"
            )
        })
        || contains_any(
            &normalized,
            &[
                "command line",
                "terminal ui",
                "text ui",
                "gpu training",
                "training time",
                "loan calculator",
            ],
        );
    if !has_artifact {
        return None;
    }
    Some(ImplementationIntent {
        tui: implementation_mentions_tui(&normalized),
        gpu_training_estimator: implementation_mentions_gpu_training_estimator(&normalized),
    })
}

pub(crate) fn implementation_mentions_tui(normalized: &str) -> bool {
    contains_any(
        normalized,
        &["tui", "terminal ui", "text ui", "ratatui", "crossterm"],
    )
}

pub(crate) fn implementation_mentions_gpu_training_estimator(normalized: &str) -> bool {
    contains_any(
        normalized,
        &[
            "gpu training",
            "training time",
            "train time",
            "training calculator",
            "training estimator",
            "how long training",
            "how long it will take to train",
        ],
    ) || (normalized.contains("gpu")
        && normalized.contains("train")
        && normalized.contains("calculator"))
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
    if intent.gpu_training_estimator {
        rules.push(gpu_training_estimator_recipe());
    }
    format!("{input}\n\n{}", rules.join("\n"))
}

pub(crate) fn gpu_training_estimator_recipe() -> String {
    "GPU training estimator requirements: inputs for parameter count, training tokens, precision, and utilization percentage; editable GPU counts for H100 80GB, A100 80GB, L40S, RTX 4090, RTX 3090, and MI300X; estimate `training_flops = 6 * params * tokens` and `seconds = training_flops / (sum(gpu_count * gpu_tflops) * utilization)`; keep pure estimator functions separate from the TUI layer and cover them with unit tests.".to_string()
}
