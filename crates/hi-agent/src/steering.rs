//! Read-only review and implementation "steering": intent classification,
//! evidence/implementation trackers, preflight call planning, and the nudge
//! strings injected back into the transcript when the model's answer lacks
//! inspected evidence, concrete file citations, or post-edit validation.
//!
//! All of this is pure input classification and text generation — none of it
//! touches `Agent` state directly — so it lives outside the main `lib.rs`.

/// Sent when the model re-issues the exact same tool call as the previous
/// round. The command already ran and its output is in the history just above —
/// re-running it will only produce the same result. This nudges the model to act
/// on that output (edit the code, move on, or finish) instead of looping.
pub(crate) const REPEAT_NUDGE: &str = "You just ran that exact command last round and its output is already \
in the conversation above — running it again will only repeat the same result. Act on that output \
now: make the edit it points to, move to the next step, or if the task is already complete, stop \
and give your final recap. Do not re-run the same command.";

pub(crate) const NO_EVIDENCE_REVIEW_NUDGE: &str = "This read-only review has no inspected evidence yet. \
Do not finalize. Use read-only inspection tools first, then answer from the inspected evidence. \
If inspection is impossible, explicitly say the evidence is insufficient.";
pub(crate) const READ_ONLY_SAFE_CONTEXT_WINDOW: u32 = 12_000;
pub(crate) const READ_ONLY_PREFLIGHT_GREP_MAX_LINES: usize = 32;
pub(crate) const READ_ONLY_PREFLIGHT_DIFF_MAX_LINES: usize = 160;
pub(crate) const SECURITY_PREFLIGHT_EXTRA_READ_LIMIT: u32 = 90;
pub(crate) const DEFAULT_PREFLIGHT_EXTRA_READ_LIMIT: u32 = 120;
pub(crate) const READ_ONLY_PREFLIGHT_MAX_EXTRA_READS: usize = 3;
pub(crate) const NO_EVIDENCE_SECURITY_NUDGE: &str = "This security review has no inspected evidence yet. \
Do not finalize. Search for unsafe, unwrap, expect, panic!, command execution, filesystem/env \
access, and secret/token/auth patterns, then read the most relevant matching files before answering.";
pub(crate) const NO_EVIDENCE_STATUS_NUDGE: &str = "This status review has no inspected evidence yet. \
Do not finalize. Inspect git status or diff summary, workspace manifests, README/docs if present, \
main crate or module entrypoints, and tests before making status claims.";
pub(crate) const NO_EVIDENCE_GAP_NUDGE: &str = "This gap or roadmap review has no inspected evidence yet. \
Do not finalize. Inspect manifests, owning modules, tests, and TODO/FIXME or missing-coverage \
search results before naming gaps or build-next work.";
pub(crate) const REVIEW_DEEPEN_NUDGE: &str = "This read-only review only has a directory listing so far. \
Do not finalize yet. Use a targeted search or read relevant files, then answer from the inspected \
evidence. If deeper inspection is impossible, explicitly say the evidence is insufficient.";
pub(crate) const SECURITY_DEEPEN_NUDGE: &str = "This security review only has a directory listing so far. \
Do not finalize yet. Search for unsafe, unwrap, expect, panic!, command execution, filesystem/env \
access, and secret/token/auth patterns, then read the most relevant matching files before answering.";
pub(crate) const STATUS_DEEPEN_NUDGE: &str = "This status review only has a directory listing so far. Do \
not finalize yet. Inspect git status or diff summary, workspace manifests, README/docs if present, \
main crate or module entrypoints, and tests before making status claims.";
pub(crate) const GAP_DEEPEN_NUDGE: &str = "This gap or roadmap review only has a directory listing so far. \
Do not finalize yet. Inspect manifests, owning modules, tests, and TODO/FIXME or missing-coverage \
search results before naming gaps or build-next work.";
pub(crate) const CONCRETE_REVIEW_NUDGE: &str = "Your read-only review answer did not cite concrete files or \
modules from the inspected evidence. Do not use mutating tools. Answer again with bounded findings \
tied to inspected paths, or explicitly say the evidence is insufficient.";
pub(crate) const READ_AFTER_SEARCH_NUDGE: &str = "The targeted search result is already in the transcript. \
Do not rerun the same search and do not use mutating tools. Read the most relevant matching file, \
then answer from that inspected file. If you cannot pick a file to read, explicitly say the \
evidence is insufficient.";
pub(crate) const SECURITY_BROAD_SEARCH_NUDGE: &str = "This security review searched and read some evidence, \
but it has not covered all required pattern families yet. Do not use mutating tools. Search for \
unsafe/unwrap/expect/panic, command execution/filesystem/env access, and secret/token/auth \
patterns, then answer only from concrete inspected evidence or explicitly say the evidence is \
insufficient.";
pub(crate) const SECURITY_SCOPE_NUDGE: &str = "The security answer made repo-wide all-clear claims that are \
broader than the inspected files and search results support. Do not use mutating tools. Answer \
again with findings explicitly bounded to the searched patterns and inspected files, or explicitly \
say the evidence is insufficient for broader security claims.";
pub(crate) const GAP_SEARCH_OVERCLAIM_NUDGE: &str = "The gap or roadmap answer claimed there were no \
TODO/FIXME/missing gaps even though the targeted search returned matches. Do not use mutating \
tools. Answer again from the inspected files and search matches, or explicitly say the evidence is \
insufficient for broader roadmap claims.";
pub(crate) const SECURITY_PREFLIGHT_PATTERN: &str = "unsafe|unwrap\\(|expect\\(|panic!|std::process|process::Command|Command::new|spawn\\(|std::fs|fs::|read_to_string|std::env|env::|secret|token|auth|api_key|apikey|password|credential|bearer";
pub(crate) const GAP_PREFLIGHT_PATTERN: &str =
    "TODO|FIXME|todo!|unimplemented!|missing|gap|needs coverage|not implemented";
pub(crate) const IMPLEMENTATION_NO_CHANGES_NUDGE: &str = "This is an implementation request, but no \
successful file changes are in the transcript yet. Do not finalize. Inspect the workspace if \
needed, then create or edit the necessary files with write/edit/multi_edit/apply_patch or a \
project-local scaffold command.";
pub(crate) const IMPLEMENTATION_MISSING_VALIDATION_NUDGE: &str = "Files changed for this implementation \
request, but no successful noninteractive validation command ran after the last change. Do not \
finalize. Run the detected build/test/check command now, then finish with changed files and the \
validation command.";
pub(crate) const IMPLEMENTATION_SCAFFOLD_ONLY_NUDGE: &str = "This implementation request has only scaffold \
or dependency/setup changes so far. Do not finalize yet. Edit the actual source/config files that \
implement the requested behavior, then run validation after the final edit.";
pub(crate) const IMPLEMENTATION_EMPTY_TUI_NUDGE: &str = "The implementation preflight found no project \
manifest. This is a TUI request, so scaffold the Rust binary in the current directory now with \
`cargo init --bin .`, then add Ratatui/Crossterm, implement the estimator, and validate with \
`cargo test` or `cargo check`.";
pub(crate) const TOOL_PROTOCOL_RETRY_NUDGE: &str = "The previous response was rejected by the provider \
because it was not a valid tool turn. Continue using exactly valid tool calls from the available \
schemas. For multi-line file creation, prefer `apply_patch` with `*** Add File` hunks, or call \
`write` with JSON arguments containing `path` and `content`. For shell commands, call `bash` with \
a JSON `command`. Do not put malformed JSON, markdown fences, or prose inside a tool call.";
pub(crate) const TOOL_PROTOCOL_TEXT_FALLBACK_NUDGE: &str = "Structured tool calls have been rejected \
repeatedly by the provider. For this next response only, do not use provider/function tool calling. \
Emit exactly one plain-text tool call in this XML-ish format and no markdown fences:\n\
<tool_call>write<arg_key>path</arg_key><arg_value>src/main.rs</arg_value><arg_key>content</arg_key><arg_value>file contents here</arg_value></tool_call>\n\
For shell commands use:\n\
<tool_call>bash<arg_key>command</arg_key><arg_value>cargo test</arg_value></tool_call>\n\
Keep the edit compact; a minimal working vertical slice is better than a huge invalid tool call.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReviewIntent {
    Review,
    Security,
    Status,
    Roadmap,
    Gaps,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ImplementationIntent {
    pub(crate) tui: bool,
    pub(crate) gpu_training_estimator: bool,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ImplementationTracker {
    pub(crate) mutation_seen: bool,
    pub(crate) substantive_edit_seen: bool,
    pub(crate) validation_after_last_mutation: bool,
    pub(crate) preferred_validation: Option<String>,
    pub(crate) no_change_nudges: u32,
    pub(crate) scaffold_only_nudges: u32,
    pub(crate) missing_validation_nudges: u32,
}

impl ImplementationTracker {
    pub(crate) fn record_tool_result(&mut self, name: &str, arguments: &str, output: &str) {
        if output.starts_with("Error:") || output.starts_with("⚠ refused:") {
            return;
        }
        if implementation_tool_call_mutates(name, arguments) {
            self.mutation_seen = true;
            if implementation_tool_call_substantively_edits(name, arguments) {
                self.substantive_edit_seen = true;
            }
            self.validation_after_last_mutation = false;
            return;
        }
        if self.mutation_seen && implementation_tool_call_validates(name, arguments) {
            self.validation_after_last_mutation = true;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EvidenceKind {
    Listing,
    TargetedSearch,
    FileRead,
}

impl EvidenceKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Listing => "listing",
            Self::TargetedSearch => "targeted_search",
            Self::FileRead => "file_read",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct EvidenceTracker {
    pub(crate) saw_listing: bool,
    pub(crate) saw_search: bool,
    pub(crate) saw_read: bool,
    pub(crate) file_reads: u32,
    pub(crate) targeted_searches: u32,
    pub(crate) security_unsafe_search: bool,
    pub(crate) security_execution_search: bool,
    pub(crate) security_secret_search: bool,
    pub(crate) grep_match_lines: u32,
    pub(crate) inspected_paths: Vec<String>,
    pub(crate) search_hit_snippets: Vec<String>,
    pub(crate) first_tool_kind: Option<EvidenceKind>,
    pub(crate) quality_repair_nudges: u32,
}

impl EvidenceTracker {
    pub(crate) fn record_success(&mut self, name: &str, arguments: &str, output: &str) {
        if output.starts_with("Error:") {
            return;
        }
        let Some(kind) = evidence_kind_for_tool(name, arguments) else {
            return;
        };
        if self.first_tool_kind.is_none() {
            self.first_tool_kind = Some(kind);
        }
        match kind {
            EvidenceKind::Listing => self.saw_listing = true,
            EvidenceKind::TargetedSearch => {
                self.saw_search = true;
                self.targeted_searches = self.targeted_searches.saturating_add(1);
                let families = security_search_families_for_tool(name, arguments);
                self.security_unsafe_search |= families.unsafe_or_panic;
                self.security_execution_search |= families.execution_or_fs_env;
                self.security_secret_search |= families.secret_or_auth;
                if name == "grep" {
                    self.grep_match_lines = self
                        .grep_match_lines
                        .saturating_add(grep_match_line_count(output));
                    self.record_search_hit_snippets(output);
                }
            }
            EvidenceKind::FileRead => {
                self.saw_read = true;
                self.file_reads = self.file_reads.saturating_add(1);
                if let Some(path) = hi_tools::target_path(name, arguments)
                    && !path.is_empty()
                    && !self
                        .inspected_paths
                        .iter()
                        .any(|existing| existing == &path)
                {
                    self.inspected_paths.push(path);
                }
            }
        }
    }

    pub(crate) fn listing_only(&self) -> bool {
        self.saw_listing && !self.saw_search && !self.saw_read
    }

    pub(crate) fn has_discovery(&self) -> bool {
        self.saw_listing || self.saw_search || self.saw_read
    }

    pub(crate) fn discovery_depth(&self) -> &'static str {
        let kinds = usize::from(self.saw_listing)
            + usize::from(self.saw_search)
            + usize::from(self.saw_read);
        match (kinds, self.saw_listing, self.saw_search, self.saw_read) {
            (0, _, _, _) => "none",
            (1, true, false, false) => "listing_only",
            (1, false, true, false) => "targeted_search",
            (1, false, false, true) => "file_read",
            _ => "mixed",
        }
    }

    pub(crate) fn first_tool_kind(&self) -> &'static str {
        self.first_tool_kind
            .map(EvidenceKind::as_str)
            .unwrap_or("none")
    }

    pub(crate) fn security_search_complete(&self) -> bool {
        self.security_unsafe_search && self.security_execution_search && self.security_secret_search
    }

    pub(crate) fn record_search_hit_snippets(&mut self, output: &str) {
        const SEARCH_HIT_SNIPPET_LIMIT: usize = 8;
        let mut candidates = self.search_hit_snippets.clone();
        for line in output.lines() {
            let snippet = compact_search_hit_line(line);
            if snippet.is_empty()
                || search_hit_score(&snippet) == 0
                || candidates.iter().any(|existing| existing == &snippet)
            {
                continue;
            }
            candidates.push(snippet);
        }
        candidates.sort_by(|left, right| {
            search_hit_score(right)
                .cmp(&search_hit_score(left))
                .then_with(|| left.cmp(right))
        });
        candidates.truncate(SEARCH_HIT_SNIPPET_LIMIT);
        self.search_hit_snippets = candidates;
    }
}

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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SecuritySearchFamilies {
    pub(crate) unsafe_or_panic: bool,
    pub(crate) execution_or_fs_env: bool,
    pub(crate) secret_or_auth: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct PreflightCall {
    pub(crate) name: &'static str,
    pub(crate) arguments: String,
}

impl PreflightCall {
    pub(crate) fn new(name: &'static str, arguments: serde_json::Value) -> Self {
        Self {
            name,
            arguments: arguments.to_string(),
        }
    }

    pub(crate) fn read(path: impl Into<String>, limit: u32) -> Self {
        Self::new(
            "read",
            serde_json::json!({
                "path": path.into(),
                "limit": limit,
            }),
        )
    }
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

pub(crate) fn security_search_families_for_tool(name: &str, arguments: &str) -> SecuritySearchFamilies {
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
            ],
        )
    {
        return None;
    }
    let words: Vec<&str> = normalized.split_whitespace().collect();
    let has_direct_action = words
        .iter()
        .any(|word| matches!(*word, "build" | "create" | "make" | "develop" | "scaffold"));
    let has_generic_action = words
        .iter()
        .any(|word| matches!(*word, "implement" | "write"));
    if !has_direct_action && !has_generic_action {
        return None;
    }
    let has_artifact = words.iter().any(|word| {
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
    }) || contains_any(
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

pub(crate) fn read_only_preflight_initial_calls(intent: ReviewIntent) -> Vec<PreflightCall> {
    let mut calls = Vec::new();
    if matches!(
        intent,
        ReviewIntent::Review | ReviewIntent::Status | ReviewIntent::Roadmap | ReviewIntent::Gaps
    ) {
        calls.push(PreflightCall::new("diff", serde_json::json!({})));
    }
    push_preflight_read_if_exists(&mut calls, "Cargo.toml", 100);
    if !matches!(intent, ReviewIntent::Security) {
        push_preflight_read_if_exists(&mut calls, "README.md", 100);
    }

    match intent {
        ReviewIntent::Security => {
            calls.push(PreflightCall::new(
                "grep",
                serde_json::json!({
                    "pattern": SECURITY_PREFLIGHT_PATTERN,
                    "context": 0,
                    "glob": "*.rs",
                }),
            ));
        }
        ReviewIntent::Roadmap | ReviewIntent::Gaps => {
            calls.extend(
                preflight_entrypoint_candidates()
                    .into_iter()
                    .take(3)
                    .map(|path| PreflightCall::read(path, 180)),
            );
            calls.push(PreflightCall::new(
                "grep",
                serde_json::json!({
                    "pattern": GAP_PREFLIGHT_PATTERN,
                    "context": 0,
                }),
            ));
        }
        ReviewIntent::Review | ReviewIntent::Status => {
            calls.extend(
                preflight_entrypoint_candidates()
                    .into_iter()
                    .take(3)
                    .map(|path| PreflightCall::read(path, 180)),
            );
        }
    }
    calls
}

pub(crate) fn push_preflight_read_if_exists(calls: &mut Vec<PreflightCall>, path: &str, limit: u32) {
    if std::path::Path::new(path).is_file() {
        calls.push(PreflightCall::read(path, limit));
    }
}

pub(crate) fn preflight_entrypoint_candidates() -> Vec<String> {
    let mut candidates = Vec::new();
    for path in ["src/lib.rs", "src/main.rs"] {
        if std::path::Path::new(path).is_file() {
            candidates.push(path.to_string());
        }
    }

    let Ok(entries) = std::fs::read_dir("crates") else {
        return candidates;
    };
    let mut crate_dirs = entries
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.file_type().is_ok_and(|file_type| file_type.is_dir()))
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    crate_dirs.sort();
    for crate_dir in crate_dirs {
        for file in ["src/lib.rs", "src/main.rs"] {
            let path = crate_dir.join(file);
            if path.is_file() {
                candidates.push(path.to_string_lossy().to_string());
            }
        }
    }
    candidates
}

pub(crate) fn paths_from_grep_output(output: &str) -> Vec<String> {
    let mut paths = Vec::new();
    for line in output.lines() {
        let Some((path, _)) = line.split_once(':') else {
            continue;
        };
        let path = path.trim();
        if path.is_empty()
            || path.starts_with("no matches")
            || path.starts_with("Error:")
            || !std::path::Path::new(path).is_file()
            || paths.iter().any(|existing| existing == path)
        {
            continue;
        }
        paths.push(path.to_string());
        if paths.len() >= 4 {
            break;
        }
    }
    paths
}

pub(crate) fn preflight_path_relevant_for_intent(intent: ReviewIntent, path: &str) -> bool {
    if !matches!(intent, ReviewIntent::Security) {
        return true;
    }
    let lower = path.to_ascii_lowercase();
    matches!(
        lower.rsplit('.').next(),
        Some(
            "rs" | "toml"
                | "lock"
                | "sh"
                | "bash"
                | "zsh"
                | "py"
                | "js"
                | "jsx"
                | "ts"
                | "tsx"
                | "go"
                | "java"
                | "kt"
                | "kts"
                | "rb"
                | "php"
                | "c"
                | "cc"
                | "cpp"
                | "h"
                | "hpp"
        )
    )
}

pub(crate) fn compact_preflight_tool_output(name: &str, output: &str) -> String {
    let max_lines = match name {
        "grep" => READ_ONLY_PREFLIGHT_GREP_MAX_LINES,
        "diff" => READ_ONLY_PREFLIGHT_DIFF_MAX_LINES,
        _ => return output.to_string(),
    };
    let mut lines = output.lines().collect::<Vec<_>>();
    if lines.len() <= max_lines {
        return output.to_string();
    }
    let omitted = lines.len().saturating_sub(max_lines);
    lines.truncate(max_lines);
    format!(
        "{}\n[preflight {name} output truncated: {omitted} additional line(s) omitted]",
        lines.join("\n")
    )
}

pub(crate) fn implementation_preflight_command() -> &'static str {
    r#"set +e
printf '[git_status]\n'
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git status --short 2>&1 | head -80
else
  printf 'not a git repository\n'
fi
printf '\n[git_diff_stat]\n'
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git diff --stat 2>&1 | head -80
else
  printf 'not a git repository\n'
fi
printf '\n[workspace_manifests]\n'
find . -path './target' -prune -o -path './node_modules' -prune -o -path './.git' -prune -o \( -name Cargo.toml -o -name package.json -o -name pyproject.toml -o -name go.mod -o -name Makefile -o -name justfile \) -print | sort | head -80
printf '\n[readme_docs]\n'
find . -maxdepth 3 -path './.git' -prune -o \( -iname 'README*' -o -iname 'DESIGN*' -o -path './docs/*' \) -print | sort | head -80
printf '\n[likely_entrypoints]\n'
find . -path './target' -prune -o -path './node_modules' -prune -o -path './.git' -prune -o \( -path './src/main.rs' -o -path './src/lib.rs' -o -path './src/bin/*.rs' -o -path './main.py' -o -path './app.py' -o -path './src/index.*' -o -path './src/main.*' -o -path './app/page.*' \) -print | sort | head -80
printf '\n[detected_verification]\n'
if find . -maxdepth 3 -path './target' -prune -o -name Cargo.toml -print -quit | grep -q .; then
  printf 'primary=cargo test\n'
  printf 'alternates=cargo check; cargo build\n'
elif [ -f package.json ]; then
  if grep -q '"test"' package.json; then printf 'primary=npm test\n'; else printf 'primary=npm run build\n'; fi
  printf 'alternates=npm run check; npm run lint\n'
elif [ -f pyproject.toml ]; then
  printf 'primary=python -m pytest\n'
elif [ -f go.mod ]; then
  printf 'primary=go test ./...\n'
elif [ -f Makefile ]; then
  printf 'primary=make test\n'
else
  printf 'primary=inspect manifests first\n'
fi
"#
}

pub(crate) fn preferred_validation_from_preflight(output: &str) -> Option<String> {
    for line in output.lines() {
        let Some(command) = line.trim().strip_prefix("primary=") else {
            continue;
        };
        let command = command.trim();
        if !command.is_empty() && command != "inspect manifests first" {
            return Some(command.to_string());
        }
    }
    None
}

pub(crate) fn should_bootstrap_gpu_training_estimator(intent: ImplementationIntent) -> bool {
    intent.gpu_training_estimator && implementation_workspace_can_accept_rust_bootstrap()
}

pub(crate) fn implementation_workspace_can_accept_rust_bootstrap() -> bool {
    implementation_workspace_can_accept_rust_bootstrap_at(std::path::Path::new("."))
}

pub(crate) fn implementation_workspace_can_accept_rust_bootstrap_at(root: &std::path::Path) -> bool {
    let manifest_paths = [
        "Cargo.toml",
        "package.json",
        "pyproject.toml",
        "go.mod",
        "Makefile",
        "justfile",
    ];
    if manifest_paths.iter().any(|path| root.join(path).exists()) {
        return false;
    }
    if ["src/main.rs", "src/lib.rs"]
        .iter()
        .any(|path| root.join(path).exists())
    {
        return false;
    }
    if let Ok(mut entries) = std::fs::read_dir(root.join("src"))
        && entries.next().is_some()
    {
        return false;
    }
    true
}

pub(crate) fn gpu_training_estimator_bootstrap_files(
    intent: ImplementationIntent,
) -> Vec<(&'static str, String)> {
    vec![
        (
            "Cargo.toml",
            gpu_training_estimator_bootstrap_cargo_toml(intent),
        ),
        ("src/lib.rs", gpu_training_estimator_bootstrap_lib_rs()),
        (
            "src/main.rs",
            gpu_training_estimator_bootstrap_main_rs(intent),
        ),
    ]
}

pub(crate) fn gpu_training_estimator_bootstrap_cargo_toml(intent: ImplementationIntent) -> String {
    if intent.tui {
        include_str!("../templates/gpu_estimator/Cargo.toml.tui").to_string()
    } else {
        include_str!("../templates/gpu_estimator/Cargo.toml.cli").to_string()
    }
}

pub(crate) fn gpu_training_estimator_bootstrap_lib_rs() -> String {
    include_str!("../templates/gpu_estimator/lib.rs").to_string()
}

pub(crate) fn gpu_training_estimator_bootstrap_main_rs(intent: ImplementationIntent) -> String {
    if intent.tui {
        include_str!("../templates/gpu_estimator/main.rs.tui").to_string()
    } else {
        include_str!("../templates/gpu_estimator/main.rs.cli").to_string()
    }
}
pub(crate) fn bash_command(arguments: &str) -> Option<String> {
    let value = serde_json::from_str::<serde_json::Value>(arguments).ok()?;
    value
        .get("command")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
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

pub(crate) fn implementation_missing_validation_nudge(tracker: &ImplementationTracker) -> String {
    let preferred = tracker
        .preferred_validation
        .as_deref()
        .map(|command| format!(" Prefer `{command}`."))
        .unwrap_or_default();
    format!("{IMPLEMENTATION_MISSING_VALIDATION_NUDGE}{preferred}")
}

pub(crate) fn implementation_text_tool_nudge(reason: &str) -> String {
    format!("{reason}\n\n{TOOL_PROTOCOL_TEXT_FALLBACK_NUDGE}")
}

pub(crate) fn answer_says_insufficient_evidence(content: &str) -> bool {
    let lower = content.to_ascii_lowercase();
    contains_any(
        &lower,
        &[
            "insufficient evidence",
            "not enough evidence",
            "not enough information",
            "only a directory listing",
            "only saw a listing",
            "need to inspect",
            "need file reads",
            "need targeted search",
        ],
    )
}

pub(crate) fn should_deepen_review(
    intent: Option<ReviewIntent>,
    evidence: &EvidenceTracker,
    answer: &str,
) -> bool {
    intent.is_some() && evidence.listing_only() && !answer_says_insufficient_evidence(answer)
}

pub(crate) fn should_nudge_no_evidence_review(
    intent: Option<ReviewIntent>,
    evidence: &EvidenceTracker,
    answer: &str,
) -> bool {
    intent.is_some() && !evidence.has_discovery() && !answer_says_insufficient_evidence(answer)
}

pub(crate) fn answer_looks_like_review_repair_template(content: &str) -> bool {
    let lower = content.to_ascii_lowercase();
    contains_any(
        &lower,
        &[
            "the inspected context points to these concrete review targets",
            "review observations should stay tied to those files or modules",
            "convert any broad status claims into file-specific findings",
            "the inspected context identifies these concrete targets as the likely ownership surface",
            "gap claims should be tied to those inspected files or modules",
            "convert broad recommendations into scoped work items tied to the inspected files",
        ],
    )
}

pub(crate) fn should_reject_review_repair_template(intent: Option<ReviewIntent>, answer: &str) -> bool {
    intent.is_some()
        && !answer_says_insufficient_evidence(answer)
        && answer_looks_like_review_repair_template(answer)
}

pub(crate) fn should_nudge_concrete_review_answer(
    intent: Option<ReviewIntent>,
    evidence: &EvidenceTracker,
    answer: &str,
) -> bool {
    let Some(intent) = intent else {
        return false;
    };
    if evidence.inspected_paths.is_empty() || answer_says_insufficient_evidence(answer) {
        return false;
    }
    let cites_inspected_path = evidence
        .inspected_paths
        .iter()
        .any(|path| answer.contains(path));
    !cites_inspected_path
        || answer_looks_like_generic_inventory_summary(answer)
        || answer_lacks_review_shape(intent, answer)
}

pub(crate) fn answer_lacks_review_shape(intent: ReviewIntent, answer: &str) -> bool {
    let lower = answer.to_ascii_lowercase();
    let has_evidence_language = contains_any(
        &lower,
        &[
            "inspected",
            "reviewed",
            "evidence",
            "findings:",
            "based on",
            "limited to",
            "from the inspected",
            "in the inspected",
        ],
    );
    let has_review_language = match intent {
        ReviewIntent::Security => contains_any(
            &lower,
            &[
                "finding",
                "security",
                "unsafe",
                "unwrap",
                "expect",
                "panic",
                "secret",
                "token",
                "auth",
                "risk",
                "follow-up",
                "follow up",
            ],
        ),
        ReviewIntent::Status => contains_any(
            &lower,
            &[
                "status",
                "state",
                "build next",
                "risk",
                "validation",
                "follow-up",
                "follow up",
            ],
        ),
        ReviewIntent::Roadmap | ReviewIntent::Gaps => contains_any(
            &lower,
            &[
                "missing",
                "gap",
                "roadmap",
                "build next",
                "risk",
                "coverage",
                "follow-up",
                "follow up",
            ],
        ),
        ReviewIntent::Review => contains_any(
            &lower,
            &[
                "finding",
                "reviewed",
                "status",
                "risk",
                "validation",
                "tests pass",
                "test pass",
                "follow-up",
                "follow up",
            ],
        ),
    };
    !(has_evidence_language && has_review_language)
}

pub(crate) fn answer_looks_like_generic_inventory_summary(answer: &str) -> bool {
    let lower = answer.to_ascii_lowercase();
    let inventory_markers = [
        "codebase is",
        "project is",
        "repository is",
        "structured with",
        "consists of",
        "main components",
        "main functionality",
        "key features",
        "workspace setup",
        "entry point",
        "support for multiple",
        "supports multiple",
        "the exact count can be determined",
        "approximately ",
    ];
    let marker_count = inventory_markers
        .iter()
        .filter(|marker| lower.contains(**marker))
        .count();
    let has_bounded_review_language = contains_any(
        &lower,
        &[
            "findings:",
            "status:",
            "evidence:",
            "inspected evidence",
            "risks/validation",
            "build next",
            "missing/gaps",
            "limits:",
            "based on the inspected",
            "from the inspected",
            "not a complete",
        ],
    );
    marker_count >= 2 && !has_bounded_review_language
}

pub(crate) fn should_nudge_read_after_repeated_search(
    intent: Option<ReviewIntent>,
    evidence: &EvidenceTracker,
) -> bool {
    intent.is_some() && evidence.saw_search && !evidence.saw_read
}

pub(crate) fn should_nudge_read_after_search_final(
    intent: Option<ReviewIntent>,
    evidence: &EvidenceTracker,
    answer: &str,
) -> bool {
    intent.is_some()
        && evidence.saw_search
        && !evidence.saw_read
        && !answer_says_insufficient_evidence(answer)
}

pub(crate) fn should_nudge_security_broad_search(
    intent: Option<ReviewIntent>,
    evidence: &EvidenceTracker,
    answer: &str,
) -> bool {
    matches!(intent, Some(ReviewIntent::Security))
        && evidence.saw_search
        && evidence.saw_read
        && !evidence.security_search_complete()
        && !answer_says_insufficient_evidence(answer)
}

pub(crate) fn should_nudge_security_scope(
    intent: Option<ReviewIntent>,
    evidence: &EvidenceTracker,
    answer: &str,
) -> bool {
    matches!(intent, Some(ReviewIntent::Security))
        && evidence.saw_search
        && evidence.saw_read
        && security_answer_overclaims_scope(answer)
}

pub(crate) fn should_nudge_gap_search_overclaim(
    intent: Option<ReviewIntent>,
    evidence: &EvidenceTracker,
    answer: &str,
) -> bool {
    matches!(intent, Some(ReviewIntent::Gaps | ReviewIntent::Roadmap))
        && evidence.grep_match_lines > 0
        && gap_answer_overclaims_absence(answer)
}

pub(crate) fn security_answer_overclaims_scope(answer: &str) -> bool {
    if answer_says_insufficient_evidence(answer) {
        return false;
    }
    let lower = answer.to_ascii_lowercase();
    let broad_all_clear = contains_any(
        &lower,
        &[
            "the codebase does not contain",
            "the codebase doesn't contain",
            "the codebase appears to be secure",
            "codebase appears secure",
            "secure against common unsafe patterns",
            "there are no hardcoded secrets",
            "no hardcoded secrets",
            "no direct command execution",
            "does not contain any unsafe",
            "doesn't contain any unsafe",
            "no security issues",
            "no security-critical issues",
        ],
    );
    let bounded = contains_any(
        &lower,
        &[
            "insufficient evidence",
            "limited to",
            "based on the inspected",
            "based on searched",
            "based on the searched",
            "from the inspected",
            "in the inspected",
            "i only inspected",
            "not a complete audit",
            "cannot rule out",
            "cannot make broad",
        ],
    );
    broad_all_clear && !bounded
}

pub(crate) fn gap_answer_overclaims_absence(answer: &str) -> bool {
    if answer_says_insufficient_evidence(answer) {
        return false;
    }
    let lower = answer.to_ascii_lowercase();
    let broad_absence = contains_any(
        &lower,
        &[
            "no todo",
            "no todos",
            "no todo/fixme",
            "no fixme",
            "no fixmes",
            "no missing implementations",
            "no obvious gaps",
            "no obvious missing",
            "no obvious gaps in functionality",
            "appears mature with no obvious gaps",
            "shows no obvious gaps",
        ],
    );
    let bounded = contains_any(
        &lower,
        &[
            "based on the inspected",
            "based on searched",
            "based on the searched",
            "from the inspected",
            "in the inspected",
            "i only inspected",
            "not a complete",
            "cannot rule out",
            "cannot make broad",
        ],
    );
    broad_absence && !bounded
}

pub(crate) fn insufficient_after_repeated_search(evidence: &EvidenceTracker) -> Option<&'static str> {
    if evidence.saw_search && !evidence.saw_read {
        Some(
            "Insufficient evidence: targeted search ran, but no matching file was read, so I cannot make file-specific review findings.",
        )
    } else {
        None
    }
}

pub(crate) fn insufficient_after_incomplete_security_search(evidence: &EvidenceTracker) -> Option<String> {
    if !evidence.saw_search || !evidence.saw_read || evidence.security_search_complete() {
        return None;
    }
    let mut missing = Vec::new();
    if !evidence.security_unsafe_search {
        missing.push("unsafe/unwrap/expect/panic");
    }
    if !evidence.security_execution_search {
        missing.push("command execution/filesystem/env");
    }
    if !evidence.security_secret_search {
        missing.push("secret/token/auth");
    }
    Some(format!(
        "Insufficient evidence: the security review did not search all required pattern families (missing {}), so I cannot make broad security claims.",
        missing.join(", ")
    ))
}

pub(crate) fn insufficient_after_security_scope_overclaim() -> &'static str {
    "Insufficient evidence: the security answer made repo-wide all-clear claims that were broader than the inspected files and search results support."
}

pub(crate) fn insufficient_after_no_review_evidence() -> &'static str {
    "Insufficient evidence: no files, searches, diffs, or directory listings were inspected, so I cannot present this as a completed review."
}

pub(crate) fn insufficient_after_review_repair_template() -> &'static str {
    "Insufficient evidence: the answer was a generic review-repair template instead of concrete findings tied to inspected files, so I cannot present this as a completed review."
}

pub(crate) fn read_only_intent_label(intent: ReviewIntent) -> &'static str {
    match intent {
        ReviewIntent::Security => "security review",
        ReviewIntent::Status => "status review",
        ReviewIntent::Roadmap => "roadmap review",
        ReviewIntent::Gaps => "gap review",
        ReviewIntent::Review => "review",
    }
}

pub(crate) fn bounded_review_repair_exhaustion_answer(
    intent: ReviewIntent,
    evidence: &EvidenceTracker,
    reason: &str,
) -> String {
    let label = read_only_intent_label(intent);
    let mut lines = vec![
        format!(
            "Bounded evidence summary for an incomplete {label}: the model inspected evidence but did not produce acceptable file-specific findings after repair."
        ),
        String::new(),
        "Inspected evidence:".to_string(),
        format!("- Targeted searches: {}", evidence.targeted_searches),
        format!("- File reads: {}", evidence.file_reads),
    ];

    if matches!(intent, ReviewIntent::Security) {
        let mut families = Vec::new();
        if evidence.security_unsafe_search {
            families.push("unsafe/unwrap/expect/panic");
        }
        if evidence.security_execution_search {
            families.push("command execution/filesystem/env");
        }
        if evidence.security_secret_search {
            families.push("secret/token/auth");
        }
        let searched = if families.is_empty() {
            "none".to_string()
        } else {
            families.join(", ")
        };
        lines.push(format!("- Security pattern families searched: {searched}"));
    }

    if evidence.inspected_paths.is_empty() {
        lines.push("- Inspected files: none".to_string());
    } else {
        const INSPECTED_PATH_FALLBACK_LIMIT: usize = 6;
        let mut paths = evidence
            .inspected_paths
            .iter()
            .take(INSPECTED_PATH_FALLBACK_LIMIT)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        let omitted = evidence
            .inspected_paths
            .len()
            .saturating_sub(INSPECTED_PATH_FALLBACK_LIMIT);
        if omitted > 0 {
            paths.push_str(&format!(" (+{omitted} more)"));
        }
        lines.push(format!("- Inspected files: {paths}"));
    }

    if !evidence.search_hit_snippets.is_empty() {
        const SEARCH_HIT_FALLBACK_LIMIT: usize = 6;
        lines.push(String::new());
        lines.push("Concrete search matches from inspected evidence:".to_string());
        for snippet in evidence
            .search_hit_snippets
            .iter()
            .take(SEARCH_HIT_FALLBACK_LIMIT)
        {
            lines.push(format!("- {snippet}"));
        }
        let omitted = evidence
            .search_hit_snippets
            .len()
            .saturating_sub(SEARCH_HIT_FALLBACK_LIMIT);
        if omitted > 0 {
            lines.push(format!("- (+{omitted} more search match target(s))"));
        }
        lines.push(
            "These are pattern-match review targets, not confirmed vulnerabilities or all-clear findings."
                .to_string(),
        );
    }

    lines.push(String::new());
    lines.push(format!("Why this stopped: {reason}."));
    lines.push(
        "No file is being changed; this turn remains read-only and no broader repo-wide claim is being made."
            .to_string(),
    );
    lines.join("\n")
}

pub(crate) fn inspected_paths_for_prompt(evidence: &EvidenceTracker) -> String {
    if evidence.inspected_paths.is_empty() {
        return "none".to_string();
    }
    const LIMIT: usize = 8;
    let mut paths = evidence
        .inspected_paths
        .iter()
        .take(LIMIT)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    let omitted = evidence.inspected_paths.len().saturating_sub(LIMIT);
    if omitted > 0 {
        paths.push_str(&format!(" (+{omitted} more)"));
    }
    paths
}

pub(crate) fn summarize_inspected_evidence_nudge(intent: ReviewIntent, evidence: &EvidenceTracker) -> String {
    let label = read_only_intent_label(intent);
    let paths = inspected_paths_for_prompt(evidence);
    match intent {
        ReviewIntent::Security => format!(
            "You already have inspected evidence for this {label}. Do not answer with generic insufficient-evidence text. Produce a bounded security review from only the inspected searches/files. Cite concrete inspected files from this set: {paths}. Include: Findings, Inspected Evidence, Limits, and Follow-up. If a pattern match is only a review target, say that it is not confirmed."
        ),
        ReviewIntent::Status => format!(
            "You already have inspected evidence for this {label}. Do not answer with generic insufficient-evidence text. Produce a bounded status review from only the inspected files. Cite concrete inspected files from this set: {paths}. Include: Status, Evidence, Build Next, and Risks/Validation. Do not claim repo-wide completeness."
        ),
        ReviewIntent::Roadmap | ReviewIntent::Gaps => format!(
            "You already have inspected evidence for this {label}. Do not answer with generic insufficient-evidence text. Produce bounded gaps/build-next notes from only the inspected files and searches. Cite concrete inspected files from this set: {paths}. Include: Missing/Gaps, Build Next, Evidence, and Risks. Do not claim repo-wide completeness."
        ),
        ReviewIntent::Review => format!(
            "You already have inspected evidence for this {label}. Do not answer with generic insufficient-evidence text. Produce bounded findings from only the inspected files/searches. Cite concrete inspected files from this set: {paths}. Include findings, evidence, follow-up, and limits."
        ),
    }
}

pub(crate) fn inspected_insufficient_repair_limit(intent: ReviewIntent) -> u32 {
    match intent {
        ReviewIntent::Security => 3,
        ReviewIntent::Status
        | ReviewIntent::Roadmap
        | ReviewIntent::Gaps
        | ReviewIntent::Review => 2,
    }
}

pub(crate) fn no_evidence_review_nudge(intent: ReviewIntent) -> &'static str {
    match intent {
        ReviewIntent::Security => NO_EVIDENCE_SECURITY_NUDGE,
        ReviewIntent::Status => NO_EVIDENCE_STATUS_NUDGE,
        ReviewIntent::Roadmap | ReviewIntent::Gaps => NO_EVIDENCE_GAP_NUDGE,
        ReviewIntent::Review => NO_EVIDENCE_REVIEW_NUDGE,
    }
}

pub(crate) fn deepen_review_nudge(intent: ReviewIntent) -> &'static str {
    match intent {
        ReviewIntent::Security => SECURITY_DEEPEN_NUDGE,
        ReviewIntent::Status => STATUS_DEEPEN_NUDGE,
        ReviewIntent::Roadmap | ReviewIntent::Gaps => GAP_DEEPEN_NUDGE,
        ReviewIntent::Review => REVIEW_DEEPEN_NUDGE,
    }
}

pub(crate) fn read_only_blocks_tool(intent: Option<ReviewIntent>, name: &str) -> bool {
    intent.is_some() && !hi_tools::is_read_only(name)
}

pub(crate) fn read_only_blocked_tool_result(name: &str) -> String {
    format!(
        "Tool `{name}` blocked: this is a read-only review/discuss-only turn. Use read-only inspection tools and answer from inspected evidence; do not modify files."
    )
}

