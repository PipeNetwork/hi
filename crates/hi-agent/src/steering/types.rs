//! Core steering types: [`ReviewIntent`], [`ImplementationIntent`],
//! [`EvidenceTracker`], [`ImplementationTracker`], [`PreflightCall`], and
//! [`SecuritySearchFamilies`]. The tracker impls call into
//! [`intent`](super::intent) and [`implementation`](super::implementation)
//! for evidence classification and tool-call inspection.

use super::implementation::{
    bash_no_progress_signature, implementation_tool_call_validates,
    implementation_tool_result_landed_mutation, implementation_tool_result_landed_substantive_edit,
};
use super::intent::{
    compact_search_hit_line, evidence_kind_for_tool, grep_match_line_count, search_hit_score,
    security_search_families_for_tool,
};

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
        if implementation_tool_result_landed_mutation(name, arguments, output) {
            self.mutation_seen = true;
            if implementation_tool_result_landed_substantive_edit(name, arguments, output) {
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
    /// How many inspection-sprawl nudges have fired this turn. Incremented in
    /// the turn loop when a read-only review turn keeps issuing read-only
    /// inspections past the active intent-specific inspection cap without producing a
    /// final answer. Once this exceeds [`MAX_INSPECTION_SPRAWL_NUDGES`] the
    /// turn stops incomplete rather than fabricating a review.
    pub(crate) inspection_sprawl_nudges: u32,
    /// Inspection signatures already seen this turn, used by the no-new-evidence
    /// cycle guard. Each entry is a stable key derived from a read-only tool
    /// call's identity: `read:<path>:<offset>:<limit>`,
    /// `list:<path>`, `grep:<pattern>:<glob>:<path>:<context>`,
    /// `glob:<pattern>:<path>`, a stale background handle
    /// `bash_output:<id>`/`bash_kill:<id>`, or a narrow no-progress bash command.
    /// A round whose
    /// every read-only call's signature is already in this set adds no new
    /// evidence — re-running it can only reproduce prior output. Live
    /// `bash_output` polls are intentionally not recorded here because a running
    /// background process can emit new output later; missing/pruned/completed
    /// handles are recorded because polling them again cannot produce new
    /// output. Mutating tools are never added here; ordinary bash still counts
    /// as potentially new, but a tightly recognized no-op/control bash command
    /// gets a signature so stop/quit/done loops are bounded.
    pub(crate) seen_signatures: Vec<String>,
}

impl EvidenceTracker {
    pub(crate) fn record_success(&mut self, name: &str, arguments: &str, output: &str) {
        if output.starts_with("Error:") {
            self.record_inspection_signature(name, arguments);
            return;
        }
        if background_handle_is_terminal(name, output) {
            self.record_inspection_signature(name, arguments);
        }
        if name == "bash" {
            self.record_inspection_signature(name, arguments);
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
        // Record the inspection signature so the no-new-evidence guard can
        // spot a later round re-running the same inspection. Only read-only
        // discovery tools get a signature; mutating tools are never cyclic in
        // this sense (a re-edit is handled by the implementation tracker).
        self.record_inspection_signature(name, arguments);
    }

    /// Whether a proposed round of calls would add any new evidence. Returns
    /// `true` if the round is empty, contains a mutating tool, or contains any
    /// read-only call whose inspection signature has not been seen yet this
    /// turn. Returns `false` only when every call is a read-only inspection
    /// already performed earlier — i.e. re-running the round can only reproduce
    /// prior output. Used by the cycle guard to detect multi-step read/search
    /// cycles (A→B→C→A→B→C) that evade the exact-match repeat guard.
    pub(crate) fn round_adds_evidence(&self, calls: &[(String, String, String)]) -> bool {
        if calls.is_empty() {
            return true;
        }
        for (_, name, args) in calls {
            match name.as_str() {
                "read" | "list" | "grep" | "glob" | "bash_output" | "bash_kill" | "bash" => {
                    match inspection_signature(name, args) {
                        Some(sig) if self.seen_signatures.iter().any(|s| s == &sig) => {}
                        // A new signature, or arguments we cannot signature safely,
                        // should execute. The normal tool path will surface malformed
                        // arguments; the cycle guard must not hide them.
                        _ => return true,
                    }
                }
                // Any mutating or unclassified tool counts as potentially new
                // evidence — don't let the cycle guard suppress real work.
                _ => return true,
            }
        }
        false
    }

    fn record_inspection_signature(&mut self, name: &str, arguments: &str) {
        if let Some(sig) = inspection_signature(name, arguments)
            && !self.seen_signatures.iter().any(|s| s == &sig)
        {
            self.seen_signatures.push(sig);
        }
    }

    pub(crate) fn listing_only(&self) -> bool {
        self.saw_listing && !self.saw_search && !self.saw_read
    }

    pub(crate) fn has_discovery(&self) -> bool {
        self.saw_listing || self.saw_search || self.saw_read
    }

    /// Total read-only inspection count (file reads + targeted searches).
    /// Used by the inspection-sprawl guard to decide when a read-only review
    /// turn has gathered "enough" evidence to stop inspecting and answer.
    pub(crate) fn inspection_count(&self) -> u32 {
        self.file_reads.saturating_add(self.targeted_searches)
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

/// A stable signature for a read-only inspection call, used to detect rounds
/// that re-inspect already-seen evidence. Returns `None` for mutating or
/// unclassified tools (those always count as potentially new evidence). The
/// signature includes read pagination and grep context because those
/// arguments change the evidence returned by the tool. A malformed read-only
/// call returns `None`; callers treat that as potentially new evidence so the
/// normal tool execution path can report the argument error.
pub(crate) fn inspection_signature(name: &str, arguments: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(arguments).ok()?;
    match name {
        "read" => {
            let path = value.get("path")?.as_str()?;
            if path.is_empty() {
                return None;
            }
            let offset = optional_u64_field(&value, "offset")?.unwrap_or(1).max(1);
            let limit = optional_u64_field(&value, "limit")?
                .map(|n| n.to_string())
                .unwrap_or_else(|| "default".to_string());
            Some(format!("read:{path}:{offset}:{limit}"))
        }
        "list" => {
            let path = value.get("path").and_then(|v| v.as_str()).unwrap_or(".");
            Some(format!("list:{path}"))
        }
        "grep" => {
            let pattern = value.get("pattern")?.as_str()?;
            let glob = value.get("glob").and_then(|v| v.as_str()).unwrap_or("");
            let path = value.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let context = optional_u64_field(&value, "context")?.unwrap_or(0);
            Some(format!("grep:{pattern}:{glob}:{path}:{context}"))
        }
        "glob" => {
            let pattern = value.get("pattern")?.as_str()?;
            let path = value.get("path").and_then(|v| v.as_str()).unwrap_or("");
            Some(format!("glob:{pattern}:{path}"))
        }
        "bash_output" | "bash_kill" => {
            let id = value.get("id")?.as_str()?;
            if id.is_empty() {
                return None;
            }
            Some(format!("{name}:{id}"))
        }
        "bash" => bash_no_progress_signature(arguments).map(|sig| format!("bash:{sig}")),
        _ => None,
    }
}

fn optional_u64_field(value: &serde_json::Value, field: &str) -> Option<Option<u64>> {
    match value.get(field) {
        Some(v) if v.is_null() => Some(None),
        Some(v) => v.as_u64().map(Some),
        None => Some(None),
    }
}

fn background_handle_is_terminal(name: &str, output: &str) -> bool {
    match name {
        "bash_output" => {
            let Some(status) = output.lines().next() else {
                return false;
            };
            status.contains(": exited") || status.contains(": killed")
        }
        "bash_kill" => {
            output.starts_with('[')
                && (output.contains("] killed")
                    || output.contains("] already exited")
                    || output.contains("] already killed"))
        }
        _ => false,
    }
}
