use super::*;
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
