//! Core steering types: [`ReviewIntent`], [`ImplementationIntent`],
//! [`EvidenceTracker`], [`ImplementationTracker`], [`PreflightCall`], and
//! [`SecuritySearchFamilies`]. The tracker impls call into
//! [`intent`](super::intent) and [`implementation`](super::implementation)
//! for evidence classification and tool-call inspection.

use std::collections::{HashMap, HashSet, VecDeque};

use hi_ai::{Content, Message};

use super::implementation::{
    bash_inspection_signature, bash_no_progress_signature, implementation_tool_call_validates,
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
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ImplementationTracker {
    pub(crate) mutation_seen: bool,
    pub(crate) substantive_edit_seen: bool,
    /// A mutating call was accepted as a plan under `--dry-run`. This satisfies
    /// the dry-run contract without pretending the workspace actually changed.
    pub(crate) dry_run_mutation_planned: bool,
    /// A successful noninteractive build/test/check command observed anywhere
    /// in this turn. This is separate from post-mutation validation because a
    /// user may ask only to run an existing test suite.
    pub(crate) validation_seen: bool,
    pub(crate) validation_after_last_mutation: bool,
    pub(crate) preferred_validation: Option<String>,
    /// Tool results observed before the first successful mutation. Unlike the
    /// evidence counter this includes coordination, LSP, and subagent calls,
    /// all of which can otherwise sustain an expensive inspect/plan loop.
    pub(crate) pre_mutation_tool_calls: u32,
    /// Model tool-call batches observed before the first successful mutation.
    /// Parallel reads belong to one reasoning round and must not exhaust the
    /// discovery budget faster merely because the model batched them.
    pub(crate) pre_mutation_rounds: u32,
    /// Nudges spent specifically by the bounded pre-mutation discovery guard.
    /// Kept separate from text/repeat repair budgets.
    pub(crate) discovery_nudges: u32,
    pub(crate) no_change_nudges: u32,
    pub(crate) scaffold_only_nudges: u32,
    pub(crate) missing_validation_nudges: u32,
    pub(crate) requested_validation_nudges: u32,
}

impl ImplementationTracker {
    pub(crate) fn record_dry_run_plan(&mut self, mutates: bool) {
        self.dry_run_mutation_planned |= mutates;
    }

    pub(crate) fn record_validation_success(&mut self) {
        self.validation_seen = true;
        if self.mutation_seen {
            self.validation_after_last_mutation = true;
        }
    }

    pub(crate) fn record_tool_round(&mut self) {
        if !self.mutation_seen {
            self.pre_mutation_rounds = self.pre_mutation_rounds.saturating_add(1);
        }
    }

    pub(crate) fn record_tool_result(
        &mut self,
        name: &str,
        arguments: &str,
        output: &str,
        validation_succeeded: bool,
        mutation_applied: bool,
    ) {
        let validation_observed =
            validation_succeeded && implementation_tool_call_validates(name, arguments);
        // Some mutation-capable tools (notably `delegate`) report their exact
        // applied effects in the typed outcome rather than in display text.
        // Keep the typed effect authoritative so a successful delegated edit
        // does not get mistaken for a no-op by the completeness gate.
        if mutation_applied || implementation_tool_result_landed_mutation(name, arguments, output) {
            self.mutation_seen = true;
            if implementation_tool_result_landed_substantive_edit(name, arguments, output) {
                self.substantive_edit_seen = true;
            }
            // A successful validation command can itself create or update a
            // lockfile. Its validation completed after that side effect, so it
            // satisfies both the requested-validation and post-mutation gates.
            self.validation_after_last_mutation = validation_observed;
            if validation_observed {
                self.record_validation_success();
            }
            return;
        }
        if !self.mutation_seen {
            self.pre_mutation_tool_calls = self.pre_mutation_tool_calls.saturating_add(1);
        }
        if validation_observed {
            self.record_validation_success();
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EvidenceKind {
    Listing,
    TargetedSearch,
    FileRead,
}

/// Read-only discovery tools. Explicit user inspection caps and the repeated-
/// evidence guard treat these uniformly.
pub(crate) fn is_read_only_inspection_tool(name: &str) -> bool {
    matches!(
        name,
        "read" | "list" | "grep" | "glob" | "explore" | "repo_map" | "find_symbol"
    )
}

/// Explicit-cap accounting: every read-only inspection except `list` costs one
/// attempt.
pub(crate) fn counts_against_file_inspection_cap(name: &str) -> bool {
    is_read_only_inspection_tool(name) && name != "list"
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
    /// Read/search attempts, including typed failures such as an offset past
    /// EOF. Failed probes still consume an explicit user-supplied cap.
    /// `repo_map` / `explore` / `find_symbol` count as one attempt each.
    pub(crate) inspection_attempts: u32,
    pub(crate) security_unsafe_search: bool,
    pub(crate) security_execution_search: bool,
    pub(crate) security_secret_search: bool,
    pub(crate) grep_match_lines: u32,
    pub(crate) inspected_paths: VecDeque<String>,
    pub(crate) search_hit_snippets: Vec<String>,
    pub(crate) first_tool_kind: Option<EvidenceKind>,
    pub(crate) quality_repair_nudges: u32,
    /// How many inspection-cap nudges have fired this turn. This is incremented
    /// only after a user-supplied cap is reached. Once it exceeds
    /// [`MAX_INSPECTION_SPRAWL_NUDGES`] the turn settles with the evidence
    /// already available rather than fabricating
    /// a review.
    pub(crate) inspection_sprawl_nudges: u32,
    /// Consecutive executed model rounds whose every call was a read-only
    /// inspection tool. Reset when a mutating or unclassified tool runs.
    /// Compared against [`super::constants::INSPECTION_ONLY_ROUND_CAP`].
    pub(crate) inspection_only_rounds: u32,
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
    pub(crate) seen_signatures: VecDeque<String>,
    pub(crate) seen_signature_set: HashSet<String>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) seen_signatures_dropped: u64,
    /// Paths whose latest successful `read` invited another page via the
    /// `read more with offset` footer. A later page of such a path is new
    /// evidence until the read completes.
    pub(crate) truncated_read_paths: VecDeque<String>,
    /// Paths that have already been returned in full this turn. A later
    /// limited slice must not re-open paging — live Flash followed a complete
    /// default read with `offset:70` and we treated that as a truncated file.
    pub(crate) completed_read_paths: VecDeque<String>,
}

impl EvidenceTracker {
    const SIGNATURE_LIMIT: usize = 4_096;
    const PATH_LIMIT: usize = 2_048;

    pub(crate) fn has_seen_signature(&self, signature: &str) -> bool {
        self.seen_signature_set.contains(signature)
    }

    fn record_signature(&mut self, signature: String) {
        if !self.seen_signature_set.insert(signature.clone()) {
            return;
        }
        if self.seen_signatures.len() >= Self::SIGNATURE_LIMIT
            && let Some(evicted) = self.seen_signatures.pop_front()
        {
            self.seen_signature_set.remove(&evicted);
            self.seen_signatures_dropped = self.seen_signatures_dropped.saturating_add(1);
        }
        self.seen_signatures.push_back(signature);
    }

    fn push_path(paths: &mut VecDeque<String>, path: String) {
        if paths.len() >= Self::PATH_LIMIT {
            paths.pop_front();
        }
        paths.push_back(path);
    }

    pub(crate) fn record_success(&mut self, name: &str, arguments: &str, output: &str) {
        let evidence_kind = evidence_kind_for_tool(name, arguments);
        if counts_against_file_inspection_cap(name) {
            self.inspection_attempts = self.inspection_attempts.saturating_add(1);
        }
        if output.starts_with("Error:") {
            if let Some(sig) = inspection_infrastructure_error_signature(name, output) {
                self.record_signature(sig);
            }
            self.record_inspection_signature(name, arguments);
            return;
        }
        if background_handle_is_terminal(name, output) {
            self.record_inspection_signature(name, arguments);
        }
        if name == "bash" {
            self.record_inspection_signature(name, arguments);
        }
        let Some(kind) = evidence_kind else {
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
                {
                    if !self
                        .inspected_paths
                        .iter()
                        .any(|existing| existing == &path)
                    {
                        Self::push_path(&mut self.inspected_paths, path.clone());
                    }
                    self.record_read_page(&path, output);
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
    ///
    /// Extra pages of a **complete** file already read this turn do **not**
    /// count as new evidence — that walk used to burn hours on 100KB sources.
    /// A later page of a still-truncated file does count,
    /// because the default tool-result budget often returns ~100 lines and
    /// the footer tells the model to `read more with offset N`.
    pub(crate) fn round_adds_evidence(&self, calls: &[(String, String, String)]) -> bool {
        if calls.is_empty() {
            return true;
        }
        for (_, name, args) in calls {
            match name.as_str() {
                "read" => {
                    if let Some(sig) = inspection_signature(name, args)
                        && self.has_seen_signature(&sig)
                    {
                        continue;
                    }
                    if let Some(path) = hi_tools::target_path(name, args)
                        && self.path_already_inspected(&path)
                    {
                        if !self.path_read_is_complete(&path) && self.path_read_is_truncated(&path)
                        {
                            return true;
                        }
                        continue;
                    }
                    match inspection_signature(name, args) {
                        Some(sig) if self.has_seen_signature(&sig) => {}
                        _ => return true,
                    }
                }
                "list" | "grep" | "glob" | "bash_output" | "bash_kill" | "bash" => {
                    if name == "grep" && self.has_seen_signature("grep:error:unavailable") {
                        continue;
                    }
                    match inspection_signature(name, args) {
                        Some(sig) if self.has_seen_signature(&sig) => {}
                        // A new signature, or arguments we cannot signature safely,
                        // should execute. The normal tool path will surface malformed
                        // arguments; the cycle guard must not hide them.
                        _ => return true,
                    }
                }
                "explore" | "repo_map" | "find_symbol" => match inspection_signature(name, args) {
                    Some(sig) if self.has_seen_signature(&sig) => {}
                    _ => return true,
                },
                // Any mutating or unclassified tool counts as potentially new
                // evidence — don't let the cycle guard suppress real work.
                _ => return true,
            }
        }
        false
    }

    fn path_already_inspected(&self, path: &str) -> bool {
        self.inspected_paths
            .iter()
            .any(|seen| paths_refer_to_same_file(seen, path))
    }

    fn path_read_is_truncated(&self, path: &str) -> bool {
        self.truncated_read_paths
            .iter()
            .any(|seen| paths_refer_to_same_file(seen, path))
    }

    fn path_read_is_complete(&self, path: &str) -> bool {
        self.completed_read_paths
            .iter()
            .any(|seen| paths_refer_to_same_file(seen, path))
    }

    /// True when every call is a `read` of a file already returned in full.
    /// Those rounds should skip immediately rather than waiting for a second
    /// consecutive no-new-evidence hit.
    pub(crate) fn rereads_only_completed_files(&self, calls: &[(String, String, String)]) -> bool {
        !calls.is_empty()
            && calls.iter().all(|(_, name, args)| {
                name == "read"
                    && hi_tools::target_path(name, args)
                        .is_some_and(|path| self.path_read_is_complete(&path))
            })
    }

    fn record_read_page(&mut self, path: &str, output: &str) {
        if hi_tools::read_output_invites_paging(output) {
            if !self.path_read_is_complete(path) && !self.path_read_is_truncated(path) {
                Self::push_path(&mut self.truncated_read_paths, path.to_string());
            }
        } else {
            self.truncated_read_paths
                .retain(|seen| !paths_refer_to_same_file(seen, path));
            if !self.path_read_is_complete(path) {
                Self::push_path(&mut self.completed_read_paths, path.to_string());
            }
        }
    }

    /// In-turn elision stubs old `read` results. Treat those paths as truncated
    /// again so a later page is real evidence, not a "already returned in full"
    /// skip of content the model no longer has.
    pub(crate) fn reopen_elided_reads(&mut self, messages: &[Message]) {
        let mut ids = HashMap::new();
        for message in messages {
            for block in &message.content {
                if let Content::ToolCall {
                    id,
                    name,
                    arguments,
                } = block
                    && name == "read"
                    && let Some(path) = hi_tools::target_path(name, arguments)
                {
                    ids.insert(id.clone(), path);
                }
            }
        }
        for message in messages {
            for block in &message.content {
                if let Content::ToolResult { call_id, output } = block
                    && output.starts_with("[elided")
                    && let Some(path) = ids.get(call_id)
                {
                    self.completed_read_paths
                        .retain(|seen| !paths_refer_to_same_file(seen, path));
                    if !self.path_read_is_truncated(path) {
                        Self::push_path(&mut self.truncated_read_paths, path.clone());
                    }
                }
            }
        }
    }

    fn record_inspection_signature(&mut self, name: &str, arguments: &str) {
        if let Some(sig) = inspection_signature(name, arguments) {
            self.record_signature(sig);
        }
    }

    pub(crate) fn listing_only(&self) -> bool {
        self.saw_listing && !self.saw_search && !self.saw_read
    }

    pub(crate) fn has_discovery(&self) -> bool {
        self.saw_listing || self.saw_search || self.saw_read
    }

    /// Inspection work spent, whether or not the underlying tool succeeded.
    pub(crate) fn inspection_attempt_count(&self) -> u32 {
        self.inspection_attempts
    }

    /// Track consecutive inspection-only rounds for diagnostics. Empty rounds
    /// (chat-only wrap-up) are ignored; this counter is not a default limit.
    pub(crate) fn record_inspection_round(&mut self, calls: &[(String, String, String)]) {
        if calls.is_empty() {
            return;
        }
        if calls
            .iter()
            .all(|(_, name, _)| is_read_only_inspection_tool(name))
        {
            self.inspection_only_rounds = self.inspection_only_rounds.saturating_add(1);
        } else {
            self.inspection_only_rounds = 0;
        }
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

/// Whether two tool paths name the same file, including absolute vs workspace-
/// relative forms (`crates/foo.rs` vs `/Users/me/proj/crates/foo.rs`).
/// Bare filenames (`lib.rs`) only match exactly, so they cannot collide with
/// every `**/lib.rs` in the tree.
fn paths_refer_to_same_file(a: &str, b: &str) -> bool {
    let a = a.replace('\\', "/").trim_end_matches('/').to_string();
    let b = b.replace('\\', "/").trim_end_matches('/').to_string();
    if a == b {
        return true;
    }
    if a.len() > b.len() && b.contains('/') {
        a.ends_with(&format!("/{b}"))
    } else if b.len() > a.len() && a.contains('/') {
        b.ends_with(&format!("/{a}"))
    } else {
        false
    }
}

/// A stable signature for a read-only inspection call, used to detect rounds
/// that re-inspect already-seen evidence. Returns `None` for mutating or
/// unclassified tools (those always count as potentially new evidence). The
/// signature includes read pagination and grep context because those
/// arguments change the evidence returned by the tool. A malformed read-only
/// call returns `None`; callers treat that as potentially new evidence so the
/// normal tool execution path can report the argument error.
///
/// [`EvidenceTracker::round_adds_evidence`] treats every new page of a still-
/// truncated file as new evidence. The offset stays in the signature so
/// identical pages still fold without imposing an arbitrary page count.
pub(crate) fn inspection_signature(name: &str, arguments: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(arguments).ok()?;
    match name {
        "read" => {
            let path = value.get("path")?.as_str()?;
            if path.is_empty() {
                return None;
            }
            const DEFAULT_READ_LIMIT: u64 = 2000;
            let offset = optional_u64_field(&value, "offset")?.unwrap_or(1).max(1);
            let limit = optional_u64_field(&value, "limit")?
                .map(|n| n.max(1))
                .filter(|&n| n != DEFAULT_READ_LIMIT)
                .map_or_else(|| "default".to_string(), |n| n.to_string());
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
        "repo_map" => {
            let task = value.get("task").and_then(|v| v.as_str()).unwrap_or("");
            let path = value.get("path").and_then(|v| v.as_str()).unwrap_or("");
            Some(format!("repo_map:{task}:{path}"))
        }
        "find_symbol" => {
            let query = value.get("query")?.as_str()?;
            let path = value.get("path").and_then(|v| v.as_str()).unwrap_or("");
            Some(format!("find_symbol:{query}:{path}"))
        }
        "explore" => {
            let task = value
                .get("task")
                .or_else(|| value.get("prompt"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            Some(format!("explore:{task}"))
        }
        "bash_output" | "bash_kill" => {
            let id = value.get("id")?.as_str()?;
            if id.is_empty() {
                return None;
            }
            Some(format!("{name}:{id}"))
        }
        "bash" => bash_inspection_signature(arguments)
            .map(|command| format!("bash:inspection:{command}"))
            .or_else(|| bash_no_progress_signature(arguments).map(|sig| format!("bash:{sig}"))),
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

/// Coarse signature for tool failures that cannot produce new evidence on
/// retry with different arguments (missing `rg` under Seatbelt, etc.).
pub(crate) fn inspection_infrastructure_error_signature(
    name: &str,
    output: &str,
) -> Option<String> {
    if !output.starts_with("Error:") && !output.contains("execvp()") {
        return None;
    }
    let lower = output.to_ascii_lowercase();
    let unavailable = lower.contains("execvp()")
        || (lower.contains("ripgrep") && lower.contains("unavailable"))
        || (name == "grep"
            && lower.contains("no such file")
            && (lower.contains("'rg'") || lower.contains("of 'rg'")));
    unavailable.then(|| format!("{name}:error:unavailable"))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evidence_cycle_diagnostics_are_bounded_without_blocking_new_inspections() {
        let mut evidence = EvidenceTracker::default();
        for index in 0..5_000 {
            evidence.record_signature(format!("signature-{index}"));
        }

        assert_eq!(
            evidence.seen_signatures.len(),
            EvidenceTracker::SIGNATURE_LIMIT
        );
        assert_eq!(
            evidence.seen_signature_set.len(),
            EvidenceTracker::SIGNATURE_LIMIT
        );
        assert_eq!(evidence.seen_signatures_dropped, 904);
        assert!(!evidence.has_seen_signature("signature-0"));
        assert!(evidence.has_seen_signature("signature-4999"));

        for index in 0..2_100 {
            evidence.record_success(
                "read",
                &serde_json::json!({"path": format!("src/{index}.rs")}).to_string(),
                "1\tcontents\n",
            );
        }
        assert_eq!(evidence.inspected_paths.len(), EvidenceTracker::PATH_LIMIT);
        assert_eq!(
            evidence.completed_read_paths.len(),
            EvidenceTracker::PATH_LIMIT
        );
        assert_eq!(
            evidence.inspected_paths.front().map(String::as_str),
            Some("src/52.rs")
        );
        assert_eq!(
            evidence.inspected_paths.back().map(String::as_str),
            Some("src/2099.rs")
        );
    }

    #[test]
    fn successful_validation_is_recorded_without_a_mutation() {
        let mut tracker = ImplementationTracker::default();
        tracker.record_tool_result(
            "bash",
            r#"{"command":"cargo test --quiet"}"#,
            "",
            true,
            false,
        );
        assert!(tracker.validation_seen);
        assert!(!tracker.validation_after_last_mutation);
    }

    #[test]
    fn validation_that_updates_a_lockfile_still_counts_as_validation() {
        let mut tracker = ImplementationTracker::default();
        tracker.record_tool_result(
            "bash",
            r#"{"command":"cargo test --quiet"}"#,
            "",
            true,
            true,
        );
        assert!(tracker.mutation_seen);
        assert!(tracker.validation_seen);
        assert!(tracker.validation_after_last_mutation);
    }

    #[test]
    fn elided_completed_read_is_reopened_for_paging() {
        let mut evidence = EvidenceTracker::default();
        evidence.record_success(
            "read",
            r#"{"path":"crates/hi-tui/src/lib.rs"}"#,
            "   1\tfn main() {}\n",
        );
        assert!(
            evidence.rereads_only_completed_files(&[(
                "c".into(),
                "read".into(),
                r#"{"path":"crates/hi-tui/src/lib.rs","offset":560}"#.into(),
            )]),
            "a full read should block extra pages"
        );
        let messages = vec![
            Message::assistant(vec![Content::ToolCall {
                id: "r1".into(),
                name: "read".into(),
                arguments: r#"{"path":"crates/hi-tui/src/lib.rs"}"#.into(),
            }]),
            Message::tool_result("r1", "[elided read output — was 1289 lines]"),
        ];
        evidence.reopen_elided_reads(&messages);
        assert!(
            !evidence.rereads_only_completed_files(&[(
                "c".into(),
                "read".into(),
                r#"{"path":"crates/hi-tui/src/lib.rs","offset":560}"#.into(),
            )]),
            "elided contents are no longer 'returned in full'"
        );
        assert!(
            evidence.round_adds_evidence(&[(
                "c".into(),
                "read".into(),
                r#"{"path":"crates/hi-tui/src/lib.rs","offset":560}"#.into(),
            )]),
            "an extra page of an elided file is new evidence"
        );
    }

    #[test]
    fn truncated_file_paging_crosses_the_legacy_eight_page_boundary() {
        let mut evidence = EvidenceTracker::default();
        for page in 0..9 {
            let offset = page * 100 + 1;
            evidence.record_success(
                "read",
                &serde_json::json!({"path": "src/large.rs", "offset": offset}).to_string(),
                &format!(
                    "{offset}\tcontent\n— read more with offset {}",
                    offset + 100
                ),
            );
        }

        assert!(evidence.round_adds_evidence(&[(
            "next".into(),
            "read".into(),
            serde_json::json!({"path": "src/large.rs", "offset": 901}).to_string(),
        )]));
    }
}
