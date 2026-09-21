//! `/review`: spec-coverage audit inputs, checklist rows, and the verdict
//! contract. The prompts live in [`crate::review_prompts`], citation checks
//! and the finding fingerprint in [`crate::review_citations`]; both are
//! re-exported at the bottom of this file.
//!
//! Pure helpers shared by the review drive ([`crate::review_drive`]) and the
//! frontends. Nothing here talks to the model or the session file.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Prefix on every prompt the review drive injects. `is_harness_injection`
/// and the intent back-scan in `completion.rs` skip lines that start with
/// `[hi:`, so these prompts never become the session title or a later
/// turn's inherited intent.
pub const REVIEW_PREFIX: &str = "[hi:review]";

/// Default number of fix passes before the loop stops with open P0/P1s.
pub const DEFAULT_REVIEW_PASSES: u32 = 3;

/// Directories searched (in order) for `plan.md` / `spec.md`.
const DISCOVERY_DIRS: &[&str] = &["", "docs", ".hi"];

/// What the user asked `/review` to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReviewAction {
    /// Audit, then fix P0/P1 findings and re-audit until clean or capped.
    Run,
    /// Audit and report only.
    Audit,
    Status,
    Stop,
}

/// Parsed `/review [audit|status|stop] [all] [path...]` arguments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReviewArgs {
    pub action: ReviewAction,
    /// Explicit plan/spec files or scope directories, in the order given.
    pub paths: Vec<String>,
    /// `all`: audit the whole workspace in chunks, one turn each.
    pub all: bool,
}

impl ReviewArgs {
    pub fn parse(arg: &str) -> Self {
        let mut tokens = arg.split_whitespace().peekable();
        let action = match tokens.peek().map(|token| token.to_ascii_lowercase()) {
            Some(word) if word == "audit" || word == "report" => {
                tokens.next();
                ReviewAction::Audit
            }
            Some(word) if word == "status" => {
                tokens.next();
                ReviewAction::Status
            }
            Some(word) if word == "stop" || word == "cancel" || word == "off" => {
                tokens.next();
                ReviewAction::Stop
            }
            Some(word) if word == "run" || word == "resume" || word == "fix" => {
                tokens.next();
                ReviewAction::Run
            }
            _ => ReviewAction::Run,
        };
        let mut paths: Vec<String> = tokens.map(str::to_string).collect();
        let all = take_all(&mut paths);
        Self { action, paths, all }
    }
}

/// Remove the `all` keyword from `paths` (any case, any position); true when
/// it was there. A directory literally named `all` is reachable as `./all`.
pub fn take_all(paths: &mut Vec<String>) -> bool {
    let before = paths.len();
    paths.retain(|path| !path.eq_ignore_ascii_case("all"));
    paths.len() != before
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputKind {
    Plan,
    Spec,
    Readme,
    Other,
}

impl InputKind {
    fn classify(path: &str) -> Self {
        let name = Path::new(path)
            .file_name()
            .map(|name| name.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();
        if name.contains("plan") {
            Self::Plan
        } else if name.contains("spec") {
            Self::Spec
        } else if name.starts_with("readme") {
            Self::Readme
        } else {
            Self::Other
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Plan => "plan",
            Self::Spec => "spec",
            Self::Readme => "readme",
            Self::Other => "doc",
        }
    }
}

/// One plan/spec document the audit reads.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReviewInput {
    /// Workspace-relative path with `/` separators.
    pub path: String,
    pub kind: InputKind,
}

/// Everything `discover_inputs` decided about what to audit against.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReviewInputs {
    pub files: Vec<ReviewInput>,
    /// Directories the audit should focus on (explicit non-document paths).
    #[serde(default)]
    pub scope: Vec<String>,
    /// Explicit paths that do not exist. The caller refuses to start.
    #[serde(default)]
    pub missing: Vec<String>,
    /// One line for the user: README fallback, defects-only, and so on.
    #[serde(default)]
    pub notice: Option<String>,
    /// Recent work per git, chosen when a large workspace has no plan/spec
    /// and no scope ([`crate::review_scope::resolve_target`]).
    #[serde(default)]
    pub git: Option<GitScope>,
    /// `all`: directories audited one turn each, in this order (`.` is the
    /// workspace's own top-level files).
    #[serde(default)]
    pub chunks: Vec<String>,
}

impl ReviewInputs {
    /// True when a plan, spec, or explicitly named document is among the
    /// inputs. A README fallback alone is not a spec.
    pub fn has_spec(&self) -> bool {
        self.files
            .iter()
            .any(|input| input.kind != InputKind::Readme)
    }

    /// No document at all: the audit reports defects and no coverage rows.
    pub fn defects_only(&self) -> bool {
        self.files.is_empty()
    }

    /// `plan.md + docs/spec.md` for status lines.
    pub fn summary(&self) -> String {
        if self.files.is_empty() {
            return "no plan/spec (defects only)".to_string();
        }
        self.files
            .iter()
            .map(|input| input.path.as_str())
            .collect::<Vec<_>>()
            .join(" + ")
    }

    /// What code the audit covers, for status lines: `47 uncommitted files
    /// (git status)`, `last commit a1b2c3d (3 files)`, `7 chunks`, `in
    /// src/server`; `None` for the whole workspace in one turn.
    pub fn scope_summary(&self) -> Option<String> {
        if let Some(git) = &self.git {
            return Some(git.summary());
        }
        if !self.chunks.is_empty() {
            return Some(format!("{} chunks", self.chunks.len()));
        }
        if !self.scope.is_empty() {
            return Some(format!("in {}", self.scope.join(", ")));
        }
        None
    }
}

fn normalize_rel(path: &str) -> String {
    let trimmed = path.trim().trim_start_matches("./");
    trimmed.replace('\\', "/").trim_end_matches('/').to_string()
}

fn display_path(root: &Path, path: &Path, raw: &str) -> String {
    path.strip_prefix(root)
        .map(|rel| rel.to_string_lossy().replace('\\', "/"))
        .unwrap_or_else(|_| normalize_rel(raw))
}

/// Case-insensitive lookup of `name` directly inside `dir`.
fn find_case_insensitive(dir: &Path, name: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut hits: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .eq_ignore_ascii_case(name)
        })
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .collect();
    hits.sort();
    hits.into_iter().next()
}

/// Explicit paths win. Otherwise `plan.md` / `spec.md` (any case) in the
/// workspace root, then `docs/`, then `.hi/`. With neither, fall back to
/// `README.md` with a notice; with nothing at all, audit for defects only.
pub fn discover_inputs(root: &Path, explicit: &[String]) -> ReviewInputs {
    let mut inputs = ReviewInputs::default();
    for raw in explicit {
        let candidate = Path::new(raw);
        let path = if candidate.is_absolute() {
            candidate.to_path_buf()
        } else {
            root.join(raw)
        };
        let shown = display_path(root, &path, raw);
        if path.is_dir() {
            inputs.scope.push(shown);
        } else if path.is_file() {
            inputs.files.push(ReviewInput {
                kind: InputKind::classify(&shown),
                path: shown,
            });
        } else {
            inputs.missing.push(raw.clone());
        }
    }
    if !inputs.missing.is_empty() {
        inputs.notice = Some(format!("no such file: {}", inputs.missing.join(", ")));
        return inputs;
    }
    if inputs.files.is_empty() {
        let mut plan = None;
        let mut spec = None;
        for dir in DISCOVERY_DIRS {
            let base = if dir.is_empty() {
                root.to_path_buf()
            } else {
                root.join(dir)
            };
            if plan.is_none() {
                plan = find_case_insensitive(&base, "plan.md");
            }
            if spec.is_none() {
                spec = find_case_insensitive(&base, "spec.md");
            }
        }
        for (found, kind) in [(plan, InputKind::Plan), (spec, InputKind::Spec)] {
            if let Some(path) = found {
                inputs.files.push(ReviewInput {
                    path: display_path(root, &path, &path.to_string_lossy()),
                    kind,
                });
            }
        }
    }
    if inputs.files.is_empty() {
        if let Some(readme) = find_case_insensitive(root, "readme.md") {
            let shown = display_path(root, &readme, "README.md");
            inputs.notice = Some(format!(
                "no plan.md or spec.md found; auditing against {shown}"
            ));
            inputs.files.push(ReviewInput {
                path: shown,
                kind: InputKind::Readme,
            });
        } else {
            inputs.notice =
                Some("no plan.md, spec.md, or README.md found; auditing for defects only".into());
        }
    }
    inputs
}

/// One `- [ ]` / `- [x]` / numbered row from a plan document.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChecklistItem {
    pub text: String,
    /// `Some(true)` for `[x]`, `Some(false)` for `[ ]`, `None` for numbered rows.
    pub checked: Option<bool>,
}

/// Extract checklist rows, skipping fenced code blocks.
pub fn parse_checklist(markdown: &str) -> Vec<ChecklistItem> {
    let mut items = Vec::new();
    let mut in_fence = false;
    for raw in markdown.lines() {
        let line = raw.trim_start();
        if line.starts_with("```") || line.starts_with("~~~") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        let Some(rest) = line
            .strip_prefix("- ")
            .or_else(|| line.strip_prefix("* "))
            .or_else(|| line.strip_prefix("+ "))
        else {
            if let Some(text) = numbered_row(line) {
                items.push(ChecklistItem {
                    text,
                    checked: None,
                });
            }
            continue;
        };
        let rest = rest.trim_start();
        let (checked, text) = if let Some(text) = rest.strip_prefix("[ ]") {
            (Some(false), text)
        } else if let Some(text) = rest
            .strip_prefix("[x]")
            .or_else(|| rest.strip_prefix("[X]"))
        {
            (Some(true), text)
        } else {
            continue;
        };
        let text = text.trim();
        if !text.is_empty() {
            items.push(ChecklistItem {
                text: text.to_string(),
                checked,
            });
        }
    }
    items
}

fn numbered_row(line: &str) -> Option<String> {
    let digits = line.chars().take_while(char::is_ascii_digit).count();
    if digits == 0 {
        return None;
    }
    let rest = &line[digits..];
    let rest = rest.strip_prefix('.').or_else(|| rest.strip_prefix(')'))?;
    let text = rest.strip_prefix(' ')?.trim();
    (!text.is_empty()).then(|| text.to_string())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Severity {
    P0,
    P1,
    P2,
    P3,
}

impl Severity {
    pub fn parse(text: &str) -> Option<Self> {
        match text
            .trim()
            .trim_matches(['[', ']', '`'])
            .to_ascii_uppercase()
            .as_str()
        {
            "P0" => Some(Self::P0),
            "P1" => Some(Self::P1),
            "P2" => Some(Self::P2),
            "P3" => Some(Self::P3),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::P0 => "P0",
            Self::P1 => "P1",
            Self::P2 => "P2",
            Self::P3 => "P3",
        }
    }

    /// P0/P1 enter the fix loop; P2/P3 are reported only.
    pub fn is_blocking(self) -> bool {
        matches!(self, Self::P0 | Self::P1)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoverageState {
    Implemented,
    Partial,
    Missing,
}

impl CoverageState {
    fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "implemented" | "done" | "complete" | "yes" => Some(Self::Implemented),
            "partial" | "partially" | "incomplete" => Some(Self::Partial),
            "missing" | "not implemented" | "no" | "absent" => Some(Self::Missing),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Implemented => "implemented",
            Self::Partial => "partial",
            Self::Missing => "missing",
        }
    }

    /// Higher is more built; a chunked audit keeps the best state seen.
    fn rank(self) -> u8 {
        match self {
            Self::Missing => 0,
            Self::Partial => 1,
            Self::Implemented => 2,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoverageRow {
    pub state: CoverageState,
    pub item: String,
    #[serde(default)]
    pub evidence: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Finding {
    pub severity: Severity,
    pub title: String,
    #[serde(default)]
    pub location: Option<String>,
    /// False once `check_citations` found no file behind `location`.
    #[serde(default = "default_true")]
    pub verified: bool,
    /// True when the title restates an unchecked plan row: a known gap the
    /// model filed as a defect. Reported, never handed to the fix loop
    /// (see [`ReviewVerdict::mark_feature_gaps`]).
    #[serde(default)]
    pub feature_gap: bool,
}

fn default_true() -> bool {
    true
}

impl Finding {
    pub fn label(&self) -> String {
        match &self.location {
            Some(location) => format!("[{}] {} — {location}", self.severity.as_str(), self.title),
            None => format!("[{}] {}", self.severity.as_str(), self.title),
        }
    }
}

/// The tagged block every audit reply must end with.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReviewVerdict {
    /// Taken from the coverage rows, not the `verdict:` word; see
    /// [`derive_complete`].
    pub complete: bool,
    pub coverage: Vec<CoverageRow>,
    pub findings: Vec<Finding>,
    #[serde(default)]
    pub residual: Option<String>,
    /// What the `verdict:` row said, when there was one.
    #[serde(default)]
    pub stated_complete: Option<bool>,
    /// `coverage:` rows dropped for a bad state or an empty item.
    #[serde(default)]
    pub unparsed_coverage_rows: usize,
}

impl ReviewVerdict {
    /// Parse the last `<review>…</review>` block. `None` when there is no
    /// block, or neither a `verdict:` row nor a `coverage:` row parsed;
    /// malformed rows are skipped.
    pub fn parse(text: &str) -> Option<Self> {
        let block = last_tag_block(text, "review")?;
        let mut stated = None;
        let mut coverage = Vec::new();
        let mut unparsed = 0usize;
        let mut findings: Vec<Finding> = Vec::new();
        let mut residual = None;
        for raw in block.lines() {
            let line = raw.trim().trim_start_matches(['-', '*', '•']).trim();
            let Some((key, value)) = line.split_once(':') else {
                continue;
            };
            let value = value.trim();
            if is_template_row(value) {
                continue;
            }
            match key.trim().to_ascii_lowercase().as_str() {
                "verdict" => {
                    let word = value
                        .split(|ch: char| !ch.is_ascii_alphabetic())
                        .find(|word| !word.is_empty())
                        .unwrap_or("")
                        .to_ascii_uppercase();
                    stated = match word.as_str() {
                        "COMPLETE" | "DONE" | "PASS" => Some(true),
                        "INCOMPLETE" | "PARTIAL" | "FAIL" => Some(false),
                        _ => stated,
                    };
                }
                "coverage" => {
                    // `coverage: none` is a defects-only audit saying so,
                    // like `finding: none`; not a row that failed to parse.
                    // (A `<state> | item | <…>` row still has cells.)
                    if !value.contains('|') && optional_cell(value).is_none() {
                        continue;
                    }
                    let mut parts = value.splitn(3, '|').map(str::trim);
                    let Some(state) = parts.next().and_then(CoverageState::parse) else {
                        unparsed += 1;
                        continue;
                    };
                    let Some(item) = parts.next().filter(|item| !item.is_empty()) else {
                        unparsed += 1;
                        continue;
                    };
                    coverage.push(CoverageRow {
                        state,
                        item: item.to_string(),
                        evidence: parts.next().and_then(optional_cell),
                    });
                }
                "finding" => {
                    let mut parts = value.splitn(3, '|').map(str::trim);
                    let Some(severity) = parts.next().and_then(Severity::parse) else {
                        continue;
                    };
                    let Some(title) = parts.next().filter(|title| !title.is_empty()) else {
                        continue;
                    };
                    let finding = Finding {
                        severity,
                        title: title.to_string(),
                        location: parts.next().and_then(optional_cell),
                        verified: true,
                        feature_gap: false,
                    };
                    if !findings.iter().any(|seen| same_finding(seen, &finding)) {
                        findings.push(finding);
                    }
                }
                "residual" => residual = optional_cell(value),
                _ => {}
            }
        }
        if stated.is_none() && coverage.is_empty() {
            return None;
        }
        Some(Self {
            complete: derive_complete(stated, &coverage, unparsed),
            coverage,
            findings,
            residual,
            stated_complete: stated,
            unparsed_coverage_rows: unparsed,
        })
    }

    /// One line when the block contradicted itself or lost rows, so the
    /// report says why the verdict differs from the model's word.
    pub fn verdict_note(&self) -> Option<String> {
        let mut notes = Vec::new();
        match self.stated_complete {
            Some(false) if self.complete => notes.push(
                "the model wrote INCOMPLETE with every coverage row implemented; verdict taken from the rows"
                    .to_string(),
            ),
            Some(true) if !self.complete => {
                let gaps = self
                    .coverage
                    .iter()
                    .filter(|row| row.state != CoverageState::Implemented)
                    .count();
                notes.push(format!(
                    "the model wrote COMPLETE with {gaps} missing/partial coverage row(s); verdict taken from the rows"
                ));
            }
            _ => {}
        }
        if self.unparsed_coverage_rows > 0 {
            notes.push(format!(
                "{} coverage row(s) could not be parsed and were dropped",
                self.unparsed_coverage_rows
            ));
        }
        (!notes.is_empty()).then(|| notes.join("; "))
    }

    /// Flag findings that restate an unchecked plan row. The plan already
    /// says those items are not built; the fix loop only takes defects in
    /// what is claimed done, so a "P1: implement /topic" is reported as a
    /// gap and never handed to a fix pass.
    pub fn mark_feature_gaps(&mut self, unchecked_items: &[String]) {
        for finding in &mut self.findings {
            finding.feature_gap = unchecked_items
                .iter()
                .any(|item| restates_item(item, &finding.title));
        }
    }

    /// P0/P1 defects for the fix loop; feature gaps are excluded.
    pub fn blocking_findings(&self) -> Vec<Finding> {
        self.findings
            .iter()
            .filter(|finding| finding.severity.is_blocking() && !finding.feature_gap)
            .cloned()
            .collect()
    }

    pub fn has_blocking(&self) -> bool {
        self.findings
            .iter()
            .any(|f| f.severity.is_blocking() && !f.feature_gap)
    }

    /// P0/P1 findings that are really unchecked plan rows (reported only).
    pub fn blocking_feature_gaps(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| f.severity.is_blocking() && f.feature_gap)
            .count()
    }

    /// No `partial` or `missing` rows and the model said COMPLETE.
    pub fn coverage_complete(&self) -> bool {
        self.complete
            && self
                .coverage
                .iter()
                .all(|row| row.state == CoverageState::Implemented)
    }

    /// Fold another chunk's verdict into this one: findings are appended
    /// (duplicates dropped), a coverage item keeps the best state any chunk
    /// saw (implemented over partial over missing, with that row's
    /// evidence), residuals are joined, and `complete` is re-derived.
    pub fn merge(&mut self, other: ReviewVerdict) {
        for row in other.coverage {
            match self
                .coverage
                .iter_mut()
                .find(|mine| mine.item.trim().eq_ignore_ascii_case(row.item.trim()))
            {
                Some(mine) if row.state.rank() > mine.state.rank() => *mine = row,
                Some(_) => {}
                None => self.coverage.push(row),
            }
        }
        for finding in other.findings {
            if !self
                .findings
                .iter()
                .any(|seen| same_finding(seen, &finding))
            {
                self.findings.push(finding);
            }
        }
        self.residual = match (self.residual.take(), other.residual) {
            (Some(mine), Some(theirs)) if mine != theirs => Some(format!("{mine}; {theirs}")),
            (mine, theirs) => mine.or(theirs),
        };
        self.stated_complete = match (self.stated_complete, other.stated_complete) {
            (Some(mine), Some(theirs)) => Some(mine && theirs),
            (mine, theirs) => mine.or(theirs),
        };
        self.unparsed_coverage_rows += other.unparsed_coverage_rows;
        self.complete = derive_complete(
            self.stated_complete,
            &self.coverage,
            self.unparsed_coverage_rows,
        );
    }
}

/// The rows are the contract. A `missing` or `partial` row makes the
/// verdict incomplete whatever the model wrote (a live run wrote COMPLETE
/// over a missing row), and rows that are all `implemented` make it
/// complete (another wrote INCOMPLETE over eight implemented rows and then
/// said so in its residual). When a row failed to parse the dropped row may
/// be the gap, so the model's word stands. With no rows at all the word is
/// all there is.
fn derive_complete(stated: Option<bool>, coverage: &[CoverageRow], unparsed: usize) -> bool {
    if coverage.is_empty() {
        return stated.unwrap_or(false);
    }
    if coverage
        .iter()
        .any(|row| row.state != CoverageState::Implemented)
    {
        return false;
    }
    if unparsed > 0 {
        return stated.unwrap_or(false);
    }
    true
}

/// `-`, `none`, and an unfilled `<placeholder>` from the skeleton are "no
/// value".
fn optional_cell(text: &str) -> Option<String> {
    let text = text.trim().trim_matches('`').trim();
    let placeholder = text.starts_with('<') && text.ends_with('>');
    if text.is_empty() || text == "-" || text.eq_ignore_ascii_case("none") || placeholder {
        return None;
    }
    Some(text.to_string())
}

/// A row copied verbatim from the format block (or the re-ask skeleton) is
/// not an answer. Without this, `verdict: COMPLETE | INCOMPLETE` would read
/// as COMPLETE and `finding: P0|P1|P2|P3 | …` as a P0 titled "P1". A row
/// whose only leftover placeholder is the citation cell is still an answer
/// (`optional_cell` drops the placeholder); an unfilled `<state>` fails
/// `CoverageState::parse` on its own.
fn is_template_row(value: &str) -> bool {
    if value.contains("P0|P1|P2|P3")
        || value.contains("implemented|partial|missing")
        || value.contains("<plan or spec item>")
        || value.contains("<imperative title>")
    {
        return true;
    }
    let words: Vec<String> = value
        .split(|ch: char| !ch.is_ascii_alphabetic())
        .filter(|word| !word.is_empty())
        .map(str::to_ascii_uppercase)
        .collect();
    words.iter().any(|w| w == "COMPLETE") && words.iter().any(|w| w == "INCOMPLETE")
}

pub use crate::review_citations::{check_citations, fingerprint, restates_item};
pub(crate) use crate::review_citations::{last_tag_block, same_finding};
pub use crate::review_prompts::{
    REVIEW_DEFECTS_FORMAT_BLOCK, REVIEW_FORMAT_BLOCK, REVIEW_FORMAT_HINT, audit_prompt, fix_prompt,
    format_reask_prompt, reaudit_prompt, transcript_label,
};
pub use crate::review_scope::{GitScope, GitSource, chunk_label};

#[cfg(test)]
#[path = "review_tests.rs"]
mod tests;
