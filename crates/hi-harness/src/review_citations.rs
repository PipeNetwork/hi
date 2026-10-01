//! Finding identity and citation checks for `/review` verdicts: which paths
//! a `path:line` cell names, whether they exist, whether a finding restates
//! a plan row, and the order-independent fingerprint that tells "same
//! findings after a fix pass" from progress.
//!
//! Split from [`crate::review`] (inputs, checklist, verdict parsing) so both
//! stay under the file-size ratchet; `review` re-exports every public item.

use std::path::Path;

use crate::review::{Finding, ReviewVerdict};

/// Cited lines this far apart still mean the same place: a fix pass moves
/// code by about the size of its edit.
const LINE_SLACK: u32 = 10;

/// The same `finding:` row reported twice (in one block, or by two turns of
/// one audit): same severity, normalized title, and cited file. The merge
/// key; it must stay stricter than [`same_finding`], which would fold a P0
/// into a P2 a few lines away that shares one word.
pub(crate) fn duplicate_finding(a: &Finding, b: &Finding) -> bool {
    a.severity == b.severity
        && normalize_title(&a.title) == normalize_title(&b.title)
        && citation_path(a.location.as_deref()) == citation_path(b.location.as_deref())
}

/// One defect, reported twice: the same cited file and either a similar
/// title (half the content words shared) or a citation within
/// [`LINE_SLACK`] lines with a word in common. Severity is not part of the
/// identity (a P1 re-filed as P0 is the same defect), and a reworded title
/// at the same place no longer counts as progress. With no cited file,
/// titles decide.
fn same_finding(a: &Finding, b: &Finding) -> bool {
    let (path_a, path_b) = (
        citation_path(a.location.as_deref()),
        citation_path(b.location.as_deref()),
    );
    if path_a != path_b {
        return false;
    }
    let shared = shared_tokens(&a.title, &b.title);
    if similar_titles(&a.title, &b.title, shared) {
        return true;
    }
    path_a.is_some()
        && shared >= 1
        && matches!(
            (
                citation_line(a.location.as_deref()),
                citation_line(b.location.as_deref()),
            ),
            (Some(la), Some(lb)) if la.abs_diff(lb) <= LINE_SLACK
        )
}

/// Content words the two titles share.
fn shared_tokens(a: &str, b: &str) -> usize {
    let tokens_a = match_tokens(a);
    let tokens_b = match_tokens(b);
    tokens_a
        .iter()
        .filter(|token| tokens_b.contains(token))
        .count()
}

/// Half of the shorter title's content words shared (identical titles
/// trivially; titles with no content words only when normalized equal).
fn similar_titles(a: &str, b: &str, shared: usize) -> bool {
    if normalize_title(a) == normalize_title(b) {
        return true;
    }
    let shorter = match_tokens(a).len().min(match_tokens(b).len());
    shorter > 0 && shared * 2 >= shorter && shared >= 1
}

/// True when the two blocking sets are the same defects (each matched by
/// [`same_finding`] in the other), so a re-audit that only reworded them
/// is no progress.
pub fn same_finding_set(prior: &[Finding], now: &[Finding]) -> bool {
    prior.len() == now.len()
        && now
            .iter()
            .all(|finding| prior.iter().any(|seen| same_finding(seen, finding)))
        && prior
            .iter()
            .all(|finding| now.iter().any(|seen| same_finding(seen, finding)))
}

/// Files a fix pass changed that none of its findings cite, minus tests,
/// lockfiles, and the like (a fix is asked to add a regression test). The
/// re-audit is told to look at them; the report warns.
pub fn uncited_changes(changed: &[String], findings: &[Finding]) -> Vec<String> {
    let cited: Vec<String> = findings
        .iter()
        .flat_map(|finding| citation_paths(finding.location.as_deref()))
        .collect();
    changed
        .iter()
        .filter(|file| {
            let file = file.trim_start_matches("./");
            !cited.iter().any(|path| path == file) && !is_expected_side_change(file)
        })
        .cloned()
        .collect()
}

/// Test files, lockfiles, snapshots: files a fix legitimately touches
/// without a finding citing them.
fn is_expected_side_change(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    let name = lower.rsplit('/').next().unwrap_or(&lower);
    lower.split('/').any(|part| {
        matches!(
            part,
            "tests"
                | "test"
                | "__tests__"
                | "spec"
                | "specs"
                | "snapshots"
                | "__snapshots__"
                | "fixtures"
        )
    }) || name.ends_with("_test.rs")
        || name.ends_with("_tests.rs")
        || name.starts_with("test_")
        || name.contains(".test.")
        || name.contains(".spec.")
        || name.contains("_test.")
        || matches!(
            name,
            "cargo.lock"
                | "package-lock.json"
                | "yarn.lock"
                | "pnpm-lock.yaml"
                | "poetry.lock"
                | "go.sum"
        )
}

/// The line of the first citation in a cell (`src/a.rs:42-50` -> 42).
pub fn citation_line(location: Option<&str>) -> Option<u32> {
    let raw = location?.split([',', ';']).next()?.trim().trim_matches('`');
    let raw = match raw.find('(') {
        Some(open) if open > 0 => raw[..open].trim(),
        _ => raw,
    };
    // Trailing numeric segments are `line`, `line-end`, or `line:col`; the
    // one right after the path is the line.
    let mut segments: Vec<&str> = raw.split(':').collect();
    let mut numeric: Vec<&str> = Vec::new();
    while segments.len() > 1 {
        let last = segments[segments.len() - 1];
        let digits_first = last.chars().next().is_some_and(|ch| ch.is_ascii_digit());
        if digits_first && last.chars().all(|ch| ch.is_ascii_digit() || ch == '-') {
            numeric.push(last);
            segments.pop();
        } else {
            break;
        }
    }
    let line = numeric.last()?;
    line.chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .ok()
}

/// Words that carry no meaning for matching a finding to a plan row.
const MATCH_STOPWORDS: &[&str] = &[
    "the",
    "and",
    "with",
    "for",
    "from",
    "that",
    "this",
    "into",
    "onto",
    "when",
    "then",
    "than",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "not",
    "its",
    "via",
    "per",
    "all",
    "any",
    "each",
    "every",
    "after",
    "before",
    "also",
    "still",
    "should",
    "must",
    "does",
    "did",
    "add",
    "implement",
    "build",
    "create",
    "support",
    "missing",
    "feature",
];

/// Lowercased content words with a crude stem (`removes`/`removed` ->
/// `remov`), so "KICK removes the target" and "Remove kicked target" share
/// tokens. Words under three letters and [`MATCH_STOPWORDS`] are dropped.
fn match_tokens(text: &str) -> Vec<String> {
    let mut tokens: Vec<String> = text
        .to_ascii_lowercase()
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|word| word.len() >= 3 && !MATCH_STOPWORDS.contains(word))
        .map(stem)
        .collect();
    tokens.sort();
    tokens.dedup();
    tokens
}

fn stem(word: &str) -> String {
    let mut word = word;
    for suffix in ["ing", "ed", "es", "s"] {
        if let Some(base) = word.strip_suffix(suffix)
            && base.len() >= 4
        {
            word = base;
            break;
        }
    }
    word.to_string()
}

/// True when `title` covers at least half of `item`'s content words (and
/// at least two of them): the finding is the plan row restated as work.
pub fn restates_item(item: &str, title: &str) -> bool {
    let item_tokens = match_tokens(item);
    if item_tokens.len() < 2 {
        return false;
    }
    let title_tokens = match_tokens(title);
    let shared = item_tokens
        .iter()
        .filter(|token| title_tokens.contains(token))
        .count();
    shared >= 2 && shared * 2 >= item_tokens.len()
}

/// Body of the last `<tag>…</tag>` in `text`.
pub(crate) fn last_tag_block<'a>(text: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = text.rfind(&open)? + open.len();
    let end = text[start..].find(&close)? + start;
    Some(&text[start..end])
}

fn normalize_title(title: &str) -> String {
    title
        .to_ascii_lowercase()
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Every path cited in one cell. A cell may hold several citations
/// (`a.rs:10, b.rs:20-31; c.rs`) and a parenthetical (`lib.rs (CONST)`), as
/// a live model wrote; each becomes one path. Bare line lists left over
/// from splitting `x.rs:10,12` are dropped.
pub fn citation_paths(location: Option<&str>) -> Vec<String> {
    let Some(raw) = location else {
        return Vec::new();
    };
    let mut paths: Vec<String> = raw
        .split([',', ';'])
        .filter_map(|segment| {
            let segment = segment.trim().trim_matches('`').trim();
            let segment = match segment.find('(') {
                Some(0) => segment.trim_matches(['(', ')']).trim(),
                Some(open) => segment[..open].trim(),
                None => segment,
            };
            if segment
                .chars()
                .all(|ch| ch.is_ascii_digit() || matches!(ch, '-' | ':' | ' '))
            {
                return None;
            }
            single_citation_path(segment)
        })
        .collect();
    paths.dedup();
    paths
}

/// The first path in a cell; findings are keyed by one file. See
/// [`citation_paths`].
pub fn citation_path(location: Option<&str>) -> Option<String> {
    citation_paths(location).into_iter().next()
}

/// `src/main.rs:42` → `src/main.rs`; strips backticks and a trailing `:line[:col]`.
fn single_citation_path(raw: &str) -> Option<String> {
    let raw = raw.trim().trim_matches('`').trim();
    if raw.is_empty() || raw == "-" {
        return None;
    }
    let mut path = raw;
    while let Some((head, tail)) = path.rsplit_once(':') {
        if !tail.is_empty()
            && tail
                .chars()
                .all(|ch| ch.is_ascii_digit() || ch == '-' || ch == ',')
        {
            path = head;
        } else {
            break;
        }
    }
    let path = path.trim_start_matches("./");
    (!path.is_empty()).then(|| path.to_string())
}

/// Mark findings that cite a file missing under `root` as unverified.
/// Returns the distinct cited paths (findings and coverage evidence) that
/// do not exist.
pub fn check_citations(root: &Path, verdict: &mut ReviewVerdict) -> Vec<String> {
    let mut unverified: Vec<String> = Vec::new();
    let mut note = |paths: Vec<String>| {
        for path in paths {
            if !unverified.contains(&path) {
                unverified.push(path);
            }
        }
    };
    for finding in &mut verdict.findings {
        let missing = missing_paths(root, finding.location.as_deref());
        if !missing.is_empty() {
            finding.verified = false;
            note(missing);
        }
    }
    for row in &verdict.coverage {
        note(missing_paths(root, row.evidence.as_deref()));
    }
    unverified
}

fn missing_paths(root: &Path, cell: Option<&str>) -> Vec<String> {
    citation_paths(cell)
        .into_iter()
        .filter(|path| !root.join(path).exists())
        .collect()
}

/// Order-independent identity of a finding set: severity, normalized title,
/// and cited path (line numbers move after a fix, so they are excluded).
pub fn fingerprint(findings: &[Finding]) -> String {
    let mut keys: Vec<String> = findings
        .iter()
        .map(|finding| {
            format!(
                "{}|{}|{}",
                finding.severity.as_str(),
                normalize_title(&finding.title),
                citation_path(finding.location.as_deref()).unwrap_or_default()
            )
        })
        .collect();
    keys.sort();
    keys.dedup();
    let digest = blake3::hash(keys.join("\n").as_bytes());
    digest.to_hex()[..16].to_string()
}
