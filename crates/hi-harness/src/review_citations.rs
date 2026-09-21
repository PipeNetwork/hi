//! Finding identity and citation checks for `/review` verdicts: which paths
//! a `path:line` cell names, whether they exist, whether a finding restates
//! a plan row, and the order-independent fingerprint that tells "same
//! findings after a fix pass" from progress.
//!
//! Split from [`crate::review`] (inputs, checklist, verdict parsing) so both
//! stay under the file-size ratchet; `review` re-exports every public item.

use std::path::Path;

use crate::review::{Finding, ReviewVerdict};

/// Same severity, title (normalized), and cited file: the merge key for
/// duplicate `finding:` rows.
pub(crate) fn same_finding(a: &Finding, b: &Finding) -> bool {
    a.severity == b.severity
        && normalize_title(&a.title) == normalize_title(&b.title)
        && citation_path(a.location.as_deref()) == citation_path(b.location.as_deref())
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
