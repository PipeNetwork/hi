//! Typed activity rows for the session transcript.
//!
//! Grok-build's feed is a bullet list of verbs: collapsed Read / Edit / Run
//! rows, mixed exploration folded into one live header, and color diffs that
//! open only when asked. This module is that vocabulary for hi — not a port
//! of grok's pager.

use std::path::Path;
use std::time::Instant;

use ansi_to_tui::IntoText;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

use crate::Density;
use crate::render::dim;
use crate::theme::theme;

/// One foldable activity row in the transcript.
#[derive(Clone, Debug)]
pub(crate) struct ActivityBlock {
    pub kind: ActivityKind,
    pub expanded: bool,
}

/// The verb a row represents.
#[derive(Clone, Debug)]
pub(crate) enum ActivityKind {
    /// Consecutive non-destructive tools folded into one header.
    VerbGroup(VerbGroup),
    /// A file mutation. Collapsed shows `Edit path +N/-M`; expanded shows hunks.
    Edit {
        path: String,
        additions: u32,
        deletions: u32,
        /// Raw UI preview (ANSI or unified). Empty when the tool returned
        /// only a terse model-facing line.
        diff: String,
    },
    /// A shell / agent command. Collapsed is a one-liner; expanded shows stdout.
    Run {
        command: String,
        body: String,
        idle: bool,
        poll_count: u32,
    },
    /// Anything else (MCP, unknown). One-liner, optional body.
    Other {
        verb: String,
        detail: String,
        body: String,
    },
    /// A child explore/delegate/task. Enter/click inspects; not a tool dump.
    Subagent {
        id: String,
        kind: String,
        description: String,
        background: bool,
        activity: String,
        status: Option<String>,
        started_at: Instant,
        elapsed_ms: u64,
    },
}

/// Counts for a live or finished exploration fold.
#[derive(Clone, Debug, Default)]
pub(crate) struct VerbGroup {
    pub reads: u32,
    pub searches: u32,
    pub lists: u32,
    pub fetches: u32,
    /// Path/pattern for a singleton group.
    pub detail: Option<String>,
    pub lines: u32,
    pub all_empty: bool,
    /// Present tense while an explore tool in this group is in flight.
    pub live: bool,
    /// Further explore tools still fold into this group.
    pub open: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExploreVerb {
    Read,
    Search,
    List,
    Fetch,
}

impl ExploreVerb {
    pub(crate) fn from_tool(name: &str) -> Option<Self> {
        match name {
            "read" => Some(Self::Read),
            "grep" | "web_search" => Some(Self::Search),
            "list" => Some(Self::List),
            "web_fetch" => Some(Self::Fetch),
            _ => None,
        }
    }

    fn present(self) -> &'static str {
        match self {
            Self::Read => "Reading",
            Self::Search => "Searching",
            Self::List => "Listing",
            Self::Fetch => "Fetching",
        }
    }

    fn past(self) -> &'static str {
        match self {
            Self::Read => "Read",
            Self::Search => "Searched",
            Self::List => "Listed",
            Self::Fetch => "Fetched",
        }
    }

    fn noun(self, count: u32) -> &'static str {
        match (self, count) {
            (Self::Read, 1) => "file",
            (Self::Read, _) => "files",
            (Self::Search, 1) => "pattern",
            (Self::Search, _) => "patterns",
            (Self::List, 1) => "dir",
            (Self::List, _) => "dirs",
            (Self::Fetch, 1) => "website",
            (Self::Fetch, _) => "websites",
        }
    }
}

impl VerbGroup {
    pub(crate) fn add(&mut self, verb: ExploreVerb, detail: Option<String>) {
        match verb {
            ExploreVerb::Read => self.reads += 1,
            ExploreVerb::Search => self.searches += 1,
            ExploreVerb::List => self.lists += 1,
            ExploreVerb::Fetch => self.fetches += 1,
        }
        let total = self.total();
        if total == 1 {
            self.detail = detail;
        } else {
            self.detail = None;
        }
        self.live = true;
        self.open = true;
    }

    pub(crate) fn total(&self) -> u32 {
        self.reads + self.searches + self.lists + self.fetches
    }

    fn label(&self) -> String {
        let mut parts = Vec::new();
        let push = |parts: &mut Vec<String>, verb: ExploreVerb, count: u32| {
            if count == 0 {
                return;
            }
            let word = if self.live {
                verb.present()
            } else {
                verb.past()
            };
            if count == 1
                && self.total() == 1
                && let Some(detail) = &self.detail
            {
                parts.push(format!("{word} {detail}"));
                return;
            }
            parts.push(format!("{word} {count} {}", verb.noun(count)));
        };
        push(&mut parts, ExploreVerb::Read, self.reads);
        push(&mut parts, ExploreVerb::Search, self.searches);
        push(&mut parts, ExploreVerb::List, self.lists);
        push(&mut parts, ExploreVerb::Fetch, self.fetches);
        let mut label = if parts.is_empty() {
            if self.live {
                "Reading".to_string()
            } else {
                "Read".to_string()
            }
        } else {
            parts.join(", ")
        };
        if self.total() == 1 {
            if self.all_empty {
                label.push_str(" · (no output)");
            } else if self.lines > 0 {
                let s = if self.lines == 1 { "" } else { "s" };
                label.push_str(&format!(" · {} line{s}", self.lines));
            }
        }
        label
    }
}

impl ActivityKind {
    fn is_foldable(&self) -> bool {
        match self {
            Self::VerbGroup(_) => false,
            Self::Edit { diff, .. } => !diff.trim().is_empty(),
            Self::Run { body, idle, .. } => !*idle && !body.trim().is_empty(),
            Self::Other { body, .. } => !body.trim().is_empty(),
            Self::Subagent { .. } => false,
        }
    }
}

impl ActivityBlock {
    pub(crate) fn verb_group(verb: ExploreVerb, detail: Option<String>) -> Self {
        let mut group = VerbGroup::default();
        group.add(verb, detail);
        Self {
            kind: ActivityKind::VerbGroup(group),
            expanded: false,
        }
    }

    pub(crate) fn is_foldable(&self) -> bool {
        self.kind.is_foldable()
    }

    pub(crate) fn as_verb_group_mut(&mut self) -> Option<&mut VerbGroup> {
        match &mut self.kind {
            ActivityKind::VerbGroup(g) => Some(g),
            _ => None,
        }
    }

    pub(crate) fn as_run_mut(&mut self) -> Option<(&str, &mut bool, &mut u32)> {
        match &mut self.kind {
            ActivityKind::Run {
                command,
                idle,
                poll_count,
                ..
            } => Some((command.as_str(), idle, poll_count)),
            _ => None,
        }
    }

    pub(crate) fn subagent_id(&self) -> Option<&str> {
        match &self.kind {
            ActivityKind::Subagent { id, .. } => Some(id.as_str()),
            _ => None,
        }
    }

    pub(crate) fn flatten(&self, show_tool_output: bool, density: Density) -> Vec<Line<'static>> {
        let show = density.show_tool_output(show_tool_output) || self.expanded;
        let header = self.header_line();
        if !show || !self.is_foldable() {
            return vec![header];
        }
        let mut lines = vec![header];
        lines.extend(self.body_lines());
        lines
    }

    pub(crate) fn text(&self) -> String {
        match &self.kind {
            ActivityKind::VerbGroup(g) => g.label(),
            ActivityKind::Edit {
                path,
                additions,
                deletions,
                diff,
            } => {
                let mut s = edit_header_text(path, *additions, *deletions);
                if !diff.trim().is_empty() {
                    s.push('\n');
                    s.push_str(&strip_ansi(diff));
                }
                s
            }
            ActivityKind::Run {
                command,
                body,
                idle,
                poll_count,
            } => {
                let mut s = run_header_text(command, *idle, *poll_count, body);
                if !idle && !body.trim().is_empty() {
                    s.push('\n');
                    s.push_str(&strip_ansi(body));
                }
                s
            }
            ActivityKind::Other {
                verb, detail, body, ..
            } => {
                let mut s = other_header_text(verb, detail);
                if !body.trim().is_empty() {
                    s.push('\n');
                    s.push_str(&strip_ansi(body));
                }
                s
            }
            ActivityKind::Subagent { .. } => subagent_header_text(&self.kind),
        }
    }

    fn header_line(&self) -> Line<'static> {
        let th = theme();
        let bullet = Span::styled("• ", Style::default().fg(th.gray_dim));
        match &self.kind {
            ActivityKind::VerbGroup(g) => Line::from(vec![
                bullet,
                Span::styled(
                    g.label(),
                    Style::default()
                        .fg(th.text_primary)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            ActivityKind::Edit {
                path,
                additions,
                deletions,
                ..
            } => {
                let mut spans = vec![
                    bullet,
                    Span::styled(
                        "Edit ".to_string(),
                        Style::default()
                            .fg(th.text_primary)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(display_path(path).to_string(), Style::default().fg(th.path)),
                ];
                if *additions > 0 || *deletions > 0 {
                    if *additions > 0 {
                        spans.push(Span::styled(
                            format!(" +{additions}"),
                            Style::default().fg(th.diff_add),
                        ));
                    }
                    if *additions > 0 && *deletions > 0 {
                        spans.push(Span::styled("/", Style::default().fg(th.gray_dim)));
                    }
                    if *deletions > 0 {
                        spans.push(Span::styled(
                            format!(" -{deletions}"),
                            Style::default().fg(th.diff_del),
                        ));
                    }
                }
                Line::from(spans)
            }
            ActivityKind::Run {
                command,
                body,
                idle,
                poll_count,
            } => Line::from(vec![
                bullet,
                Span::styled(
                    run_header_text(command, *idle, *poll_count, body),
                    Style::default()
                        .fg(th.text_primary)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            ActivityKind::Other { verb, detail, .. } => {
                let mut spans = vec![
                    bullet,
                    Span::styled(
                        title_case(verb),
                        Style::default()
                            .fg(th.text_primary)
                            .add_modifier(Modifier::BOLD),
                    ),
                ];
                if !detail.is_empty() {
                    spans.push(Span::styled(
                        format!(" {detail}"),
                        Style::default().fg(th.text_secondary),
                    ));
                }
                Line::from(spans)
            }
            ActivityKind::Subagent { status, .. } => {
                let color = match status.as_deref() {
                    Some("completed") => th.accent_success,
                    Some("failed") | Some("denied") | Some("cancelled") => th.accent_error,
                    _ => th.accent_running,
                };
                Line::from(vec![
                    bullet,
                    Span::styled(
                        subagent_header_text(&self.kind),
                        Style::default().fg(color).add_modifier(Modifier::BOLD),
                    ),
                ])
            }
        }
    }

    fn body_lines(&self) -> Vec<Line<'static>> {
        match &self.kind {
            ActivityKind::VerbGroup(_) => Vec::new(),
            ActivityKind::Edit { diff, .. } => edit_body_lines(diff),
            ActivityKind::Run { body, idle, .. } => {
                if *idle {
                    Vec::new()
                } else {
                    output_body_lines(body)
                }
            }
            ActivityKind::Other { body, .. } => output_body_lines(body),
            ActivityKind::Subagent { .. } => Vec::new(),
        }
    }
}

fn edit_body_lines(diff: &str) -> Vec<Line<'static>> {
    let plain = strip_ansi(diff);
    if plain.trim().is_empty() {
        return Vec::new();
    }
    if crate::render::looks_like_diff(&plain) {
        return crate::render::diff_lines(&plain);
    }
    plain
        .lines()
        .filter_map(|line| {
            if line.contains(" addition") || line.contains(" deletion") {
                return None;
            }
            if line.contains('⋯') {
                return Some(crate::render::banded_diff_line(
                    crate::render::DiffBand::Meta,
                    "    ",
                    line.trim(),
                ));
            }
            if let Some((sign, gutter, content)) = hi_display_line_parts(line) {
                let band = match sign {
                    '+' => crate::render::DiffBand::Add,
                    '-' => crate::render::DiffBand::Del,
                    _ => crate::render::DiffBand::Context,
                };
                return Some(crate::render::banded_diff_line(band, gutter, content));
            }
            Some(crate::render::banded_diff_line(
                crate::render::DiffBand::Meta,
                "    ",
                line,
            ))
        })
        .collect()
}

fn output_body_lines(body: &str) -> Vec<Line<'static>> {
    if body.trim().is_empty() {
        return vec![Line::from(vec![
            Span::raw("  "),
            Span::styled("(no output)", dim()),
        ])];
    }
    let plain = strip_ansi(body);
    if crate::render::looks_like_diff(&plain) {
        return crate::render::diff_lines(&plain)
            .into_iter()
            .map(|mut line| {
                line.spans.insert(0, Span::raw("  "));
                line
            })
            .collect();
    }
    let th = theme();
    let text = body
        .into_text()
        .unwrap_or_else(|_| ratatui::text::Text::from(strip_ansi(body)));
    text.lines
        .into_iter()
        .map(|mut line| {
            line.spans.insert(0, Span::raw("  "));
            if th.paints_backgrounds() {
                line.style = line.style.bg(th.panel);
            }
            line
        })
        .collect()
}

fn edit_header_text(path: &str, additions: u32, deletions: u32) -> String {
    let name = display_path(path);
    match (additions, deletions) {
        (0, 0) => format!("Edit {name}"),
        (a, 0) => format!("Edit {name} +{a}"),
        (0, d) => format!("Edit {name} -{d}"),
        (a, d) => format!("Edit {name} +{a}/-{d}"),
    }
}

fn run_header_text(command: &str, idle: bool, poll_count: u32, body: &str) -> String {
    if idle {
        if poll_count <= 1 {
            format!("{command} · still running")
        } else {
            format!("{command} · still running · polled {poll_count}×")
        }
    } else if body.trim().is_empty() {
        format!("Run {command} · (no output)")
    } else {
        format!("Run {command}")
    }
}

fn other_header_text(verb: &str, detail: &str) -> String {
    if detail.is_empty() {
        title_case(verb)
    } else {
        format!("{} {detail}", title_case(verb))
    }
}

fn title_case(verb: &str) -> String {
    let mut chars = verb.chars();
    match chars.next() {
        Some(c) => format!("{}{}", c.to_uppercase(), chars.as_str()),
        None => verb.to_string(),
    }
}

fn display_path(path: &str) -> &str {
    Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or(path)
}

/// Salient argument from a `tool_label` (`read src/main.rs` → `src/main.rs`).
pub(crate) fn label_detail(label: &str) -> Option<String> {
    label.split_once(' ').map(|(_, rest)| rest.to_string())
}

pub(crate) fn is_edit_tool(name: &str) -> bool {
    matches!(name, "write" | "edit" | "multi_edit" | "apply_patch")
}

pub(crate) fn is_run_tool(name: &str) -> bool {
    matches!(name, "bash" | "bash_output" | "bash_kill")
}

pub(crate) fn is_parent_subagent_tool(name: &str) -> bool {
    matches!(name, "explore" | "delegate" | "task")
        || name.starts_with("explore:")
        || name.starts_with("delegate:")
        || name.starts_with("task:")
}

fn clip_desc(text: &str, max: usize) -> String {
    let count = text.chars().count();
    let clipped: String = text.chars().take(max).collect();
    if count > max {
        format!("{clipped}…")
    } else {
        clipped
    }
}

fn subagent_kind_label(kind: &str, background: bool) -> String {
    if background {
        return "Task".to_string();
    }
    match kind {
        "explore" => "Explore".into(),
        "delegate" => "Delegate".into(),
        "plan" => "Plan".into(),
        "general-purpose" => "Task".into(),
        other => title_case(other),
    }
}

fn subagent_header_text(kind: &ActivityKind) -> String {
    let ActivityKind::Subagent {
        kind,
        description,
        background,
        activity,
        status,
        started_at,
        elapsed_ms,
        ..
    } = kind
    else {
        return String::new();
    };
    let label = subagent_kind_label(kind, *background);
    let desc = clip_desc(description, 48);
    let elapsed = if status.is_some() {
        crate::util::fmt_elapsed(*elapsed_ms / 1000)
    } else {
        crate::util::fmt_elapsed(started_at.elapsed().as_secs())
    };
    if *background && status.is_none() {
        return format!("{label} \"{desc}\" started");
    }
    if let Some(status) = status {
        if *background {
            format!("{label} \"{desc}\" {status} in {elapsed}")
        } else {
            format!("{label} \"{desc}\" — {status} in {elapsed}")
        }
    } else if activity.is_empty() {
        format!("{label} \"{desc}\" · {elapsed}")
    } else {
        format!("{label} \"{desc}\" — {activity} · {elapsed}")
    }
}

/// Strip CSI/OSC ANSI sequences so we can parse colored tool previews.
pub(crate) fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            match chars.peek() {
                Some('[') => {
                    chars.next();
                    for n in chars.by_ref() {
                        if n.is_ascii_alphabetic() {
                            break;
                        }
                    }
                }
                Some(']') => {
                    chars.next();
                    for n in chars.by_ref() {
                        if n == '\u{7}' {
                            break;
                        }
                    }
                }
                _ => {}
            }
            continue;
        }
        out.push(c);
    }
    out
}

/// Count additions/deletions from a UI preview or unified diff.
pub(crate) fn parse_diff_stats(s: &str) -> (u32, u32) {
    let plain = strip_ansi(s);
    if let Some(adds) = number_before(&plain, " addition") {
        let dels = number_before(&plain, " deletion").unwrap_or(0);
        return (adds, dels);
    }
    let mut adds = 0u32;
    let mut dels = 0u32;
    for line in plain.lines() {
        match classify_diff_line(line) {
            DiffLineKind::Add => adds += 1,
            DiffLineKind::Del => dels += 1,
            _ => {}
        }
    }
    (adds, dels)
}

fn number_before(s: &str, marker: &str) -> Option<u32> {
    let idx = s.find(marker)?;
    let prefix = s[..idx].rsplit(|c: char| !c.is_ascii_digit()).next()?;
    prefix.parse().ok()
}

#[derive(Clone, Copy)]
pub(crate) enum DiffLineKind {
    Add,
    Del,
    Context,
    Meta,
}

pub(crate) fn classify_diff_line(line: &str) -> DiffLineKind {
    let t = line.trim_start();
    if t.starts_with("@@")
        || t.starts_with("diff ")
        || t.starts_with("+++")
        || t.starts_with("---")
        || t.contains('⋯')
        || t.starts_with("addition")
        || t.contains(" addition")
    {
        return DiffLineKind::Meta;
    }
    if let Some(sign) = hi_display_sign(line) {
        return match sign {
            '+' => DiffLineKind::Add,
            '-' => DiffLineKind::Del,
            _ => DiffLineKind::Context,
        };
    }
    if t.starts_with('+') {
        DiffLineKind::Add
    } else if t.starts_with('-') {
        DiffLineKind::Del
    } else {
        DiffLineKind::Context
    }
}

/// `hi_tools::edit::diff` paints `{:>4} {sign} {text}`.
fn hi_display_sign(line: &str) -> Option<char> {
    let bytes = line.as_bytes();
    if bytes.len() < 7 {
        return None;
    }
    if bytes[4] != b' ' || bytes[6] != b' ' {
        return None;
    }
    let gutter_ok = bytes[..4].iter().all(|&b| b.is_ascii_digit() || b == b' ');
    if !gutter_ok {
        return None;
    }
    match bytes[5] {
        b'+' | b'-' | b' ' => Some(bytes[5] as char),
        _ => None,
    }
}

pub(crate) fn hi_display_line_parts(line: &str) -> Option<(char, &str, &str)> {
    hi_display_sign(line)?;
    let sign = line.as_bytes()[5] as char;
    Some((sign, &line[..4], &line[7..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_stats_from_display_summary() {
        let s = "\x1b[1m12 additions, 3 deletions\x1b[0m\n  10 + foo\n";
        assert_eq!(parse_diff_stats(s), (12, 3));
    }

    #[test]
    fn parse_stats_from_unified_diff() {
        let s = "--- a/x\n+++ b/x\n@@ -1,1 +1,2 @@\n-old\n+new\n+also\n";
        assert_eq!(parse_diff_stats(s), (2, 1));
    }

    #[test]
    fn verb_group_mixed_label() {
        let mut g = VerbGroup::default();
        g.add(ExploreVerb::Read, Some("a.rs".into()));
        g.add(ExploreVerb::Search, Some("TODO".into()));
        g.live = false;
        assert_eq!(g.label(), "Read 1 file, Searched 1 pattern");
    }

    #[test]
    fn verb_group_singleton_keeps_path() {
        let mut g = VerbGroup::default();
        g.add(ExploreVerb::Read, Some("src/main.rs".into()));
        g.lines = 3;
        g.live = false;
        assert_eq!(g.label(), "Read src/main.rs · 3 lines");
    }

    #[test]
    fn subagent_header_live_and_finished() {
        let kind = ActivityKind::Subagent {
            id: "explore-1".into(),
            kind: "explore".into(),
            description: "crate boundaries".into(),
            background: false,
            activity: "Reading lib.rs".into(),
            status: None,
            started_at: Instant::now(),
            elapsed_ms: 0,
        };
        let live = subagent_header_text(&kind);
        assert!(
            live.contains("Explore") && live.contains("Reading lib.rs"),
            "{live}"
        );
        let done = ActivityKind::Subagent {
            id: "explore-1".into(),
            kind: "explore".into(),
            description: "crate boundaries".into(),
            background: false,
            activity: String::new(),
            status: Some("completed".into()),
            started_at: Instant::now(),
            elapsed_ms: 12_000,
        };
        let finished = subagent_header_text(&done);
        assert!(
            finished.contains("completed") && finished.contains("crate boundaries"),
            "{finished}"
        );
    }
}
