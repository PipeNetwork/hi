//! Parse `/goal` planner output.
//!
//! Grok-build's plan writer emits Markdown (`## Goal kind`, `## Acceptance
//! criteria`, `## Verification plan`, `## Task checklist`). hi's planner is a
//! chat-only one-shot, so that Markdown *is* the reply. Older mocks and
//! uncooperative models still emit one imperative line per milestone — keep
//! that as the fallback so [`parse_sub_goals`](super::parse_sub_goals) callers
//! (and the completion auditor's contract) stay valid.

use crate::GoalKind;
use crate::GoalPlan;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Section {
    /// Preamble / `# Plan:` title / unknown skip-headings: keep lines only if
    /// no task checklist shows up later.
    Loose,
    Kind,
    Acceptance,
    Verification,
    Checklist,
    /// Non-goals, assumed scope, approach, risks — never become milestones.
    Skip,
}

/// Structured planner reply, or a line-per-milestone list when there are no
/// named sections. `KIND:` lines set [`GoalPlan::kind`] in either form.
pub(crate) fn parse_planner_output(text: &str) -> GoalPlan {
    if looks_structured(text) {
        parse_structured(text)
    } else {
        parse_unstructured(text)
    }
}

fn looks_structured(text: &str) -> bool {
    text.lines().any(|line| heading_of(line).is_some())
}

fn parse_unstructured(text: &str) -> GoalPlan {
    let mut kind = None;
    let mut milestones = Vec::new();
    for line in super::parse_sub_goals(text) {
        if let Some(parsed) = kind_from_prefix(&line) {
            kind = Some(parsed);
            continue;
        }
        milestones.push(line);
    }
    GoalPlan {
        kind,
        milestones,
        ..GoalPlan::default()
    }
}

fn parse_structured(text: &str) -> GoalPlan {
    let mut section = Section::Loose;
    let mut kind = None;
    let mut acceptance = Vec::new();
    let mut verification = Vec::new();
    let mut checklist = Vec::new();
    let mut loose = Vec::new();

    for raw in text.lines() {
        if let Some(parsed) = kind_from_prefix(raw) {
            kind = Some(parsed);
            continue;
        }
        if let Some(next) = heading_of(raw) {
            section = next;
            continue;
        }
        let item = strip_item(raw);
        if item.is_empty() || is_terminal_done(&item) {
            continue;
        }
        match section {
            Section::Kind => {
                if kind.is_none() {
                    kind = GoalKind::parse_label(&item);
                }
            }
            Section::Acceptance => acceptance.push(item),
            Section::Verification => verification.push(strip_role_tag(&item)),
            Section::Checklist => checklist.push(item),
            Section::Loose => loose.push(item),
            Section::Skip => {}
        }
    }

    let mut milestones = if !checklist.is_empty() {
        checklist
    } else if !loose.is_empty() {
        loose
    } else {
        // Analysis/research plans often omit a checklist; drive the criteria.
        acceptance.clone()
    };

    if milestones.is_empty() {
        milestones = super::parse_sub_goals(text)
            .into_iter()
            .filter(|line| kind_from_prefix(line).is_none() && heading_of(line).is_none())
            .filter(|line| !is_terminal_done(line))
            .collect();
    }

    GoalPlan {
        kind,
        milestones,
        acceptance,
        verification,
    }
}

fn heading_of(line: &str) -> Option<Section> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    let markdown = trimmed.starts_with('#');
    let title = if markdown {
        trimmed.trim_start_matches('#').trim()
    } else {
        trimmed
    };
    let (title, rest) = match title.split_once(':') {
        Some((head, rest)) => (head.trim(), rest.trim()),
        None => (title, ""),
    };
    // Bare "KIND: code-change" is a kind assignment, not a section heading.
    // Markdown headings (`## Goal kind`) may still carry a trailing label.
    if !markdown && !rest.is_empty() {
        return None;
    }
    let key = title.to_ascii_lowercase();
    let section = match key.as_str() {
        "goal kind" | "kind" => Section::Kind,
        "acceptance criteria" | "acceptance" => Section::Acceptance,
        "verification plan" | "verification" => Section::Verification,
        "task checklist" | "checklist" | "milestones" => Section::Checklist,
        "non-goals"
        | "non goals"
        | "assumed scope"
        | "implementation approach"
        | "risks"
        | "risks / contradictions"
        | "risks/contradictions" => Section::Skip,
        "plan" if markdown => Section::Loose,
        _ if markdown => Section::Skip,
        _ => return None,
    };
    Some(section)
}

fn kind_from_prefix(line: &str) -> Option<GoalKind> {
    let stripped = super::strip_list_marker(line);
    let lower = stripped.to_ascii_lowercase();
    let rest = lower.strip_prefix("kind:")?;
    GoalKind::parse_label(rest)
}

fn strip_item(line: &str) -> String {
    let stripped = super::strip_list_marker(line);
    let trimmed = stripped.trim_start();
    for prefix in ["[ ]", "[x]", "[X]", "[>]", "[!]", "[-]"] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            return rest.trim().to_string();
        }
    }
    stripped.trim().to_string()
}

fn strip_role_tag(item: &str) -> String {
    let lower = item.to_ascii_lowercase();
    for tag in ["gating:", "evidence:"] {
        if let Some(rest) = lower.strip_prefix(tag) {
            let start = item.len().saturating_sub(rest.len());
            return item[start..].trim().to_string();
        }
    }
    item.to_string()
}

fn is_terminal_done(item: &str) -> bool {
    item.eq_ignore_ascii_case("done")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structured_markdown_fills_kind_criteria_and_checklist() {
        let raw = "# Plan: port the parser\n\
                   \n\
                   ## Goal kind\n\
                   code-change\n\
                   \n\
                   ## Acceptance criteria\n\
                   1. the lexer emits tokens for representative input\n\
                   2. the parser builds an AST from those tokens\n\
                   \n\
                   ## Verification plan\n\
                   1. gating: exercise the shipped lexer on a sample file\n\
                   2. evidence: capture the parse of a fixture\n\
                   \n\
                   ## Non-goals\n\
                   - a pretty-printer\n\
                   \n\
                   ## Task checklist\n\
                   - [ ] write the lexer\n\
                   - [ ] write the parser\n\
                   \n\
                   Done\n";
        let plan = parse_planner_output(raw);
        assert_eq!(plan.kind, Some(GoalKind::CodeChange));
        assert_eq!(
            plan.acceptance,
            vec![
                "the lexer emits tokens for representative input",
                "the parser builds an AST from those tokens",
            ]
        );
        assert_eq!(
            plan.verification,
            vec![
                "exercise the shipped lexer on a sample file",
                "capture the parse of a fixture",
            ]
        );
        assert_eq!(plan.milestones, vec!["write the lexer", "write the parser"]);
    }

    #[test]
    fn kind_prefix_plus_line_list_still_works() {
        let plan = parse_planner_output(
            "KIND: analysis\n\
             Explain how the auth middleware works\n\
             Name the request path it wraps\n",
        );
        assert_eq!(plan.kind, Some(GoalKind::Analysis));
        assert_eq!(
            plan.milestones,
            vec![
                "Explain how the auth middleware works",
                "Name the request path it wraps",
            ]
        );
        assert!(plan.acceptance.is_empty());
    }

    #[test]
    fn line_list_fallback_keeps_existing_mocks() {
        let raw = "Implement all missing frontend UI components and pages\n\
                   Set up authentication and API endpoints\n\
                   Add client-side state management\n";
        let plan = parse_planner_output(raw);
        assert_eq!(plan.kind, None);
        assert_eq!(
            plan.milestones,
            vec![
                "Implement all missing frontend UI components and pages",
                "Set up authentication and API endpoints",
                "Add client-side state management",
            ]
        );
    }

    #[test]
    fn analysis_without_checklist_drives_acceptance() {
        let raw = "## Goal kind\n\
                   analysis\n\
                   \n\
                   ## Acceptance criteria\n\
                   1. the auth middleware's request path is named\n\
                   2. failure modes are listed\n\
                   \n\
                   ## Verification plan\n\
                   1. the write-up cites the responsible functions\n";
        let plan = parse_planner_output(raw);
        assert_eq!(plan.kind, Some(GoalKind::Analysis));
        assert_eq!(
            plan.milestones,
            vec![
                "the auth middleware's request path is named",
                "failure modes are listed",
            ]
        );
        assert_eq!(plan.acceptance, plan.milestones);
        assert_eq!(
            plan.verification,
            vec!["the write-up cites the responsible functions"]
        );
    }

    #[test]
    fn numbered_and_bullet_checklists_are_cleaned() {
        let raw = "## Task checklist\n\
                   1. Add the parser module\n\
                   2) Wire it into main\n\
                   - Add a test\n\
                   * Update docs\n";
        let plan = parse_planner_output(raw);
        assert_eq!(
            plan.milestones,
            vec![
                "Add the parser module",
                "Wire it into main",
                "Add a test",
                "Update docs",
            ]
        );
    }

    #[test]
    fn backticks_around_kind_still_parse() {
        let plan =
            parse_planner_output("## Goal kind\n`research`\n## Task checklist\ncite the RFC\n");
        assert_eq!(plan.kind, Some(GoalKind::Research));
        assert_eq!(plan.milestones, vec!["cite the RFC"]);
    }
}
