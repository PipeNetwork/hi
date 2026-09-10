//! Grok-build `## Goal kind` tag, derived deterministically from the prompt.
//!
//! Grok-build's planner writes this once at goal creation (`code-change` |
//! `analysis` | `research`) and every later gate keys on it: analysis/research
//! judge the written deliverable; code-change treats prose as not evidence.
//! hi's `/goal` planner may emit the same tag; [`GoalKind::for_objective`] is
//! the fallback when the reply has no kind, and [`GoalKind::derive`] still
//! classifies ordinary (non-goal) turns from the mutation contract.

use serde::{Deserialize, Serialize};

/// Planner `## Goal kind` values from grok-build's `goal_planner_prompt.md`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GoalKind {
    /// Modify the workspace; the diff (and tests) are the evidence.
    CodeChange,
    /// Understand existing code; the deliverable is prose, and the diff may be empty.
    #[default]
    Analysis,
    /// Gather external information; the deliverable is a cited summary.
    Research,
}

impl GoalKind {
    /// `expected_mutation` is the harness fact that this turn owes a workspace
    /// change (explicit fix/implement, greenfield build, or a goal/plan drive).
    /// Everything else is analysis unless the prompt is clearly external research.
    pub fn derive(prompt: &str, expected_mutation: bool) -> Self {
        if expected_mutation {
            return Self::CodeChange;
        }
        if looks_like_research(prompt) {
            return Self::Research;
        }
        Self::Analysis
    }

    /// Code-change turns require tool-backed workspace evidence. Analysis and
    /// research are judged on the written deliverable, matching grok-build's
    /// kind lenses.
    pub fn requires_workspace_evidence(self) -> bool {
        matches!(self, Self::CodeChange)
    }

    /// Kind frozen at `/goal` creation when the planner omits `## Goal kind`.
    /// Coding objectives default to `code-change`; only clearly external-research
    /// or analysis prompts pick the other two.
    pub fn for_objective(objective: &str) -> Self {
        if looks_like_research(objective) {
            return Self::Research;
        }
        let lower = objective.to_ascii_lowercase();
        const ANALYSIS: &[&str] = &[
            "explain ",
            "analyze ",
            "analyse ",
            "what does ",
            "how does ",
            "summarize the code",
            "summarise the code",
        ];
        if ANALYSIS.iter().any(|cue| lower.contains(cue))
            && !lower.contains("fix")
            && !lower.contains("implement")
            && !lower.contains("add ")
            && !lower.contains("port ")
        {
            return Self::Analysis;
        }
        Self::CodeChange
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::CodeChange => "code-change",
            Self::Analysis => "analysis",
            Self::Research => "research",
        }
    }

    /// Parse a planner `## Goal kind` / `KIND:` value. Unknown labels are `None`
    /// so the caller can keep [`Self::for_objective`].
    pub fn parse_label(raw: &str) -> Option<Self> {
        let label = raw
            .trim()
            .trim_matches('`')
            .trim_matches('"')
            .trim_matches('\'')
            .to_ascii_lowercase();
        match label.as_str() {
            "code-change" | "code_change" | "codechange" | "code change" => Some(Self::CodeChange),
            "analysis" => Some(Self::Analysis),
            "research" => Some(Self::Research),
            _ => None,
        }
    }
}

fn looks_like_research(prompt: &str) -> bool {
    let lower = prompt.to_ascii_lowercase();
    const CUES: &[&str] = &[
        "search the web",
        "web search",
        "look up",
        "look this up",
        "cite sources",
        "with citations",
        "wikipedia",
        "according to rfc",
    ];
    if CUES.iter().any(|cue| lower.contains(cue)) {
        return true;
    }
    let trimmed = lower.trim_start();
    let research_lead = trimmed == "research"
        || trimmed
            .strip_prefix("research")
            .is_some_and(|rest| rest.starts_with(char::is_whitespace));
    if !research_lead {
        return false;
    }
    // "research the codebase" is analysis, not external fact-gathering.
    ![
        "codebase",
        "code base",
        "repo",
        "repository",
        "crate",
        "file",
    ]
    .iter()
    .any(|token| lower.contains(token))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn review_and_fix_is_code_change() {
        assert_eq!(
            GoalKind::derive("review for any major issues and fix", true),
            GoalKind::CodeChange
        );
        assert!(GoalKind::CodeChange.requires_workspace_evidence());
    }

    #[test]
    fn how_can_we_improve_is_analysis() {
        for prompt in [
            "how can we improve this",
            "how can we imrpove this",
            "review this for major issues",
            "explain how src/parser.rs works",
            "Run cargo test to verify the rate limiter change didn't break anything.",
        ] {
            assert_eq!(
                GoalKind::derive(prompt, false),
                GoalKind::Analysis,
                "{prompt}"
            );
        }
        assert!(!GoalKind::Analysis.requires_workspace_evidence());
    }

    #[test]
    fn web_research_is_research() {
        assert_eq!(
            GoalKind::derive("search the web for the RFC 1459 NICK rules", false),
            GoalKind::Research
        );
        assert_eq!(
            GoalKind::derive("research the history of IRC rate limiting", false),
            GoalKind::Research
        );
        assert_eq!(
            GoalKind::derive("research the codebase for auth leaks", false),
            GoalKind::Analysis
        );
        assert_eq!(
            GoalKind::derive("look this up with citations", false),
            GoalKind::Research
        );
        assert_eq!(
            GoalKind::derive("research the repository layout", false),
            GoalKind::Analysis
        );
    }

    #[test]
    fn expected_mutation_wins_over_research_cues() {
        assert_eq!(
            GoalKind::derive("search the web then fix the parser", true),
            GoalKind::CodeChange
        );
    }

    #[test]
    fn for_objective_defaults_to_code_change() {
        assert_eq!(
            GoalKind::for_objective("port the parser to Rust"),
            GoalKind::CodeChange
        );
        assert_eq!(
            GoalKind::for_objective("search the web for RFC 1459 NICK rules"),
            GoalKind::Research
        );
        assert_eq!(
            GoalKind::for_objective("explain how the auth middleware works"),
            GoalKind::Analysis
        );
        assert_eq!(
            GoalKind::for_objective("analyze the login bug and fix it"),
            GoalKind::CodeChange
        );
    }

    #[test]
    fn parse_label_accepts_planner_spellings() {
        assert_eq!(
            GoalKind::parse_label("code-change"),
            Some(GoalKind::CodeChange)
        );
        assert_eq!(
            GoalKind::parse_label("`analysis`"),
            Some(GoalKind::Analysis)
        );
        assert_eq!(GoalKind::parse_label("KIND: research"), None);
        assert_eq!(GoalKind::parse_label("research"), Some(GoalKind::Research));
        assert_eq!(
            GoalKind::parse_label("code change"),
            Some(GoalKind::CodeChange)
        );
        assert_eq!(GoalKind::parse_label("investigate"), None);
    }
}
