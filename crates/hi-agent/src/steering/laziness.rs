//! Grok-build Layer-3 laziness classifier: JSON parse, evaluate, nudge text.
//!
//! The idle side-model actor is out of scope. This module is the pure
//! decision surface plus a deterministic claim-vs-evidence helper the turn
//! loop can call without a second provider.

use super::goal_kind::GoalKind;
use super::settlement::{StallCategory, classify_text_answer, offers_next_work};

#[cfg(test)]
pub(crate) const LAZINESS_STALLED_NARRATION: &str = "stalled_narration";
#[cfg(test)]
pub(crate) const LAZINESS_STALLED_PERMISSION_ASKING: &str = "stalled_permission_asking";
#[cfg(test)]
pub(crate) const LAZINESS_STALLED_NO_TODOS_BUT_TASK_IN_FLIGHT: &str =
    "stalled_no_todos_but_task_in_flight";
#[cfg(test)]
pub(crate) const LAZINESS_STALLED_FALSE_COMPLETION: &str = "stalled_false_completion";
#[cfg(test)]
pub(crate) const LAZINESS_NOT_STALLED_COMPLETE: &str = "not_stalled_complete";
#[cfg(test)]
pub(crate) const LAZINESS_NOT_STALLED_WAITING_BG: &str = "not_stalled_waiting_on_background";
#[cfg(test)]
pub(crate) const LAZINESS_NOT_STALLED_WAITING_USER: &str = "not_stalled_waiting_on_user";

pub(crate) const LAZINESS_DEFAULT_MIN_CONFIDENCE: f32 = 0.7;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LazinessCategory {
    StalledNarration,
    StalledPermissionAsking,
    StalledNoTodosButTaskInFlight,
    StalledFalseCompletion,
    NotStalledComplete,
    NotStalledWaitingOnBackground,
    NotStalledWaitingOnUser,
}

impl LazinessCategory {
    #[cfg(test)]
    pub(crate) fn as_const_str(self) -> &'static str {
        match self {
            Self::StalledNarration => LAZINESS_STALLED_NARRATION,
            Self::StalledPermissionAsking => LAZINESS_STALLED_PERMISSION_ASKING,
            Self::StalledNoTodosButTaskInFlight => LAZINESS_STALLED_NO_TODOS_BUT_TASK_IN_FLIGHT,
            Self::StalledFalseCompletion => LAZINESS_STALLED_FALSE_COMPLETION,
            Self::NotStalledComplete => LAZINESS_NOT_STALLED_COMPLETE,
            Self::NotStalledWaitingOnBackground => LAZINESS_NOT_STALLED_WAITING_BG,
            Self::NotStalledWaitingOnUser => LAZINESS_NOT_STALLED_WAITING_USER,
        }
    }

    pub(crate) fn is_stalled(self) -> bool {
        matches!(
            self,
            Self::StalledNarration
                | Self::StalledPermissionAsking
                | Self::StalledNoTodosButTaskInFlight
                | Self::StalledFalseCompletion
        )
    }
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub(crate) struct ClassifierOutput {
    pub(crate) category: LazinessCategory,
    pub(crate) confidence: f32,
    pub(crate) evidence: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LazinessConfig {
    pub enabled: bool,
    pub min_confidence: Option<f32>,
    pub max_nudges_per_session: u32,
}

impl Default for LazinessConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_confidence: Some(LAZINESS_DEFAULT_MIN_CONFIDENCE),
            max_nudges_per_session: 3,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum LazinessDecision {
    Nudge {
        category: LazinessCategory,
        confidence: f32,
        evidence: String,
    },
    NoNudge {
        category: LazinessCategory,
        confidence: f32,
        reason: NoNudgeReason,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NoNudgeReason {
    NotStalled,
    LowConfidence,
    CapExhausted,
    FeatureDisabled,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ClassifierParseError {
    Unparseable,
    ConfidenceOutOfRange,
}

#[cfg(test)]
impl std::fmt::Display for ClassifierParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unparseable => f.write_str("classifier output not parseable as JSON"),
            Self::ConfidenceOutOfRange => f.write_str("confidence outside [0.0, 1.0]"),
        }
    }
}

#[cfg(test)]
fn strip_code_fence(raw: &str) -> Option<&str> {
    let trimmed = raw.trim();
    let body = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```JSON"))
        .or_else(|| trimmed.strip_prefix("```"))?;
    let body = body.trim_start_matches(['\n', '\r']);
    body.trim_end()
        .strip_suffix("```")
        .map(|s| s.trim_end_matches(['\n', '\r']))
}

#[cfg(test)]
fn extract_first_balanced_object(raw: &str) -> Option<&str> {
    let bytes = raw.as_bytes();
    let start = bytes.iter().position(|&b| b == b'{')?;
    let mut depth: i32 = 0;
    let mut in_str = false;
    let mut escape = false;
    for (offset, &b) in bytes[start..].iter().enumerate() {
        if in_str {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    let end = start + offset + 1;
                    return raw.get(start..end);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
pub(crate) fn parse_classifier_output(raw: &str) -> Result<ClassifierOutput, ClassifierParseError> {
    fn try_parse(slice: &str) -> Option<Result<ClassifierOutput, f32>> {
        let parsed: ClassifierOutput = serde_json::from_str(slice).ok()?;
        if (0.0..=1.0).contains(&parsed.confidence) {
            Some(Ok(parsed))
        } else {
            Some(Err(parsed.confidence))
        }
    }
    let mut out_of_range = false;
    let mut accept = |attempt: Option<Result<ClassifierOutput, f32>>| match attempt {
        Some(Ok(parsed)) => Some(parsed),
        Some(Err(_)) => {
            out_of_range = true;
            None
        }
        None => None,
    };
    for attempt in [
        try_parse(raw),
        strip_code_fence(raw).and_then(try_parse),
        extract_first_balanced_object(raw).and_then(try_parse),
    ] {
        if let Some(parsed) = accept(attempt) {
            return Ok(parsed);
        }
    }
    if out_of_range {
        return Err(ClassifierParseError::ConfidenceOutOfRange);
    }
    Err(ClassifierParseError::Unparseable)
}

pub(crate) fn evaluate_laziness(
    parsed: &ClassifierOutput,
    cfg: &LazinessConfig,
    nudges_used_this_session: u32,
    default_min_confidence: f32,
) -> LazinessDecision {
    let category = parsed.category;
    let confidence = parsed.confidence;
    if !cfg.enabled {
        return LazinessDecision::NoNudge {
            category,
            confidence,
            reason: NoNudgeReason::FeatureDisabled,
        };
    }
    if !category.is_stalled() {
        return LazinessDecision::NoNudge {
            category,
            confidence,
            reason: NoNudgeReason::NotStalled,
        };
    }
    let min_conf = cfg.min_confidence.unwrap_or(default_min_confidence);
    if confidence < min_conf {
        return LazinessDecision::NoNudge {
            category,
            confidence,
            reason: NoNudgeReason::LowConfidence,
        };
    }
    if nudges_used_this_session >= cfg.max_nudges_per_session {
        return LazinessDecision::NoNudge {
            category,
            confidence,
            reason: NoNudgeReason::CapExhausted,
        };
    }
    LazinessDecision::Nudge {
        category,
        confidence,
        evidence: parsed.evidence.clone(),
    }
}

pub(crate) fn build_laziness_nudge(category: LazinessCategory, evidence: &str) -> String {
    let rule = match category {
        LazinessCategory::StalledNarration => {
            "Don't narrate progress in prose without a corresponding tool call. Make the next concrete tool call this turn."
        }
        LazinessCategory::StalledPermissionAsking => {
            "Don't ask permission to continue a task that is in flight. Resume work — only pause for genuine ambiguity that changes the approach."
        }
        LazinessCategory::StalledNoTodosButTaskInFlight => {
            "A multi-step task is clearly in flight — make the next concrete tool call now."
        }
        LazinessCategory::StalledFalseCompletion => {
            "You declared completion but evidence is missing in the transcript. Either run the tool calls that back your claims, or correct the claim and continue the actual work."
        }
        LazinessCategory::NotStalledComplete
        | LazinessCategory::NotStalledWaitingOnBackground
        | LazinessCategory::NotStalledWaitingOnUser => return String::new(),
    };
    format!("Idle-stall detector flagged this session: {evidence}\n\n{rule}")
}

/// Deterministic claim-vs-evidence audit used when no classifier JSON is present.
pub(crate) fn claim_evidence_category(
    text: &str,
    kind: GoalKind,
    tests_seen: bool,
    plan_incomplete: bool,
    awaiting_background: bool,
) -> LazinessCategory {
    if awaiting_background && !text.trim().is_empty() {
        return LazinessCategory::NotStalledWaitingOnBackground;
    }
    if !kind.requires_workspace_evidence() && offers_next_work(text) {
        return LazinessCategory::NotStalledWaitingOnUser;
    }
    if kind.requires_workspace_evidence() && looks_like_false_completion(text) && !tests_seen {
        return LazinessCategory::StalledFalseCompletion;
    }
    match classify_text_answer(kind, text, plan_incomplete, awaiting_background) {
        StallCategory::StalledNarration => LazinessCategory::StalledNarration,
        StallCategory::StalledPermissionAsking => LazinessCategory::StalledPermissionAsking,
        StallCategory::StalledFalseCompletion => LazinessCategory::StalledFalseCompletion,
        StallCategory::NotStalledComplete => LazinessCategory::NotStalledComplete,
        StallCategory::NotStalledWaitingOnBackground => {
            LazinessCategory::NotStalledWaitingOnBackground
        }
        StallCategory::NotStalledWaitingOnUser => LazinessCategory::NotStalledWaitingOnUser,
    }
}

fn looks_like_false_completion(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    [
        "all green",
        "tests passed",
        "test result: ok",
        "cargo test",
        "production-ready",
        "ready to merge",
        "ready to ship",
        "shipped",
        "successfully completed",
        "all tests pass",
    ]
    .iter()
    .any(|cue| lower.contains(cue))
        && [
            "done", "complete", "success", "shipped", "green", "passed", "finished",
        ]
        .iter()
        .any(|cue| lower.contains(cue))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_classifier_output_round_trips_grok_build_goldens() {
        let raw = r#"{"category":"stalled_false_completion","confidence":0.88,"evidence":"final message claims `make test -race` ran clean but no run_terminal_command for make appears in the transcript."}"#;
        let parsed = parse_classifier_output(raw).expect("golden JSON");
        assert_eq!(parsed.category, LazinessCategory::StalledFalseCompletion);
        assert!((parsed.confidence - 0.88).abs() < f32::EPSILON);
        let fenced = format!("```json\n{raw}\n```");
        assert_eq!(
            parse_classifier_output(&fenced).unwrap().category,
            LazinessCategory::StalledFalseCompletion
        );
        let wrapped = format!("noise {raw} trailing");
        assert_eq!(
            parse_classifier_output(&wrapped).unwrap().category,
            LazinessCategory::StalledFalseCompletion
        );
    }

    #[test]
    fn evaluate_laziness_nudges_false_completion_above_threshold() {
        let parsed = parse_classifier_output(
            r#"{"category":"stalled_false_completion","confidence":0.88,"evidence":"claimed cargo test with no tool_call"}"#,
        )
        .unwrap();
        let decision = evaluate_laziness(&parsed, &LazinessConfig::default(), 0, 0.7);
        assert!(matches!(
            decision,
            LazinessDecision::Nudge {
                category: LazinessCategory::StalledFalseCompletion,
                ..
            }
        ));
        assert!(!build_laziness_nudge(LazinessCategory::StalledFalseCompletion, "e").is_empty());
    }

    #[test]
    fn evaluate_laziness_waiting_on_user_is_no_nudge() {
        let parsed = parse_classifier_output(
            r#"{"category":"not_stalled_waiting_on_user","confidence":0.91,"evidence":"offered to implement next"}"#,
        )
        .unwrap();
        let decision = evaluate_laziness(&parsed, &LazinessConfig::default(), 0, 0.7);
        assert_eq!(
            decision,
            LazinessDecision::NoNudge {
                category: LazinessCategory::NotStalledWaitingOnUser,
                confidence: parsed.confidence,
                reason: NoNudgeReason::NotStalled,
            }
        );
    }

    #[test]
    fn evaluate_laziness_cap_exhausted_is_no_nudge() {
        let parsed = parse_classifier_output(
            r#"{"category":"stalled_narration","confidence":0.9,"evidence":"claimed a launch with no tool"}"#,
        )
        .unwrap();
        let cfg = LazinessConfig {
            max_nudges_per_session: 1,
            ..LazinessConfig::default()
        };
        let decision = evaluate_laziness(&parsed, &cfg, 1, 0.7);
        assert!(matches!(
            decision,
            LazinessDecision::NoNudge {
                reason: NoNudgeReason::CapExhausted,
                ..
            }
        ));
        let low = parse_classifier_output(
            r#"{"category":"stalled_narration","confidence":0.2,"evidence":"weak"}"#,
        )
        .unwrap();
        assert!(matches!(
            evaluate_laziness(&low, &LazinessConfig::default(), 0, 0.7),
            LazinessDecision::NoNudge {
                reason: NoNudgeReason::LowConfidence,
                ..
            }
        ));
    }

    #[test]
    fn laziness_category_wire_strings_match_grok_build() {
        assert_eq!(
            LazinessCategory::StalledFalseCompletion.as_const_str(),
            LAZINESS_STALLED_FALSE_COMPLETION
        );
        assert_eq!(
            LazinessCategory::NotStalledWaitingOnUser.as_const_str(),
            LAZINESS_NOT_STALLED_WAITING_USER
        );
        assert_eq!(
            LazinessCategory::StalledNarration.as_const_str(),
            "stalled_narration"
        );
        assert_eq!(
            LazinessCategory::StalledPermissionAsking.as_const_str(),
            LAZINESS_STALLED_PERMISSION_ASKING
        );
        assert_eq!(
            LazinessCategory::StalledNoTodosButTaskInFlight.as_const_str(),
            LAZINESS_STALLED_NO_TODOS_BUT_TASK_IN_FLIGHT
        );
        assert_eq!(
            LazinessCategory::NotStalledComplete.as_const_str(),
            LAZINESS_NOT_STALLED_COMPLETE
        );
        assert_eq!(
            LazinessCategory::NotStalledWaitingOnBackground.as_const_str(),
            LAZINESS_NOT_STALLED_WAITING_BG
        );
    }

    #[test]
    fn claim_evidence_false_completion_without_tools() {
        let category = claim_evidence_category(
            "SUCCESS: cargo test --quiet is all green and production-ready.",
            GoalKind::CodeChange,
            false,
            false,
            false,
        );
        assert_eq!(category, LazinessCategory::StalledFalseCompletion);
        let after_edit = claim_evidence_category(
            "SUCCESS: cargo test --quiet is all green and production-ready.",
            GoalKind::CodeChange,
            false,
            false,
            false,
        );
        assert_eq!(
            after_edit,
            LazinessCategory::StalledFalseCompletion,
            "an edit without a test tool is still a false completion claim"
        );
        let with_tests = claim_evidence_category(
            "SUCCESS: cargo test --quiet is all green and production-ready.",
            GoalKind::CodeChange,
            true,
            false,
            false,
        );
        assert_ne!(with_tests, LazinessCategory::StalledFalseCompletion);
        let analysis = claim_evidence_category(
            "Here is the review.\n\nLet me implement the parser.",
            GoalKind::Analysis,
            false,
            false,
            false,
        );
        assert_eq!(analysis, LazinessCategory::NotStalledWaitingOnUser);
    }
}
