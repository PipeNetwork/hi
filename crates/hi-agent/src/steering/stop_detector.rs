//! Grok-build last-paragraph premature-stop panel (`goal_stop_detector.rs`).
//!
//! Only the last non-empty paragraph is considered, and only line-initial
//! bail/hand-off phrasings match. "Let me implement…" is intentionally absent:
//! that is a next-step offer, not a stall.

/// Bail-out: the model surrendered instead of delivering.
pub(crate) const PATTERN_UNABLE_TO_PROCEED: &str = "unable_to_proceed";
pub(crate) const PATTERN_GIVING_UP: &str = "giving_up";
pub(crate) const PATTERN_STOPPING_HERE: &str = "stopping_here";
pub(crate) const PATTERN_PLEASE_DEFLECTION: &str = "please_deflection";

/// Goal-drive hand-off: used by grok-build to keep an active goal moving.
pub(crate) const PATTERN_AGENTS_IN_FLIGHT: &str = "agents_in_flight";
pub(crate) const PATTERN_CHECK_BACK_LATER: &str = "check_back_later";
pub(crate) const PATTERN_VERDICT_LINE: &str = "verdict_line";
pub(crate) const PATTERN_COMMIT_PUSH_PR: &str = "commit_push_pr";
pub(crate) const PATTERN_READY_FOR_REVIEW: &str = "ready_for_review";

const BAIL_OUT: &[&str] = &[
    PATTERN_UNABLE_TO_PROCEED,
    PATTERN_GIVING_UP,
    PATTERN_STOPPING_HERE,
    PATTERN_PLEASE_DEFLECTION,
];

/// Last-paragraph match, grok-build's `matched_stop_pattern`.
pub(crate) fn matched_stop_pattern(text: &str) -> Option<&'static str> {
    let normalised = normalise_line_endings(text);
    let last = last_non_empty_paragraph(&normalised)?;
    last.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .find_map(line_pattern)
}

/// `\r\n` and bare `\r` must split paragraphs the same as `\n`.
fn normalise_line_endings(text: &str) -> std::borrow::Cow<'_, str> {
    if text.contains('\r') {
        std::borrow::Cow::Owned(text.replace("\r\n", "\n").replace('\r', "\n"))
    } else {
        std::borrow::Cow::Borrowed(text)
    }
}

/// Wrap-up / forced-final only treats surrender phrasings as unusable.
/// Hand-off lines ("Ready for review", "Opened PR") are valid recaps.
pub(crate) fn matched_bail_out(text: &str) -> Option<&'static str> {
    let label = matched_stop_pattern(text)?;
    BAIL_OUT.contains(&label).then_some(label)
}

/// Grok-build `GOAL_CONTINUATION_BAIL_PREFACE`: last-paragraph surrender
/// while code-change / goal work is still owed.
pub(crate) const BAIL_CONTINUE_NUDGE: &str = "You appear to be stopping or handing off, but the work is NOT complete. Do not end the turn here — keep working. Use your tools on the next concrete step.";

fn last_non_empty_paragraph(text: &str) -> Option<&str> {
    text.split("\n\n")
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .last()
}

fn line_pattern(line: &str) -> Option<&'static str> {
    let lower = line.to_ascii_lowercase();
    if unable_to_proceed(&lower) {
        return Some(PATTERN_UNABLE_TO_PROCEED);
    }
    if giving_up(&lower) {
        return Some(PATTERN_GIVING_UP);
    }
    if stopping_here(&lower) {
        return Some(PATTERN_STOPPING_HERE);
    }
    if please_deflection(&lower, line) {
        return Some(PATTERN_PLEASE_DEFLECTION);
    }
    if agents_in_flight(&lower) {
        return Some(PATTERN_AGENTS_IN_FLIGHT);
    }
    if check_back_later(&lower) {
        return Some(PATTERN_CHECK_BACK_LATER);
    }
    if lower.starts_with("verdict: pass") || lower.starts_with("verdict: fail") {
        return Some(PATTERN_VERDICT_LINE);
    }
    if commit_push_pr(line) {
        return Some(PATTERN_COMMIT_PUSH_PR);
    }
    if ready_for_review(&lower) {
        return Some(PATTERN_READY_FOR_REVIEW);
    }
    None
}

fn unable_to_proceed(lower: &str) -> bool {
    let rest = lower
        .strip_prefix("i can't ")
        .or_else(|| lower.strip_prefix("i cant "))
        .or_else(|| lower.strip_prefix("i cannot "))
        .or_else(|| lower.strip_prefix("i am unable to "))
        .or_else(|| lower.strip_prefix("i'm unable to "));
    rest.is_some_and(|rest| {
        rest.starts_with("proceed")
            || rest.starts_with("continue")
            || rest.starts_with("make progress")
            || rest.starts_with("make any progress")
            || rest.starts_with("complete")
            || rest.starts_with("fix this")
    })
}

fn giving_up(lower: &str) -> bool {
    lower.starts_with("giving up")
        || lower.starts_with("i'm giving up")
        || lower.starts_with("i am giving up")
        || lower.starts_with("the task is not actionable")
}

fn stopping_here(lower: &str) -> bool {
    let rest = lower
        .strip_prefix("stopping here")
        .or_else(|| lower.strip_prefix("i've stopped here"))
        .or_else(|| lower.strip_prefix("parked the branch"))
        .or_else(|| lower.strip_prefix("parked this branch"))
        .or_else(|| lower.strip_prefix("paused here"));
    let Some(rest) = rest else {
        return false;
    };
    rest.is_empty()
        || rest.starts_with('.')
        || rest.starts_with(',')
        || rest.starts_with(';')
        || rest.starts_with(" for ")
        || rest.starts_with(" until")
        || rest.starts_with(" pending")
        || rest.starts_with(" since")
        || rest.starts_with(" because")
        || rest.starts_with(" —")
        || rest.starts_with(" -")
}

fn please_deflection(lower: &str, original: &str) -> bool {
    let Some(rest) = lower.strip_prefix("please ") else {
        return false;
    };
    if [
        "start",
        "run",
        "provide",
        "grant",
        "export",
        "add",
        "install",
        "configure",
        "give me",
        "paste",
        "point me",
        "set the ",
        "set up ",
    ]
    .iter()
    .any(|verb| rest.starts_with(verb))
    {
        return true;
    }
    // Grok-build: `Please set GROK_API_KEY` / `Please set `ENV``. The env
    // token is matched on the original line so lowercasing cannot hide A-Z.
    let Some((_, after_please)) = original.split_once(char::is_whitespace) else {
        return false;
    };
    let after = after_please.trim_start();
    if !after.to_ascii_lowercase().starts_with("set ") {
        return false;
    }
    let token = after[4..].trim_start().trim_start_matches('`');
    let name: String = token
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    name.chars().next().is_some_and(|c| c.is_ascii_uppercase())
        && name
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

fn agents_in_flight(lower: &str) -> bool {
    if lower == "waiting." || lower == "waiting" {
        return true;
    }
    if lower.starts_with("waiting for ") || lower.starts_with("agents will report back") {
        return true;
    }
    let body = lower.strip_prefix("continuous ").unwrap_or(lower);
    if let Some(rest) = ["loop", "cron", "crons", "babysit"]
        .iter()
        .find_map(|noun| body.strip_prefix(noun))
    {
        let rest = rest.trim_start();
        return rest.is_empty()
            || rest.starts_with('.')
            || rest.starts_with("active")
            || rest.starts_with("healthy")
            || rest.starts_with("continuing")
            || rest.starts_with("running")
            || rest.starts_with("will keep")
            || rest.starts_with("continues");
    }
    let words: Vec<&str> = lower.split_whitespace().collect();
    let Some(count) = words.first() else {
        return false;
    };
    let count = count.trim_start_matches('*');
    if count.is_empty() || !count.bytes().all(|b| b.is_ascii_digit()) || count == "0" {
        return false;
    }
    let Some(noun) = words.get(1) else {
        return false;
    };
    let noun = noun.trim_end_matches(|c: char| !c.is_ascii_alphanumeric());
    let noun = noun.trim_end_matches('s');
    matches!(
        noun,
        "agent" | "cron" | "task" | "fork" | "job" | "worker" | "pr" | "check"
    ) && words.iter().any(|word| {
        let word = word.trim_end_matches(|c: char| !c.is_ascii_alphanumeric());
        matches!(
            word,
            "flight" | "remaining" | "active" | "running" | "working" | "pending" | "launched"
        )
    })
}

fn check_back_later(lower: &str) -> bool {
    let rest = lower
        .strip_prefix("i will ")
        .or_else(|| lower.strip_prefix("i'll "))
        .or_else(|| lower.strip_prefix("will "));
    let Some(rest) = rest else {
        return false;
    };
    let action = [
        "check back",
        "recheck",
        "re-check",
        "poll",
        "look again",
        "retry",
        "rerun",
        "re-run",
        "try again",
    ]
    .iter()
    .find_map(|verb| rest.strip_prefix(verb).map(str::trim_start));
    let Some(tail) = action else {
        return false;
    };
    if tail.starts_with("in ") || tail.starts_with("again") {
        return true;
    }
    let target = ["when ", "once ", "after ", "until "]
        .iter()
        .find_map(|conj| tail.strip_prefix(conj));
    let Some(target) = target else {
        return false;
    };
    let token = target
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .next()
        .unwrap_or("");
    !matches!(token, "you" | "your")
}

fn commit_push_pr(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    if lower.starts_with("pushed to ") {
        return true;
    }
    if let Some(rest) = lower.strip_prefix("pushed `") {
        return hex_prefix_len(rest) >= 7;
    }
    if let Some(rest) = lower
        .strip_prefix("committed as ")
        .or_else(|| lower.strip_prefix("commit: "))
    {
        return hex_prefix_len(rest.trim_start_matches('`')) >= 7;
    }
    if let Some(rest) = lower
        .strip_prefix("opened pr")
        .or_else(|| lower.strip_prefix("created pr"))
    {
        let rest = rest.trim_start().trim_start_matches('#');
        return rest.chars().next().is_some_and(|c| c.is_ascii_digit());
    }
    false
}

fn hex_prefix_len(s: &str) -> usize {
    s.chars().take_while(|c| c.is_ascii_hexdigit()).count()
}

fn ready_for_review(lower: &str) -> bool {
    lower.starts_with("ready for review")
        || lower.starts_with("ready to upload")
        || lower.starts_with("ready to merge")
        || lower.starts_with("ready to ship")
        || lower.starts_with("ready to land")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fires(text: &str) -> bool {
        matched_stop_pattern(text).is_some()
    }

    #[test]
    fn unable_to_proceed_canonical_phrases_trigger() {
        for phrase in [
            "I can't proceed.",
            "I cannot continue.",
            "I can't make any progress here.",
            "I am unable to complete this task.",
            "I cant fix this without help.",
        ] {
            assert_eq!(
                matched_stop_pattern(phrase),
                Some(PATTERN_UNABLE_TO_PROCEED),
                "should flag bail phrase: {phrase}",
            );
        }
    }

    #[test]
    fn giving_up_phrases_trigger() {
        for phrase in [
            "Giving up.",
            "I'm giving up on this branch.",
            "I am giving up.",
            "The task is not actionable as stated.",
        ] {
            assert_eq!(
                matched_stop_pattern(phrase),
                Some(PATTERN_GIVING_UP),
                "should flag giving-up phrase: {phrase}",
            );
        }
    }

    #[test]
    fn stopping_here_phrases_trigger() {
        for phrase in [
            "Stopping here.",
            "I've stopped here pending review.",
            "Parked the branch until you confirm.",
            "Paused here because the gate failed.",
            "Stopping here for now.",
            "Stopping here, will come back later.",
            "Paused here; review needed.",
        ] {
            assert_eq!(
                matched_stop_pattern(phrase),
                Some(PATTERN_STOPPING_HERE),
                "should flag stopping-here phrase: {phrase}",
            );
        }
        assert!(matched_stop_pattern("Stopping hereafter we ship").is_none());
        assert!(
            matched_stop_pattern("Stopping here forever, I quit.").is_none(),
            "`for` without trailing space must NOT match the ` for ` trailer",
        );
        assert_eq!(
            matched_bail_out("Stopping here."),
            Some(PATTERN_STOPPING_HERE)
        );
    }

    #[test]
    fn agents_in_flight_phrases_trigger() {
        for phrase in [
            "3 agents in flight.",
            "2 PRs remaining.",
            "Loop active.",
            "Continuous loop continuing.",
            "Waiting for the cron.",
            "Agents will report back.",
            "Waiting.",
        ] {
            assert_eq!(
                matched_stop_pattern(phrase),
                Some(PATTERN_AGENTS_IN_FLIGHT),
                "should flag hand-off phrase: {phrase}",
            );
        }
    }

    #[test]
    fn verdict_line_triggers() {
        assert_eq!(
            matched_stop_pattern("VERDICT: PASS"),
            Some(PATTERN_VERDICT_LINE),
        );
        assert_eq!(
            matched_stop_pattern("VERDICT: FAIL"),
            Some(PATTERN_VERDICT_LINE),
        );
        assert!(matched_stop_pattern("VERDICT: maybe").is_none());
    }

    #[test]
    fn check_back_later_basic_phrases_trigger() {
        for phrase in [
            "I'll check back in 5 minutes.",
            "I will retry once the build is green.",
            "I'll re-run when the queue clears.",
            "Will poll again in a bit.",
        ] {
            assert_eq!(
                matched_stop_pattern(phrase),
                Some(PATTERN_CHECK_BACK_LATER),
                "should flag check-back phrase: {phrase}",
            );
        }
    }

    #[test]
    fn check_back_later_walks_all_non_user_targets() {
        for conjunction in ["when", "once", "after", "until"] {
            for target in [
                "the build",
                "it settles",
                "this lands",
                "that passes",
                "they merge",
                "I retry",
                "tests pass",
                "CI is green",
                "we deploy",
                "stuff lands",
                "Susan signs off",
            ] {
                let line = format!("I'll retry {conjunction} {target}.");
                assert_eq!(
                    matched_stop_pattern(&line),
                    Some(PATTERN_CHECK_BACK_LATER),
                    "should flag {line}",
                );
            }
        }
    }

    #[test]
    fn check_back_later_does_not_flag_user_deferrals() {
        for line in [
            "I'll check back when your patch lands.",
            "I'll check back when you confirm.",
            "I'll retry once Your team reviews.",
            "I will re-run after YOU sign off.",
        ] {
            assert!(
                matched_stop_pattern(line).is_none(),
                "user deferrals must not fire: {line}",
            );
        }
    }

    #[test]
    fn check_back_later_user_pronoun_requires_word_boundary() {
        for line in [
            "I'll check back when yours arrives.",
            "I'll check back when your_team approves.",
            "I'll check back when youthful errors return.",
        ] {
            assert_eq!(
                matched_stop_pattern(line),
                Some(PATTERN_CHECK_BACK_LATER),
                "non-pronoun trailer must fire as a bail: {line}",
            );
        }
    }

    #[test]
    fn commit_push_pr_phrases_trigger() {
        for phrase in [
            "Pushed to `feature/branch`",
            "Pushed to `abcdef1234567`",
            "Committed as `abcdef1234`",
            "Commit: abcdef1234",
            "Opened PR #123",
            "Created PR #4567",
        ] {
            assert_eq!(
                matched_stop_pattern(phrase),
                Some(PATTERN_COMMIT_PUSH_PR),
                "should flag commit/push/PR hand-off: {phrase}",
            );
        }
        assert!(matched_stop_pattern("Commit: abc").is_none());
        assert!(
            matched_stop_pattern("Pushed `abc`").is_none(),
            "<7 hex must not fire on the Pushed-without-`to` backtick branch",
        );
        assert!(
            matched_stop_pattern("Committed as abcde").is_none(),
            "<7 hex must not fire on the Committed branch",
        );
        assert!(
            matched_stop_pattern("Opened PR").is_none(),
            "no number after PR must not fire",
        );
        assert!(matched_bail_out("Opened PR #12").is_none());
    }

    #[test]
    fn ready_for_review_phrases_trigger() {
        for phrase in [
            "Ready for review.",
            "Ready to merge.",
            "Ready to ship soon.",
            "Ready to land tomorrow.",
            "Ready to upload now.",
        ] {
            assert_eq!(
                matched_stop_pattern(phrase),
                Some(PATTERN_READY_FOR_REVIEW),
                "should flag ready-for-X hand-off: {phrase}",
            );
        }
        assert!(matched_stop_pattern("Ready to work on the next item").is_none());
        assert!(matched_bail_out("Ready for review.").is_none());
    }

    #[test]
    fn please_deflection_phrases_trigger() {
        for phrase in [
            "Please start the deploy.",
            "Please run the migrations.",
            "Please provide the credentials.",
            "Please grant access.",
            "Please export the dump.",
            "Please add the env var.",
            "Please install Docker.",
            "Please configure SSO.",
            "Please give me the token.",
            "Please paste the log.",
            "Please point me at the source.",
            "Please set the GROK_API_KEY.",
            "Please set up the cluster.",
            "Please set the auth header.",
            "Please set GROK_API_KEY.",
        ] {
            assert_eq!(
                matched_stop_pattern(phrase),
                Some(PATTERN_PLEASE_DEFLECTION),
                "should flag please-deflection: {phrase}",
            );
        }
        assert!(matched_stop_pattern("Please review the PR when you have time").is_none());
    }

    #[test]
    fn mid_sentence_phrasing_does_not_trigger() {
        let text = "Although I can't continue here without confirmation, \
                    I will keep iterating in the next turn.";
        assert!(
            !fires(text),
            "patterns are line-initial; mid-sentence mention must not fire",
        );
    }

    #[test]
    fn trailing_whitespace_is_tolerated() {
        assert!(fires("Giving up.   \n   \n"));
        assert!(fires("   Giving up.\n"));
    }

    #[test]
    fn last_paragraph_match_triggers_when_earlier_paragraphs_innocuous() {
        let text = "Wrote the fix and added a regression test.\n\
                    Running the suite now.\n\n\
                    \n\n\
                    Giving up.";
        assert!(fires(text));
    }

    #[test]
    fn earlier_paragraph_match_does_not_trigger() {
        let text = "Giving up on the old plan.\n\n\
                    Switched to the new one and finished the integration test. \
                    Re-running the suite to confirm.";
        assert!(!fires(text));
        let text = "I can't proceed with the old approach.\n\nHere is the actual review.";
        assert!(matched_stop_pattern(text).is_none());
    }

    #[test]
    fn crlf_line_endings_do_not_break_paragraph_split() {
        let crlf = "Wrote the fix.\r\n\r\nGiving up.\r\n";
        assert_eq!(
            matched_stop_pattern(crlf),
            Some(PATTERN_GIVING_UP),
            "CRLF last paragraph must still fire",
        );
        let mixed = "Giving up on the old plan.\r\n\r\n\
                     Switched to the new one and finished the integration test.\n";
        assert!(
            !fires(mixed),
            "CRLF earlier-paragraph bail must NOT fire: {mixed}",
        );
        let bare_cr = "work\rGiving up.\r";
        assert_eq!(
            matched_stop_pattern(bare_cr),
            Some(PATTERN_GIVING_UP),
            "bare CR must normalise into a line break",
        );
    }

    #[test]
    fn empty_input_does_not_trigger() {
        assert!(!fires(""));
        assert!(!fires("   \n\n  \n"));
    }

    #[test]
    fn ordinary_progress_narration_does_not_trigger() {
        for phrase in [
            "Implemented the helper and wired it through the planner.",
            "Tests pass on the fast path; one flake remains in the slow path.",
            "Ran cargo fmt and clippy; both are clean.",
            "Next: extend the gate to cover the resume site.",
        ] {
            assert!(!fires(phrase), "progress narration must not fire: {phrase}");
        }
    }

    #[test]
    fn multi_line_paragraph_any_line_can_match() {
        let text = "Update on the run:\n\
                    Tests are green.\n\
                    Giving up on the doc rewrite.";
        assert_eq!(matched_stop_pattern(text), Some(PATTERN_GIVING_UP));
    }

    #[test]
    fn let_me_implement_is_not_a_stop() {
        let text = "Here are three issues.\n\nLet me implement fixes for #1, #2, and #3.";
        assert!(matched_stop_pattern(text).is_none());
        assert!(matched_bail_out(text).is_none());
    }
}
