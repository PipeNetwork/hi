//! Small heuristics and formatters used by the agent loop.

use hi_ai::ToolMode;
use hi_tools::ToolOutput;

use crate::ui::Ui;

/// Route a tool's output to the right UI surface: a plan update drives the live
/// tracker (in place), everything else renders as a tool result — its richer
/// `display` if present, else the model-facing `content`.
pub(crate) fn emit_tool_output(ui: &mut dyn Ui, output: &ToolOutput) {
    if let Some(plan) = output.plan.as_deref() {
        ui.plan(plan);
    } else {
        ui.tool_result(output.display.as_deref().unwrap_or(&output.content));
    }
}

/// Humanize a token count compactly and consistently: `991`, `1.2k`, `22k`, `1.0M`.
/// Shared by the live working line and the settled usage summary so they agree.
pub fn humanize_count(n: u64) -> String {
    match n {
        0..=999 => n.to_string(),
        1_000..=9_999 => format!("{:.1}k", n as f64 / 1000.0),
        10_000..=999_999 => format!("{}k", n / 1000),
        _ => format!("{:.1}M", n as f64 / 1_000_000.0),
    }
}

pub(crate) fn tool_mode_label(mode: ToolMode) -> &'static str {
    match mode {
        ToolMode::Auto => "auto",
        ToolMode::Required => "required",
        ToolMode::ChatOnly => "chat-only",
        ToolMode::ReadOnly => "read-only",
    }
}

pub(crate) fn looks_mutating(input: &str) -> bool {
    let s = input.to_ascii_lowercase();
    [
        "edit",
        "fix",
        "change",
        "update",
        "write",
        "create",
        "delete",
        "remove",
        "rename",
        "implement",
        "add ",
        "modify",
        "refactor",
        "format",
        "run ",
    ]
    .iter()
    .any(|needle| s.contains(needle))
}

/// Heuristic: does the model's final text read like an *announced but unperformed*
/// next step — e.g. "Now let me rewrite main.rs:" or a "Here's my plan:" followed
/// by a numbered to-do list — rather than a finished answer or a past-tense recap?
///
/// It judges the trailing non-empty line, with one twist: when the message trails
/// off into a plan/to-do list, the intent lives in the line that *introduces* the
/// list ("Here's my plan:"), not the last bullet — so it judges that lead-in
/// instead, and only when the lead-in looks forward. That way a proper codex-style
/// recap that ends in a bullet list ("Key changes:\n- …") doesn't read as a stall,
/// while a model that announces a plan and quits without doing it does.
pub(crate) fn looks_like_unfinished_step(text: &str) -> bool {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let Some(&last) = lines.last() else {
        return false;
    };
    if is_list_item(last) {
        // Trailing plan/to-do list: unfinished only if the line introducing it
        // looks forward ("Here's my plan:"). A past-tense recap list is done.
        let lead = lines
            .iter()
            .rev()
            .find(|l| !is_list_item(l))
            .copied()
            .unwrap_or(last);
        return is_forward_intent(lead);
    }
    // Otherwise judge the trailing line: a dangling colon ("Now let me rewrite
    // main.rs:") or a forward-looking phrase means work was announced, not done.
    last.ends_with(':') || is_forward_intent(last)
}

/// Whether a line expresses *intent to act next* rather than a finished result.
pub(crate) fn is_forward_intent(line: &str) -> bool {
    let lower = line.to_lowercase();
    // Courtesy closings address the *user* ("let me know if…", "I'll be happy
    // to…", "I'll let you know…") — they read like forward phrases but mean the
    // turn is finished, not stalled. Vetoed first so they don't trigger a nudge.
    const CLOSINGS: [&str; 6] = [
        "let me know",
        "i'll be happy",
        "i'll let you",
        "i'll wait",
        "i'm happy to",
        "feel free",
    ];
    if CLOSINGS.iter().any(|c| lower.contains(c)) {
        return false;
    }
    const FORWARD_INTENT: [&str; 12] = [
        "let me ",
        "let's ",
        "i'll ",
        "i will ",
        "i'm going to",
        "i am going to",
        "proceed to ",
        "here's my plan",
        "here is my plan",
        "my plan",
        "i need to ",
        "next, i",
    ];
    FORWARD_INTENT.iter().any(|phrase| lower.contains(phrase))
}

/// Whether a line is a markdown list item — a bullet (`- `, `* `, `• `) or a
/// numbered item (`1.`, `2)`) — used to spot a trailing plan/to-do list.
pub(crate) fn is_list_item(line: &str) -> bool {
    let l = line.trim_start();
    if l.starts_with("- ") || l.starts_with("* ") || l.starts_with("• ") {
        return true;
    }
    let digits = l.chars().take_while(|c| c.is_ascii_digit()).count();
    digits > 0 && l[digits..].starts_with(['.', ')'])
}

/// Whether recovery sampling (a hotter resample on an empty/garbled retry) is on.
/// Off (`HI_RECOVERY_SAMPLING=0/off/false/no`) re-runs the retry at the configured
/// sampling — the knob for A/B-ing recovery on the eval harness. Read once.
pub(crate) static RECOVERY_SAMPLING: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    !matches!(
        std::env::var("HI_RECOVERY_SAMPLING").ok().as_deref(),
        Some("0" | "off" | "false" | "no")
    )
});

/// Sampling for a model round, escalating with the count of consecutive
/// content-less rounds (`retries`; 0 = the normal first attempt). Returns
/// `(temperature, top_p, frequency_penalty)`. On a normal round — or when recovery
/// sampling is disabled — it passes the configured temperature through and leaves
/// `top_p`/`frequency_penalty` at the provider default (`None`). On a retry it
/// leads with anti-repetition — nucleus sampling plus a growing frequency penalty
/// — and only gently raises temperature from a ≥0.5 floor, so a repetition/garble
/// loop is broken with less coding-quality risk than a big temperature jump.
pub(crate) fn recovery_sampling(
    retries: u32,
    base_temperature: Option<f32>,
    enabled: bool,
) -> (Option<f32>, Option<f32>, Option<f32>) {
    if !enabled || retries == 0 {
        return (base_temperature, None, None);
    }
    let r = retries as f32;
    let temperature = (base_temperature.unwrap_or(0.7).max(0.5) + 0.15 * r).min(1.0);
    let frequency_penalty = (0.3 * r).min(0.6);
    (Some(temperature), Some(0.95), Some(frequency_penalty))
}

/// Which stall mode fired and triggered recovery sampling. The retry counter
/// (`retries`) is shared across the empty-response path — repeat and continue
/// nudges don't currently escalate sampling, so they surface as `mode == …` with
/// `retries == 0` and produce no telemetry line (see `recovery_telemetry`).
///
/// `Repeat`/`Continue` are modeled but not yet constructed: the plan calls out a
/// separate experiment on whether they should escalate sampling too. They're
/// kept here so the telemetry shape is fixed when that lands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StallMode {
    /// A content-less/garbled round (`EmptyCompletion`/`MalformedStream`, or no
    /// text and no tool calls). The only mode recovery sampling escalates today.
    Empty,
    /// The model re-issued the previous round's exact tool calls.
    #[allow(dead_code)]
    Repeat,
    /// The model announced a next step but emitted no tool call to perform it.
    #[allow(dead_code)]
    Continue,
}

impl StallMode {
    pub(crate) fn label(self) -> &'static str {
        match self {
            StallMode::Empty => "empty retry",
            StallMode::Repeat => "repeat nudge",
            StallMode::Continue => "continue nudge",
        }
    }
}

/// Build the recovery-sampling telemetry line, or `None` when there's nothing to
/// report. Emits only when recovery sampling is actually changing params — i.e.
/// `enabled && retries > 0` (a normal first attempt, or recovery disabled, stays
/// silent so ordinary runs aren't noisy). That keeps it behind the
/// `HI_RECOVERY_SAMPLING` A/B knob without needing a separate debug env: the line
/// appears precisely when the knob is on *and* a retry is being resampled, which is
/// the signal the A/B needs to measure rather than just aggregate.
///
/// The line names the stall mode, the retry index out of the per-mode budget, and
/// the applied sampling params, e.g.
/// `recovery sampling: empty retry 1/2 · temp=0.65 top_p=0.95 freq=0.3`.
pub(crate) fn recovery_telemetry(
    mode: StallMode,
    retries: u32,
    budget: u32,
    temperature: Option<f32>,
    top_p: Option<f32>,
    frequency_penalty: Option<f32>,
    enabled: bool,
) -> Option<String> {
    if !enabled || retries == 0 {
        return None;
    }
    let fmt = |v: Option<f32>| v.map(|x| format!("{x:.2}")).unwrap_or_else(|| "—".into());
    Some(format!(
        "recovery sampling: {} {}/{} · temp={} top_p={} freq={}",
        mode.label(),
        retries,
        budget,
        fmt(temperature),
        fmt(top_p),
        fmt(frequency_penalty),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn humanize_count_abbreviates_consistently() {
        assert_eq!(humanize_count(0), "0");
        assert_eq!(humanize_count(991), "991");
        assert_eq!(humanize_count(1234), "1.2k");
        assert_eq!(humanize_count(22864), "22k"); // the reported "22864 in"
        assert_eq!(humanize_count(12000), "12k"); // the reported "12k" ctx
        assert_eq!(humanize_count(999_999), "999k"); // last "k" before switching
        assert_eq!(humanize_count(1_000_000), "1.0M"); // a 1M window
        // A long session's cumulative input must read as millions, never a
        // 5-digit "15528k" (the pre-fix formatter that prompted this).
        assert_eq!(humanize_count(15_528_000), "15.5M");
    }

    #[test]
    fn unfinished_step_heuristic() {
        for t in [
            "Now let me rewrite main.rs:",
            "I'll add the struct",
            "Here is the plan:",
            // A "plan:" lead-in followed by a numbered to-do list — the trailing
            // line is a list item, so the lead-in is what's judged. (This is the
            // case the old line-only heuristic missed, ending the turn mid-plan.)
            "Now let me make the fixes. Here's my plan:\n\n1. Remove deps\n2. Fix gitignore\n3. Drop dead code",
        ] {
            assert!(looks_like_unfinished_step(t), "should flag: {t:?}");
        }
        for t in [
            "Done. Run `cargo build`.",
            "The answer is 42.",
            "I changed foo.rs and bar.rs.",
            // A past-tense recap that ends in a bullet list is finished, not a
            // stall — the lead-in ("Key changes:") looks back, not forward.
            "Key changes:\n- Added GOP support in encoder.rs\n- Updated the CLI in main.rs",
            // Courtesy closings address the user — a finished turn, not a stall —
            // even though they contain "let me"/"I'll". These used to false-nudge.
            "All done. Let me know if you'd like any changes.",
            "I'll be happy to help with anything else.",
            "Implemented and tested. I'll let you know if I spot any issues.",
            "Fixed it — feel free to ask if you want more detail.",
        ] {
            assert!(!looks_like_unfinished_step(t), "should not flag: {t:?}");
        }
    }

    #[test]
    fn recovery_sampling_escalates_and_toggles() {
        // Normal round: pass the configured temperature through, no overrides.
        assert_eq!(
            recovery_sampling(0, Some(0.2), true),
            (Some(0.2), None, None)
        );
        // First retry: nucleus + frequency penalty lead; temperature rises only
        // gently from the 0.5 floor (to ~0.65, well under the old 0.85).
        let (t1, p1, f1) = recovery_sampling(1, Some(0.2), true);
        assert_eq!((p1, f1), (Some(0.95), Some(0.3)));
        assert!(
            t1.unwrap() > 0.2 && t1.unwrap() < 0.7,
            "temp climbs gently: {t1:?}"
        );
        // Second retry climbs further; temperature and penalty stay bounded.
        let (t2, _, f2) = recovery_sampling(2, Some(0.2), true);
        assert!(t2.unwrap() > t1.unwrap(), "temp keeps climbing");
        assert!(f2.unwrap() > f1.unwrap(), "penalty grows");
        assert!(t2.unwrap() <= 1.0 && f2.unwrap() <= 0.6, "both bounded");
        // Disabled: a retry behaves like a normal round (no overrides).
        assert_eq!(
            recovery_sampling(2, Some(0.2), false),
            (Some(0.2), None, None)
        );
    }

    #[test]
    fn recovery_telemetry_only_when_params_change() {
        // A retry with recovery on names the stall mode, retry index, budget, and
        // the applied sampling params.
        let line = recovery_telemetry(
            StallMode::Empty,
            1,
            2,
            Some(0.65),
            Some(0.95),
            Some(0.3),
            true,
        )
        .expect("retry with recovery on should produce a line");
        assert!(
            line.contains("empty retry 1/2"),
            "expected mode + retry/budget, got {line:?}"
        );
        assert!(
            line.contains("temp=0.65") && line.contains("top_p=0.95") && line.contains("freq=0.30"),
            "expected applied params, got {line:?}"
        );

        // A normal first attempt (retries == 0) is silent regardless of mode or
        // enabled state — ordinary runs must not be noisy.
        assert_eq!(
            recovery_telemetry(StallMode::Empty, 0, 2, Some(0.2), None, None, true),
            None,
            "retries == 0 should not emit"
        );
        // Repeat/continue nudges don't escalate sampling (retries stays 0), so they
        // produce no telemetry line.
        assert_eq!(
            recovery_telemetry(StallMode::Repeat, 0, 3, Some(0.2), None, None, true),
            None,
        );
        assert_eq!(
            recovery_telemetry(StallMode::Continue, 0, 3, Some(0.2), None, None, true),
            None,
        );
        // Recovery disabled: a retry behaves like a normal round, so no line.
        assert_eq!(
            recovery_telemetry(StallMode::Empty, 2, 2, Some(0.2), None, None, false),
            None,
            "disabled recovery should not emit"
        );
    }
}
