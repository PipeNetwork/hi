//! Small heuristics and formatters used by the agent loop.

use hi_ai::ToolMode;
use hi_tools::{PlanStatus, PlanStep, ToolOutput};

use crate::ui::Ui;

/// Route a tool's output to the right UI surface: a plan update drives the live
/// tracker (in place), everything else renders as a tool result — its richer
/// `display` if present, else the model-facing `content`.
pub(crate) fn emit_tool_output(ui: &mut dyn Ui, name: &str, output: &ToolOutput) {
    if let Some(plan) = output.plan.as_deref() {
        ui.plan(plan);
    } else {
        ui.tool_result(name, output.display.as_deref().unwrap_or(&output.content));
    }
}

/// Infer within-batch tool-call dependencies so the executor can honor the
/// model's intent rather than relying on emission-order coincidence. Returns,
/// for each call index, the set of earlier call indices it must run *after*.
///
/// Rules (conservative — over-serializing is safe, under-serializing is a bug):
/// - A mutating call (`write`/`edit`/`multi_edit`/`bash`/`apply_patch`) depends
///   on every earlier mutating call, so side effects apply in emission order.
///   (Two independent writes still serialize — file edits aren't commutative
///   and a later write may depend on an earlier write's content.)
/// - A read-only call depends on any earlier mutating call whose inferred
///   target path matches the read's target path — so "write a.rs, then read
///   a.rs" reads the post-write state even if a scheduler reorders independent
///   reads. Reads with no path overlap with earlier mutations have no deps and
///   may parallelize freely.
///
/// `calls` is `(id, name, arguments)` per the executor's shape. A call with an
/// unparseable target path is treated as dependent on all earlier mutations
/// (the safe fallback — `target_path` returns `None` for `bash`, so a `bash`
/// edit followed by a read serializes).
pub(crate) fn tool_deps(calls: &[(String, String, String)]) -> Vec<Vec<usize>> {
    let n = calls.len();
    let mut deps = vec![Vec::new(); n];
    // Track, for each prior index, whether it was mutating and its target path.
    let mut prior: Vec<(bool, Option<String>)> = Vec::with_capacity(n);
    for (i, (_, name, arguments)) in calls.iter().enumerate() {
        let mutating = !hi_tools::is_read_only(name);
        let my_path = hi_tools::target_path(name, arguments);
        for (j, (was_mut, their_path)) in prior.iter().enumerate() {
            let must_wait = if mutating {
                // Mutating calls serialize after all earlier mutations.
                *was_mut
            } else {
                // Reads wait for an earlier mutation on the same path. If
                // either side has no parseable path, be safe and serialize
                // (covers `bash` edits, which have no path).
                *was_mut && paths_overlap(their_path.as_deref(), my_path.as_deref())
            };
            if must_wait {
                deps[i].push(j);
            }
        }
        prior.push((mutating, my_path));
    }
    deps
}

/// Whether two (possibly-unknown) target paths refer to the same file. `None`
/// on either side means "unknown" — treat as overlapping (the safe choice:
/// serialize rather than risk a read observing a pre-mutation state).
fn paths_overlap(a: Option<&str>, b: Option<&str>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => a == b,
        // Unknown on either side → conservatively overlap.
        _ => true,
    }
}

/// Whether an execution `order` (a permutation of `0..n`) respects the
/// dependency graph from [`tool_deps`]: every call appears after all of its
/// dependencies. Used as a debug assertion / property-test oracle so a future
/// scheduler change can't regress the "read-after-write observes the write"
/// invariant.
pub(crate) fn respects_deps(deps: &[Vec<usize>], order: &[usize]) -> bool {
    let pos = |idx: usize| order.iter().position(|&o| o == idx).unwrap();
    for (i, ds) in deps.iter().enumerate() {
        let my_pos = pos(i);
        for &d in ds {
            if pos(d) > my_pos {
                return false;
            }
        }
    }
    true
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

/// Whether a plan has unfinished work — any step that is `Pending` or `Active`.
/// Used by the continue logic to keep the turn going when the model stops
/// calling tools but the plan isn't complete. The model often writes a
/// finished-looking recap after one sub-task ("I've implemented proof.rs."),
/// which the text-based `looks_like_unfinished_step` heuristic can't catch —
/// but the plan state (2/9 done) is unambiguous.
pub(crate) fn plan_has_pending_steps(steps: &[PlanStep]) -> bool {
    steps
        .iter()
        .any(|s| s.status == PlanStatus::Pending || s.status == PlanStatus::Active)
}

/// Whether a user input looks like a "continue" command — a short prompt
/// asking the agent to keep going, as opposed to a new task. Used to decide
/// whether to persist the plan state across turns: a "continue" on an
/// incomplete plan should keep the plan so the plan-aware continue logic can
/// fire; a new task should clear it so a stale plan doesn't cause spurious
/// nudges.
pub(crate) fn looks_like_continue(input: &str) -> bool {
    let lower = input.trim().to_lowercase();
    if lower.len() > 50 {
        return false; // A continue command is short; a new task is longer.
    }
    const CONTINUE_PHRASES: &[&str] = &[
        "continue",
        "keep going",
        "go on",
        "next",
        "proceed",
        "resume",
        "carry on",
        "finish it",
        "finish up",
        "do the rest",
        "do the remaining",
        "keep working",
    ];
    CONTINUE_PHRASES
        .iter()
        .any(|p| lower == *p || lower.starts_with(p))
}

/// Whether a line expresses *intent to act next* rather than a finished result.
#[allow(dead_code)]
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
#[allow(dead_code)]
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
    fn plan_pending_steps_heuristic() {
        let step = |status: PlanStatus| PlanStep {
            title: "x".into(),
            status,
        };
        // All done → no pending work.
        assert!(!plan_has_pending_steps(&[
            step(PlanStatus::Done),
            step(PlanStatus::Done),
        ]));
        // Has a pending step → unfinished.
        assert!(plan_has_pending_steps(&[
            step(PlanStatus::Done),
            step(PlanStatus::Pending),
        ]));
        // Has an active step → unfinished.
        assert!(plan_has_pending_steps(&[
            step(PlanStatus::Done),
            step(PlanStatus::Active),
            step(PlanStatus::Pending),
        ]));
        // Empty plan → no pending work (no plan to complete).
        assert!(!plan_has_pending_steps(&[]));
    }

    #[test]
    fn looks_like_continue_heuristic() {
        // Short continue commands.
        for s in [
            "continue",
            "Continue",
            "CONTINUE",
            "keep going",
            "go on",
            "next",
            "proceed",
            "resume",
            "carry on",
            "finish it",
            "do the rest",
            "keep working",
            "  continue  ",
        ] {
            assert!(looks_like_continue(s), "should flag as continue: {s:?}");
        }
        // New tasks — should NOT be flagged as continue.
        for s in [
            "fix the bug in parser.rs",
            "implement a new feature for the CLI",
            "review the codebase and suggest improvements",
            "write tests for the auth module",
            "refactor the error handling to use anyhow",
            // Too long even if it starts with "continue".
            "continue working on the feature but also make sure to handle the edge case where the input is empty and the user has not provided a valid path",
        ] {
            assert!(
                !looks_like_continue(s),
                "should NOT flag as continue: {s:?}"
            );
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

    #[test]
    fn tool_deps_serializes_read_after_write_on_same_path() {
        // write a.rs, read a.rs, read b.rs: the read of a.rs depends on the
        // write; the read of b.rs does not (different path) — independent.
        let calls = vec![
            (
                "w".into(),
                "write".into(),
                r#"{"path":"a.rs","content":"x"}"#.into(),
            ),
            ("r1".into(), "read".into(), r#"{"path":"a.rs"}"#.into()),
            ("r2".into(), "read".into(), r#"{"path":"b.rs"}"#.into()),
        ];
        let deps = tool_deps(&calls);
        // write (0) has no deps.
        assert!(deps[0].is_empty(), "write has no deps: {deps:?}");
        // read a.rs (1) depends on the write (0).
        assert!(deps[1].contains(&0), "read a.rs depends on write: {deps:?}");
        // read b.rs (2) is independent of the write on a.rs.
        assert!(
            !deps[2].contains(&0),
            "read b.rs independent of write a.rs: {deps:?}"
        );
    }

    #[test]
    fn tool_deps_serializes_mutating_calls_in_emission_order() {
        // Two writes: the second depends on the first (edits aren't commutative).
        let calls = vec![
            (
                "w1".into(),
                "write".into(),
                r#"{"path":"a.rs","content":"1"}"#.into(),
            ),
            (
                "w2".into(),
                "edit".into(),
                r#"{"path":"a.rs","old_string":"1","new_string":"2"}"#.into(),
            ),
        ];
        let deps = tool_deps(&calls);
        assert!(deps[0].is_empty());
        assert!(
            deps[1].contains(&0),
            "second write depends on first: {deps:?}"
        );
    }

    #[test]
    fn tool_deps_bash_edit_serializes_following_read() {
        // A bash edit has no parseable path, so a following read is conservatively
        // serialized after it (the safe fallback — the read might observe the
        // bash edit's effect).
        let calls = vec![
            (
                "b".into(),
                "bash".into(),
                r#"{"command":"echo x > a.rs"}"#.into(),
            ),
            ("r".into(), "read".into(), r#"{"path":"a.rs"}"#.into()),
        ];
        let deps = tool_deps(&calls);
        assert!(
            deps[1].contains(&0),
            "read after bash edit serializes: {deps:?}"
        );
    }

    #[test]
    fn respects_deps_validates_ordering() {
        // deps: call 1 depends on 0; call 2 depends on 0.
        let deps = vec![vec![], vec![0], vec![0]];
        // Emission order respects deps.
        assert!(respects_deps(&deps, &[0, 1, 2]));
        // Reordering 0 after 1 violates (1 depends on 0).
        assert!(!respects_deps(&deps, &[1, 0, 2]));
        // 2 before 1 is fine (2 doesn't depend on 1).
        assert!(respects_deps(&deps, &[0, 2, 1]));
    }

    #[test]
    fn emission_order_respects_inferred_deps_for_a_realistic_batch() {
        // The property the executor pins: for a realistic mixed batch, the
        // emission order [0,1,2,...] always respects the inferred deps (since
        // deps only point backward). This is the regression guard.
        let calls = vec![
            ("r0".into(), "read".into(), r#"{"path":"a.rs"}"#.into()),
            (
                "w".into(),
                "write".into(),
                r#"{"path":"a.rs","content":"x"}"#.into(),
            ),
            ("r1".into(), "read".into(), r#"{"path":"a.rs"}"#.into()),
            ("r2".into(), "read".into(), r#"{"path":"b.rs"}"#.into()),
        ];
        let deps = tool_deps(&calls);
        let order: Vec<usize> = (0..calls.len()).collect();
        assert!(
            respects_deps(&deps, &order),
            "emission order respects inferred deps: {deps:?}"
        );
    }

    #[test]
    fn scheduler_allows_independent_read_to_overlap_later_write() {
        // The capability the dep-aware scheduler unlocks: [read a.rs, write b.rs,
        // read c.rs] — none overlap on a path, so the scheduler may complete
        // read c.rs before write b.rs. The dep graph permits any order where
        // each call follows its (here, empty) deps. Pin that such an order
        // respects_deps, while an order that runs a dependent read before its
        // write does not.
        let calls = vec![
            ("r0".into(), "read".into(), r#"{"path":"a.rs"}"#.into()),
            (
                "w".into(),
                "write".into(),
                r#"{"path":"b.rs","content":"x"}"#.into(),
            ),
            ("r2".into(), "read".into(), r#"{"path":"c.rs"}"#.into()),
        ];
        let deps = tool_deps(&calls);
        // No path overlaps → no deps between them → any order respects deps,
        // including overlapping read c.rs ahead of write b.rs.
        assert!(
            deps.iter().all(|d| d.is_empty()),
            "independent batch has no deps: {deps:?}"
        );
        assert!(
            respects_deps(&deps, &[0, 2, 1]),
            "read c.rs may complete before write b.rs: {deps:?}"
        );

        // Contrast: a dependent read (same path as the write) must NOT complete
        // before the write.
        let dep_calls = vec![
            (
                "w".into(),
                "write".into(),
                r#"{"path":"a.rs","content":"x"}"#.into(),
            ),
            ("r".into(), "read".into(), r#"{"path":"a.rs"}"#.into()),
        ];
        let dep = tool_deps(&dep_calls);
        assert!(
            !respects_deps(&dep, &[1, 0]),
            "dependent read before write violates deps: {dep:?}"
        );
        assert!(
            respects_deps(&dep, &[0, 1]),
            "write before dependent read respects deps: {dep:?}"
        );
    }
}
