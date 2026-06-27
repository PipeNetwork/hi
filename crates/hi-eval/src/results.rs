use serde::{Deserialize, Serialize};

/// Per-turn trajectory telemetry mirrored from the agent's `TurnTelemetry` /
/// `TurnAttribution`, captured in the `--report` JSON so the eval harness can
/// diagnose *how* a turn went (verify rounds, recovery retries, nudges fired,
/// where the last verify failure pointed) — not just whether it passed.
/// Deserialized here (not reusing the agent types) so hi-eval doesn't depend
/// on hi-agent.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Trajectory {
    pub verify_rounds: u32,
    pub recovery_retries: u32,
    pub repeat_nudges: u32,
    pub continue_nudges: u32,
    pub hit_step_cap: bool,
    pub stalled_unfinished: bool,
    pub stalled_repeating: bool,
    pub verify_attributions: Vec<TrajectoryAttribution>,
    /// Scheduler parallelism: total tool calls this turn.
    #[serde(default)]
    pub tool_calls: u32,
    /// Largest concurrent ready-batch (1 = all serial).
    #[serde(default)]
    pub max_concurrent_batch: u32,
    /// Calls that ran serially (bash or a lone ready call).
    #[serde(default)]
    pub serial_runs: u32,
    /// Per-tool-call timeline: name, path, duration (ms), error flag.
    /// Ordered by execution completion. Empty when no tools ran.
    #[serde(default)]
    pub tool_timeline: Vec<TrajectoryToolCall>,
}

/// One entry in the per-turn tool-call timeline.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrajectoryToolCall {
    pub tool: String,
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub duration_ms: u64,
    #[serde(default)]
    pub error: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrajectoryAttribution {
    pub path: String,
    pub line: Option<u32>,
    pub column: Option<u32>,
    pub message: String,
    pub kind: String,
}

impl Trajectory {
    /// The total number of "extra" model nudges this turn required beyond a
    /// clean one-shot: verify rounds + recovery retries + repeat/continue
    /// nudges. A clean turn is 0; higher means the model needed more steering.
    pub fn extra_rounds(&self) -> u32 {
        self.verify_rounds + self.recovery_retries + self.repeat_nudges + self.continue_nudges
    }
}

pub struct RunResult {
    pub config: String,
    pub task: String,
    pub trial: usize,
    pub passed: bool,
    /// Why it failed (None when it passed) — for the failure-mode breakdown.
    pub fail: Option<FailKind>,
    pub provider_error_kind: Option<String>,
    pub compat_fallbacks_used: Vec<String>,
    pub changed_files: Vec<String>,
    pub verify_output_summary: String,
    pub failure_confidence: Option<&'static str>,
    pub candidates: usize,
    pub cost_usd: f64,
    pub tokens: u64,
    pub seconds: f64,
    /// Trajectory of the representative (furthest-progressing) candidate —
    /// verify rounds, recovery retries, nudges, last verify attribution.
    pub trajectory: Trajectory,
}

pub struct Candidate {
    pub passed: bool,
    pub fail: Option<FailKind>,
    pub provider_error_kind: Option<String>,
    pub compat_fallbacks_used: Vec<String>,
    pub changed_files: Vec<String>,
    pub verify_output_summary: String,
    pub failure_confidence: Option<&'static str>,
    pub cost_usd: f64,
    pub tokens: u64,
    pub seconds: f64,
    pub trajectory: Trajectory,
}

/// Why a candidate failed — so the summary shows *where* hi loses, not just how
/// often. Ordered by how far the attempt got (Error = least, Logic = most).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FailKind {
    /// `hi` itself errored/crashed (provider failure, non-zero exit).
    Error,
    /// The model changed no files — answered, gave up, or never acted.
    NoEdits,
    /// Files changed but the code doesn't build/load (compile/type/import error).
    Compile,
    /// Builds and runs, but behavior is wrong (the model's rule was off).
    Logic,
}

#[derive(Serialize)]
pub struct RunArtifact {
    pub task: String,
    pub config: String,
    pub trial: usize,
    pub profile: String,
    /// Whether the tool-output condenser was on for this run (for the A/B).
    pub condense: bool,
    /// Whether recovery sampling was on for this run (for the A/B).
    pub recovery: bool,
    pub passed: bool,
    pub failure_bucket: Option<String>,
    pub failure_confidence: Option<&'static str>,
    pub changed_files: Vec<String>,
    pub provider_error_kind: Option<String>,
    pub compat_fallbacks_used: Vec<String>,
    pub candidates: usize,
    pub tokens: u64,
    pub cost_usd: f64,
    pub duration_seconds: f64,
    pub verify_output_summary: String,
    pub trajectory: Trajectory,
}

impl FailKind {
    pub fn label(self) -> &'static str {
        match self {
            FailKind::Error => "error",
            FailKind::NoEdits => "no-edits",
            FailKind::Compile => "compile",
            FailKind::Logic => "logic",
        }
    }
    /// Progress rank — when candidates fail different ways, the config's
    /// representative failure is the one that got furthest.
    pub fn rank(self) -> u8 {
        match self {
            FailKind::Error => 0,
            FailKind::NoEdits => 1,
            FailKind::Compile => 2,
            FailKind::Logic => 3,
        }
    }
}

/// Classify a failed candidate from the signals we have.
pub fn classify(passed: bool, hi_ok: bool, edited: bool, verify_output: &str) -> Option<FailKind> {
    if passed {
        return None;
    }
    if !hi_ok {
        return Some(FailKind::Error);
    }
    if !edited {
        return Some(FailKind::NoEdits);
    }
    if looks_like_build_error(verify_output) {
        Some(FailKind::Compile)
    } else {
        Some(FailKind::Logic)
    }
}

/// Heuristic: does verify output indicate the code didn't build/load (vs. a
/// behavioral test failure)? Strong, language-specific markers only, so test
/// assertions ("expected X got Y", "AssertionError") stay classified as logic.
pub fn looks_like_build_error(s: &str) -> bool {
    const MARKERS: &[&str] = &[
        "error[E",             // rustc
        "cannot find",         // rustc / go
        "cannot borrow",       // rustc
        "mismatched types",    // rustc
        "unresolved import",   // rustc
        "SyntaxError",         // python / js
        "IndentationError",    // python
        "NameError",           // python
        "ImportError",         // python
        "ModuleNotFoundError", // python
        "Cannot find name",    // ts
        "Cannot find module",  // ts / js
        "is not defined",      // js
        "undefined:",          // go
        "undefined reference", // c/c++ link
        "cannot use",          // go type error
        "compilation failed",
        "build failed",
    ];
    MARKERS.iter().any(|m| s.contains(m))
}

#[cfg(test)]
mod tests {
    use super::{FailKind, classify, looks_like_build_error};

    #[test]
    fn classify_covers_each_mode() {
        // Passed → no failure.
        assert_eq!(classify(true, true, true, ""), None);
        // hi crashed → error, regardless of edits.
        assert_eq!(classify(false, false, false, ""), Some(FailKind::Error));
        // Ran fine but changed nothing → no-edits.
        assert_eq!(classify(false, true, false, ""), Some(FailKind::NoEdits));
        // Edited but doesn't compile → compile.
        assert_eq!(
            classify(false, true, true, "error[E0382]: borrow of moved value"),
            Some(FailKind::Compile)
        );
        // Edited, compiles, wrong behavior → logic.
        assert_eq!(
            classify(false, true, true, "assertion failed: expected 4 got 5"),
            Some(FailKind::Logic)
        );
    }

    #[test]
    fn build_errors_vs_assertions() {
        assert!(looks_like_build_error("error[E0599]: no method named foo"));
        assert!(looks_like_build_error(
            "Traceback... ModuleNotFoundError: no module"
        ));
        assert!(looks_like_build_error("x.ts: Cannot find name 'foo'"));
        // Behavioral failures must NOT look like build errors.
        assert!(!looks_like_build_error(
            "test result: FAILED. 1 passed; 1 failed"
        ));
        assert!(!looks_like_build_error("AssertionError: expected 4, got 5"));
    }
}
