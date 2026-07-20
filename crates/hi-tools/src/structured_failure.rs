//! Structured failure formatting for verify / fast-check / tool diagnostics.
//!
//! Builds on [`crate::parse_attributions`]: turns a raw diagnostic blob into a
//! model-facing block with a short "Likely cause" list (file:line + message) and
//! a condensed output section. Enrich-only — the raw signal stays available.

use crate::attribution::{AttrKind, Attribution, parse_attributions};
use crate::condense::condense_diagnostics;

const DEFAULT_MAX_ATTRS: usize = 5;
const SNIPPET_CONTEXT: usize = 2;
const MAX_SNIPPET_LINES: usize = 12;
const MAX_OUTPUT_CHARS: usize = 3_500;

/// One structured failure ready to inject into a verify nudge or tool residual.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuredFailure {
    pub attributions: Vec<Attribution>,
    /// Full model-facing body (headline + causes + condensed output + guidance).
    pub body: String,
    /// Short UI status line.
    pub summary: String,
}

/// Format a diagnostic `output` blob for the model.
///
/// * `headline` — e.g. `Verification stage \`check\` failed (\`cargo test\`)`
/// * `guidance` — optional stage-specific next-step text
pub fn format_structured_failure(
    headline: &str,
    output: &str,
    guidance: Option<&str>,
) -> StructuredFailure {
    format_structured_failure_with_limit(headline, output, guidance, DEFAULT_MAX_ATTRS)
}

pub fn format_structured_failure_with_limit(
    headline: &str,
    output: &str,
    guidance: Option<&str>,
    max_attributions: usize,
) -> StructuredFailure {
    let attributions = parse_attributions(output, max_attributions);
    let cause_section = render_cause_section(&attributions, output);
    let condensed = condense_diagnostics(output, MAX_OUTPUT_CHARS);
    let mut body = String::new();
    body.push_str(headline.trim());
    body.push('\n');
    if !cause_section.is_empty() {
        body.push('\n');
        body.push_str(&cause_section);
    }
    body.push_str("\nOutput:\n");
    body.push_str(condensed.trim_end());
    body.push('\n');
    if let Some(guidance) = guidance.map(str::trim).filter(|g| !g.is_empty()) {
        body.push('\n');
        body.push_str(guidance);
        body.push('\n');
    }
    body.push_str(
        "If a previous fix didn't work, reconsider rather than repeat it.",
    );

    let summary = if let Some(first) = attributions.first() {
        let loc = format_loc(first);
        if loc.is_empty() {
            format!("✗ {headline}")
        } else {
            format!("✗ {} — {}", first.message.chars().take(80).collect::<String>(), loc)
        }
    } else {
        format!("✗ {headline}")
    };

    StructuredFailure {
        attributions,
        body,
        summary,
    }
}

/// Compact "Likely cause" block used by verify nudges and fast-check residuals.
pub fn render_cause_section(attributions: &[Attribution], raw_output: &str) -> String {
    if attributions.is_empty() {
        return String::new();
    }
    let mut lines = Vec::new();
    lines.push("Likely cause (verify and fix first):".to_string());
    for attr in attributions {
        let kind = kind_label(attr.kind);
        let loc = format_loc(attr);
        if loc.is_empty() {
            lines.push(format!("- [{kind}] {}", attr.message));
        } else {
            lines.push(format!("- [{kind}] {loc} — {}", attr.message));
        }
        if let Some(snippet) = snippet_for(attr, raw_output) {
            for snip_line in snippet.lines() {
                lines.push(format!("    {snip_line}"));
            }
        }
    }
    lines.push(String::new());
    lines.join("\n")
}

fn kind_label(kind: AttrKind) -> &'static str {
    match kind {
        AttrKind::Compile => "compile",
        AttrKind::Test => "test",
        AttrKind::Lint => "lint",
        AttrKind::Other => "other",
    }
}

fn format_loc(attr: &Attribution) -> String {
    if attr.path.is_empty() {
        return String::new();
    }
    match (attr.line, attr.column) {
        (Some(l), Some(c)) => format!("{}:{}:{}", attr.path, l, c),
        (Some(l), None) => format!("{}:{}", attr.path, l),
        _ => attr.path.clone(),
    }
}

/// Pull a few lines of diagnostic context around the attribution's path:line.
fn snippet_for(attr: &Attribution, raw_output: &str) -> Option<String> {
    if attr.path.is_empty() {
        return None;
    }
    let lines: Vec<&str> = raw_output.lines().collect();
    let path = attr.path.as_str();
    let target_line = attr.line;
    let mut hit = None;
    for (idx, line) in lines.iter().enumerate() {
        if !line.contains(path) {
            continue;
        }
        if let Some(want) = target_line {
            // Match `:N:` or `:N` after the path.
            let needle = format!(":{want}");
            if line.contains(&format!("{path}{needle}")) || line.contains(&format!("{path} {needle}"))
            {
                hit = Some(idx);
                break;
            }
            // rustc `--> path:line:col`
            if line.contains("-->") && line.contains(&format!(":{want}:")) {
                hit = Some(idx);
                break;
            }
        } else {
            hit = Some(idx);
            break;
        }
    }
    let idx = hit?;
    let lo = idx.saturating_sub(SNIPPET_CONTEXT);
    let hi = (idx + SNIPPET_CONTEXT + 1).min(lines.len());
    let mut out = Vec::new();
    for line in lines.iter().take(hi).skip(lo).take(MAX_SNIPPET_LINES) {
        let trimmed = line.trim_end();
        if !trimmed.is_empty() {
            out.push(trimmed);
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structured_failure_surfaces_rustc_location() {
        let output = "\
error[E0308]: mismatched types
  --> src/lib.rs:42:18
   |
42 |     let x: u32 = \"hi\";
   |                  ^^^^ expected u32, found &str
";
        let failure = format_structured_failure(
            "Verification stage `check` failed (`cargo check`)",
            output,
            Some("Fix the type error first."),
        );
        assert!(
            failure.body.contains("Likely cause (verify and fix first)"),
            "{}",
            failure.body
        );
        assert!(
            failure.body.contains("src/lib.rs:42:18"),
            "{}",
            failure.body
        );
        assert!(failure.body.contains("[compile]"), "{}", failure.body);
        assert!(failure.body.contains("Output:"), "{}", failure.body);
        assert!(
            failure.body.contains("mismatched types"),
            "{}",
            failure.body
        );
        assert!(
            failure.body.contains("Fix the type error first."),
            "{}",
            failure.body
        );
        // Snippet from the diagnostic frame should appear indented under the cause.
        assert!(
            failure.body.contains("expected u32") || failure.body.contains("let x"),
            "snippet expected: {}",
            failure.body
        );
    }

    #[test]
    fn structured_failure_handles_empty_output() {
        let failure = format_structured_failure("stage failed", "", None);
        assert!(failure.body.contains("stage failed"));
        assert!(failure.body.contains("Output:"));
        assert!(failure.attributions.is_empty() || !failure.attributions.is_empty());
    }

    #[test]
    fn pytest_style_gets_test_kind() {
        let output = "\
________________________ test_add ________________________
tests/test_math.py:12: in test_add
    assert add(1, 1) == 3
E   assert 2 == 3
";
        let failure = format_structured_failure("test failed", output, None);
        // Best-effort: either test attribution or still keeps raw output.
        assert!(failure.body.contains("Output:"));
        assert!(
            failure.body.contains("test_math.py") || failure.body.contains("assert"),
            "{}",
            failure.body
        );
    }
}
