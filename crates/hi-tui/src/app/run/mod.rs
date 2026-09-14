//! Interactive session loop backed by `hi-harness` (Pipe Network).

mod hydrate;
mod idle;
mod session;

pub use session::{SessionOptions, run_session};

/// Find the painted line index of the next (dir=1) or previous (dir=-1) hunk
/// in `diff`. Hunks are grok-build inline rows separated by `…` gaps, not raw
/// `@@` headers. Clamps to the painted bounds; returns `from` unchanged if
/// there's no hunk in the requested direction.
pub(crate) fn review_next_hunk(diff: Option<&str>, from: usize, dir: i32) -> usize {
    let Some(diff) = diff else { return from };
    let lines = crate::render::diff_lines(diff);
    if lines.is_empty() {
        return from;
    }
    let starts = crate::render::hunk_start_indices(&lines);
    if starts.is_empty() {
        return from.min(lines.len().saturating_sub(1));
    }
    if dir > 0 {
        starts
            .iter()
            .copied()
            .find(|&i| i > from)
            .unwrap_or(lines.len().saturating_sub(1))
    } else {
        starts
            .iter()
            .copied()
            .rev()
            .find(|&i| i < from)
            .unwrap_or(0)
    }
}
