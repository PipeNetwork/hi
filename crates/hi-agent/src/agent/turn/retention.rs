//! Turn-local bounded diagnostic trails.

use crate::diagnostic_retention::BoundedDiagnosticLog;

/// Retain both the first actions (intent/routing evidence) and the latest
/// actions (terminal diagnosis). Aggregate counters remain authoritative for
/// the full unlimited turn.
pub(super) const TOOL_TIMELINE_LIMIT: usize = 512;
const TOOL_TIMELINE_HEAD: usize = 64;

pub(in crate::agent) type ToolTimeline =
    BoundedDiagnosticLog<crate::ToolCallEntry, TOOL_TIMELINE_LIMIT, TOOL_TIMELINE_HEAD>;

pub(super) const PROGRESS_EVENT_LIMIT: usize = 256;
pub(super) const PROGRESS_EVENT_HEAD: usize = 32;

pub(super) type ProgressEventLog =
    BoundedDiagnosticLog<crate::ProgressEvent, PROGRESS_EVENT_LIMIT, PROGRESS_EVENT_HEAD>;

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_entry(index: u32) -> crate::ToolCallEntry {
        crate::ToolCallEntry {
            tool: format!("tool-{index}"),
            path: String::new(),
            duration_ms: 0,
            queue_delay_ms: 0,
            completion_index: index,
            status: hi_tools::ToolStatus::Succeeded,
            background: None,
            process: None,
            effects: hi_tools::ToolEffects::default(),
            truncation: hi_tools::TruncationState::Complete,
            error: false,
            progress_kind: "weak".into(),
            progress_reason: "tool completed".into(),
            normalized_signature: None,
            command: None,
            arg_chars: 0,
            result_chars: 0,
            truncated: false,
            kind: "other".into(),
        }
    }

    #[test]
    fn tool_timeline_stays_bounded_across_an_unlimited_turn() {
        let mut timeline = ToolTimeline::default();
        for index in 0..700 {
            timeline.push(tool_entry(index));
        }

        assert_eq!(timeline.len(), TOOL_TIMELINE_LIMIT);
        assert_eq!(timeline.dropped(), 700 - TOOL_TIMELINE_LIMIT as u64);
        assert_eq!(timeline.first().unwrap().tool, "tool-0");
        assert_eq!(timeline.last().unwrap().tool, "tool-699");
    }
}
