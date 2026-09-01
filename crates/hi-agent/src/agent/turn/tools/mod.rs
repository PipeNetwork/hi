//! Tool-batch execution (TurnPhase::Tools).
//!
//! - [`batch`] — dep-aware scheduler + execute path for one model round

mod batch;

pub(in crate::agent::turn) use batch::ToolBatchOutcome;
