//! CLI compatibility wrapper for the reusable control-plane approval store.

use anyhow::{Context, Result};

pub(crate) type SqliteApprovalStore = hi_control::ControlStore;

pub(crate) fn open_for_state(state_root: &std::path::Path) -> Result<SqliteApprovalStore> {
    hi_control::ControlStore::open_for_state(state_root)
        .with_context(|| format!("opening approval store under {}", state_root.display()))
}
