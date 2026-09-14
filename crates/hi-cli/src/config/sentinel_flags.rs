use std::path::PathBuf;

use clap::Args;

/// Sentinel flags, flattened onto `Cli` so `cli.rs` stays under the line ratchet.
#[derive(Args, Clone, Debug, Default)]
pub struct SentinelFlags {
    /// Wrap this process in hi-sentinel (crash/stall/invariant repair).
    #[arg(long, conflicts_with_all = ["rsi_managed", "rsi", "subagent"])]
    pub autoharnessfix: bool,

    /// Disable Sentinel even if machine config enabled it.
    #[arg(long, conflicts_with = "autoharnessfix")]
    pub no_autoharnessfix: bool,

    /// After a verified repair, apply without prompting (still never pushes).
    #[arg(long)]
    pub autoharnessfix_apply: bool,

    /// Git checkout of Hi used for repair worktrees.
    #[arg(long, value_name = "PATH")]
    pub autoharnessfix_checkout: Option<PathBuf>,

    /// Hidden: restore a pre-turn checkpoint. Requires HI_SENTINEL_ROLE=restore.
    #[arg(long, hide = true, value_name = "ID")]
    pub sentinel_restore_checkpoint: Option<String>,
}
