//! Hi Sentinel — interactive process supervisor for `hi`.
//!
//! This crate is not the RSI candidate launcher (`hi-bootstrap` /
//! `rsi-hi-worker`) and is not clap's "unlimited internal sentinel" cap
//! (`parse_finite_u32_cap`). It observes a child, classifies failures, writes
//! forensic bundles, and on a classified harness bug may run a locked-down
//! repair agent in a detached worktree. Verified repairs install sidecar
//! binaries; they never move `main`.

mod apply;
mod args;
mod budget;
mod checkout;
mod classify;
mod config;
mod fsutil;
mod gate;
mod history;
mod incident;
mod ipc;
mod monitor;
mod paths;
mod repair;
mod rollback;
mod slash;
mod spawn;
mod supervise;
mod worktree;

pub use classify::{BugKind, Class, ClassifyContext, Confidence, ExternalKind, ReportKind};
pub use config::{MonitorConfig, SupervisorConfig, set_machine_enabled};
pub use slash::{
    SlashOutcome, dispatch as dispatch_slash, exec_repair, exec_supervisor, restart_line,
    status_text,
};
pub use supervise::{SupervisorOutcome, run, supervise};

#[cfg(test)]
pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub const ENV_BINARY: &str = "HI_SENTINEL_BINARY";
pub const ENV_SESSION_TOKEN: &str = "HI_SENTINEL_SESSION_TOKEN";
pub const ENV_KNOWN_GOOD: &str = "HI_SENTINEL_KNOWN_GOOD";
pub const ENV_STATE_DIR: &str = "HI_SENTINEL_STATE_DIR";
pub const ENV_TEE: &str = "HI_SENTINEL_TEE";
pub const ENV_CHECKOUT: &str = "HI_CHECKOUT";

#[cfg(test)]
mod fake_child_tests;
#[cfg(test)]
mod repair_tests;
#[cfg(test)]
mod test_fixture;
