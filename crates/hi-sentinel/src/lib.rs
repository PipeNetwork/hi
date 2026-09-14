//! Hi Sentinel — interactive process supervisor for `hi`.
//!
//! This crate is not the RSI candidate launcher (`hi-bootstrap` /
//! `rsi-hi-worker`) and is not clap's "unlimited internal sentinel" cap
//! (`parse_finite_u32_cap`). It only observes a child, classifies failures,
//! and writes forensic bundles. Repair/apply live elsewhere.

mod args;
mod budget;
mod checkout;
mod classify;
mod config;
mod fsutil;
mod history;
mod incident;
mod monitor;
mod paths;
mod spawn;
mod supervise;

pub use classify::{BugKind, Class, ClassifyContext, Confidence, ExternalKind, ReportKind};
pub use config::{MonitorConfig, SupervisorConfig};
pub use supervise::{SupervisorOutcome, run, supervise};

pub const ENV_BINARY: &str = "HI_SENTINEL_BINARY";
pub const ENV_SESSION_TOKEN: &str = "HI_SENTINEL_SESSION_TOKEN";
pub const ENV_KNOWN_GOOD: &str = "HI_SENTINEL_KNOWN_GOOD";
pub const ENV_STATE_DIR: &str = "HI_SENTINEL_STATE_DIR";
pub const ENV_TEE: &str = "HI_SENTINEL_TEE";
pub const ENV_CHECKOUT: &str = "HI_CHECKOUT";

#[cfg(test)]
mod fake_child_tests;
