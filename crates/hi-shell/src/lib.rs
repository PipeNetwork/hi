//! ACP frontend for the `hi-agent` coding harness.

mod server;

pub use server::{HiShell, ShellConfig, serve_stdio};
