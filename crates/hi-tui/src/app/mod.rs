//! `impl App` methods split by responsibility. Each submodule re-opens
//! `impl crate::App` for its group of methods; the `App` struct definition
//! and the session entry point stay in `lib.rs`.

mod commands;
mod completion;
mod composer;
mod lifecycle;
mod render;
mod run;
pub(crate) mod session_projection;
mod transcript;
pub(crate) mod voice;

pub(crate) use run::review_next_hunk;
pub use run::{SessionOptions, run_session};
