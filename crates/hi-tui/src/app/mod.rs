//! `impl App` methods split by responsibility. Each submodule re-opens
//! `impl crate::App` for its group of methods; the `App` struct definition,
//! the run loop, and the entry point stay in `lib.rs`.

mod command_helpers;
mod commands;
mod completion;
mod composer;
mod lifecycle;
mod models;
mod render;
mod run;
pub(crate) mod session_projection;
pub(crate) mod sync_commands;
mod transcript;
pub(crate) mod voice;

pub use run::run;
pub(crate) use run::{handle_normal_mode, review_next_hunk};
#[cfg(test)]
pub(crate) use run::{search_transcript, send_now_queued_follow_up};
pub(crate) use sync_commands::SteeringRemote;
