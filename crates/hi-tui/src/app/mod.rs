//! `impl App` methods split by responsibility. Each submodule re-opens
//! `impl crate::App` for its group of methods; the `App` struct definition,
//! the run loop, and the entry point stay in `lib.rs`.

mod commands;
mod completion;
mod lifecycle;
mod models;
mod render;
mod run;
mod sync_commands;
mod transcript;

pub use run::run;
#[cfg(test)]
pub(crate) use run::review_next_hunk;
