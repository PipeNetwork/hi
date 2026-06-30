//! `impl App` methods split by responsibility. Each submodule re-opens
//! `impl crate::App` for its group of methods; the `App` struct definition,
//! the run loop, and the entry point stay in `lib.rs`.

mod lifecycle;
mod completion;
mod models;
mod transcript;
mod commands;
mod render;
mod run;

pub use run::run;

