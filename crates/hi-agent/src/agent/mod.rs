//! `impl Agent` methods split by responsibility. Each submodule re-opens
//! `impl super::Agent` for its group of methods; the struct definition and
//! the orchestration entry points stay in `lib.rs`.

mod preflight;
mod memory_turn;
mod compaction_turn;
mod goal_turn;
mod turn;
mod lifecycle;