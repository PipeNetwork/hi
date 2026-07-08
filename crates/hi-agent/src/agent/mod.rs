//! `impl Agent` methods split by responsibility. Each submodule re-opens
//! `impl super::Agent` for its group of methods; the struct definition and
//! the orchestration entry points stay in `lib.rs`.

mod compaction_turn;
mod curate_turn;
mod goal_turn;
mod lifecycle;
mod memory_turn;
mod preflight;
mod turn;

pub(crate) use curate_turn::MAX_AUTO_SKILLS_PER_SESSION;
