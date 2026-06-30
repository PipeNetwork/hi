//! Read-only review and implementation "steering": intent classification,
//! evidence/implementation trackers, preflight call planning, and the nudge
//! strings injected back into the transcript when the model's answer lacks
//! inspected evidence, concrete file citations, or post-edit validation.
//!
//! All of this is pure input classification and text generation — none of it
//! touches `Agent` state directly — so it lives outside the main `lib.rs`.

mod constants;
mod types;
mod intent;
mod preflight;
mod implementation;
mod nudges;

pub(crate) use constants::*;
pub(crate) use types::*;
pub(crate) use intent::*;
pub(crate) use preflight::*;
pub(crate) use implementation::*;
pub(crate) use nudges::*;