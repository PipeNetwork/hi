//! Linux system sleep/wake via systemd-logind D-Bus `PrepareForSleep` signal.
//!
//! Uses a `delay` inhibitor lock so logind waits briefly for our callback to
//! finish before proceeding to suspend. On systems without systemd-logind,
//! `start` returns `None`.
//!
//! This is a stub that requires the `zbus` crate. For now, we return `None`
//! (no power notifications) — the feature degrades gracefully.

use super::{PowerCallback, PowerState};

pub(crate) struct Listener;

impl Listener {
    pub(crate) fn start(_callback: PowerCallback) -> Option<Self> {
        // TODO: implement via zbus logind D-Bus PrepareForSleep signal.
        None
    }
}

pub(crate) fn current_power_state() -> PowerState {
    PowerState::Unknown
}
