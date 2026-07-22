//! Windows system sleep/wake via `PowerRegisterSuspendResumeNotification`.
//!
//! Uses the Win32 power API with a `DEVICE_NOTIFY_CALLBACK` to receive
//! suspend/resume events. This is a stub that returns `None` for now.
//!
//! Full implementation would use `windows-sys` Win32_System_Power.

use super::{PowerCallback, PowerState};

pub(crate) struct Listener;

impl Listener {
    pub(crate) fn start(_callback: PowerCallback) -> Option<Self> {
        // TODO: implement via PowerRegisterSuspendResumeNotification.
        None
    }
}

pub(crate) fn current_power_state() -> PowerState {
    PowerState::Unknown
}
