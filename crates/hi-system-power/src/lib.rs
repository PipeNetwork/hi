//! Cross-platform system **sleep/wake** (suspend/resume) notifications.
//!
//! The motivating use case: an OIDC token refresh that is *in flight when the
//! laptop sleeps* can lose its rotated successor token (the server processes
//! the request, rotates/revokes the old refresh token, and the response is
//! lost across the suspend). On wake the client is holding a dead refresh
//! token and the user is forced to re-login. Consumers can use these events to
//! avoid *starting* a refresh just before sleep.
//!
//! This crate exposes a single tiny abstraction — [`SystemPowerListener`] —
//! with per-OS implementations behind `#[cfg]` and a no-op fallback:
//!
//! | OS      | Mechanism                                                            |
//! |---------|----------------------------------------------------------------------|
//! | macOS   | IOKit `IORegisterForSystemPower` on a dedicated `CFRunLoop` thread    |
//! | Windows | `PowerRegisterSuspendResumeNotification` (`DEVICE_NOTIFY_CALLBACK`)   |
//! | Linux   | logind D-Bus `PrepareForSleep` signal + a `delay` inhibitor lock      |
//! | other   | no-op (returns `None` from [`SystemPowerListener::start`])            |
//!
//! Inspired by grok-build's `xai-system-power` crate.

/// A system power transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerEvent {
    /// The system is about to sleep (lid close / suspend). There is a short
    /// window to react before sleep proceeds.
    WillSleep,
    /// The system has woken from sleep.
    DidWake,
}

/// Boxed user callback invoked on each [`PowerEvent`].
pub type PowerCallback = Box<dyn Fn(PowerEvent) + Send + Sync + 'static>;

/// A coarse, synchronously-queryable system power state.
///
/// The motivating distinction is **dark wake**: on macOS the system wakes
/// briefly for background/maintenance work with the display off and no user
/// present, then re-sleeps — frequently *without* delivering a
/// [`PowerEvent`] at all. Code that starts irreversible network work should
/// avoid doing so during a dark wake.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerState {
    /// Full / user wake: display (graphics) capability present.
    FullWake,
    /// Dark wake: CPU up for background work, display off.
    DarkWake,
    /// State could not be determined.
    Unknown,
}

/// Query the current system power state synchronously.
///
/// Cheap, non-blocking, and never panics. Returns [`PowerState::Unknown`] on
/// platforms without a real implementation or when the platform query fails.
pub fn current_power_state() -> PowerState {
    imp::current_power_state()
}

#[cfg(target_os = "macos")]
#[path = "macos.rs"]
mod imp;

#[cfg(target_os = "windows")]
#[path = "windows.rs"]
mod imp;

#[cfg(target_os = "linux")]
#[path = "linux.rs"]
mod imp;

#[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
mod imp {
    use super::PowerCallback;

    pub(crate) struct Listener;

    impl Listener {
        pub(crate) fn start(_callback: PowerCallback) -> Option<Self> {
            None
        }
    }

    pub(crate) fn current_power_state() -> super::PowerState {
        super::PowerState::Unknown
    }
}

/// A running system-power listener. On macOS/Windows, dropping it stops the
/// listener and releases its OS resources. On Linux the worker parks on a
/// blocking logind signal and cannot be cleanly interrupted, so it runs until
/// process exit. Intended as a process-lifetime singleton.
pub struct SystemPowerListener {
    #[allow(dead_code)]
    inner: imp::Listener,
}

impl SystemPowerListener {
    /// Start listening for system sleep/wake events.
    ///
    /// Returns `None` when the platform mechanism is unavailable. Callers
    /// should treat `None` as "no power notifications" and degrade gracefully.
    ///
    /// `callback` is invoked from a platform event thread, so it must be
    /// `Send + Sync`, cheap, and non-blocking.
    pub fn start<F>(callback: F) -> Option<Self>
    where
        F: Fn(PowerEvent) + Send + Sync + 'static,
    {
        imp::Listener::start(Box::new(callback)).map(|inner| Self { inner })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_event_is_copy_eq() {
        let e = PowerEvent::WillSleep;
        let copied = e;
        assert_eq!(e, copied);
        assert_ne!(PowerEvent::WillSleep, PowerEvent::DidWake);
    }

    /// `start` + `drop` must be clean on every platform: no panic and no hang.
    #[test]
    fn start_and_drop_is_clean() {
        let _listener = SystemPowerListener::start(|_event| {});
    }

    #[test]
    fn current_power_state_doesnt_panic() {
        let _state = current_power_state();
    }
}
