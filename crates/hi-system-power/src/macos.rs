//! macOS system sleep/wake via IOKit `IORegisterForSystemPower`.
//!
//! IOKit delivers power notifications through a `CFRunLoop` source, so we run a
//! dedicated thread whose run loop receives the callbacks. The thread owns all
//! IOKit resources for their full lifetime and tears them down after the run
//! loop is stopped (from `Drop`).
//!
//! FFI is declared directly (CoreFoundation + IOKit frameworks) to avoid a
//! `core-foundation` crate dependency for this tiny surface.

use std::os::raw::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;

use super::{PowerCallback, PowerEvent, PowerState};

// `io_object_t` / `io_connect_t` are `mach_port_t` == `unsigned int`.
type MachPort = u32;
const MACH_PORT_NULL: MachPort = 0;

// IOKit power-management message types (IOMessage.h).
const K_IO_MESSAGE_CAN_SYSTEM_SLEEP: u32 = 0xe000_0270;
const K_IO_MESSAGE_SYSTEM_WILL_SLEEP: u32 = 0xe000_0280;
const K_IO_MESSAGE_SYSTEM_WILL_NOT_SLEEP: u32 = 0xe000_0290;
const K_IO_MESSAGE_SYSTEM_HAS_POWERED_ON: u32 = 0xe000_0300;

// IOPM system-power capability bits (`IOPMCapabilityBits`). These constants and
// the `IOPMConnectionGetSystemCapabilities` query below are **SPI**: declared in
// the *private* `IOPMLibPrivate.h` (IOKitUser), not the public `IOPMLib.h`.
const K_IOPM_CAPABILITY_CPU: u32 = 0x1;
const K_IOPM_CAPABILITY_VIDEO: u32 = 0x2;

type IoServiceInterestCallback = extern "C" fn(
    refcon: *mut c_void,
    service: MachPort,
    message_type: u32,
    message_argument: *mut c_void,
);

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    static kCFRunLoopCommonModes: *const c_void;
    static kCFRunLoopDefaultMode: *const c_void;
    fn CFRunLoopGetCurrent() -> *mut c_void;
    fn CFRunLoopRunInMode(
        mode: *const c_void,
        seconds: f64,
        return_after_source_handled: u8,
    ) -> i32;
    fn CFRunLoopStop(rl: *mut c_void);
    fn CFRunLoopAddSource(rl: *mut c_void, source: *mut c_void, mode: *const c_void);
}

#[link(name = "IOKit", kind = "framework")]
unsafe extern "C" {
    fn IORegisterForSystemPower(
        refcon: *mut c_void,
        the_port_ref: *mut *mut c_void,
        callback: IoServiceInterestCallback,
        notifier: *mut MachPort,
    ) -> MachPort;
    fn IODeregisterForSystemPower(notifier: *mut MachPort) -> i32;
    fn IONotificationPortGetRunLoopSource(port: *mut c_void) -> *mut c_void;
    fn IONotificationPortDestroy(port: *mut c_void);
    fn IOAllowPowerChange(kern_port: MachPort, notification_id: isize) -> i32;
    fn IOServiceClose(connect: MachPort) -> i32;
    fn IOPMConnectionGetSystemCapabilities() -> u32;
}

/// Classify raw IOPM capability bits into a coarse [`PowerState`].
fn classify_capabilities(caps: u32) -> PowerState {
    if caps & K_IOPM_CAPABILITY_CPU == 0 {
        return PowerState::Unknown;
    }
    if caps & K_IOPM_CAPABILITY_VIDEO != 0 {
        PowerState::FullWake
    } else {
        PowerState::DarkWake
    }
}

pub(crate) fn current_power_state() -> PowerState {
    let caps = unsafe { IOPMConnectionGetSystemCapabilities() };
    classify_capabilities(caps)
}

struct Context {
    callback: PowerCallback,
    root_port: MachPort,
}

struct SendRunLoop(*mut c_void);
unsafe impl Send for SendRunLoop {}

pub(crate) struct Listener {
    runloop: SendRunLoop,
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl Listener {
    pub(crate) fn start(callback: PowerCallback) -> Option<Self> {
        let (tx, rx) = mpsc::channel::<Option<SendRunLoop>>();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_thread = stop.clone();
        let handle = thread::Builder::new()
            .name("hi-power-listener".into())
            .spawn(move || run_thread(callback, tx, stop_thread))
            .ok()?;

        match rx.recv() {
            Ok(Some(runloop)) => Some(Self {
                runloop,
                stop,
                handle: Some(handle),
            }),
            _ => {
                let _ = handle.join();
                None
            }
        }
    }
}

impl Drop for Listener {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        unsafe { CFRunLoopStop(self.runloop.0) };
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn run_thread(
    callback: PowerCallback,
    tx: mpsc::Sender<Option<SendRunLoop>>,
    stop: Arc<AtomicBool>,
) {
    let ctx = Box::into_raw(Box::new(Context {
        callback,
        root_port: MACH_PORT_NULL,
    }));

    let mut notifier: MachPort = MACH_PORT_NULL;
    let mut port: *mut c_void = std::ptr::null_mut();
    let root_port = unsafe {
        IORegisterForSystemPower(ctx as *mut c_void, &mut port, power_callback, &mut notifier)
    };

    if root_port == MACH_PORT_NULL || port.is_null() {
        unsafe { drop(Box::from_raw(ctx)) };
        let _ = tx.send(None);
        return;
    }
    unsafe { (*ctx).root_port = root_port };

    let runloop = unsafe { CFRunLoopGetCurrent() };
    unsafe {
        let source = IONotificationPortGetRunLoopSource(port);
        CFRunLoopAddSource(runloop, source, kCFRunLoopCommonModes);
    }

    if tx.send(Some(SendRunLoop(runloop))).is_err() {
        unsafe {
            IODeregisterForSystemPower(&mut notifier);
            IONotificationPortDestroy(port);
            IOServiceClose(root_port);
            drop(Box::from_raw(ctx));
        }
        return;
    }

    while !stop.load(Ordering::SeqCst) {
        unsafe { CFRunLoopRunInMode(kCFRunLoopDefaultMode, 5.0, 0) };
    }

    unsafe {
        IODeregisterForSystemPower(&mut notifier);
        IONotificationPortDestroy(port);
        IOServiceClose(root_port);
        drop(Box::from_raw(ctx));
    }
}

/// Pure mapping of an IOKit power message to the [`PowerEvent`] and whether
/// the message requires an `IOAllowPowerChange` acknowledgment.
fn map_power_message(message_type: u32) -> (Option<PowerEvent>, bool) {
    match message_type {
        K_IO_MESSAGE_CAN_SYSTEM_SLEEP => (Some(PowerEvent::WillSleep), true),
        K_IO_MESSAGE_SYSTEM_WILL_SLEEP => (Some(PowerEvent::WillSleep), true),
        K_IO_MESSAGE_SYSTEM_WILL_NOT_SLEEP => (Some(PowerEvent::DidWake), false),
        K_IO_MESSAGE_SYSTEM_HAS_POWERED_ON => (Some(PowerEvent::DidWake), false),
        _ => (None, false),
    }
}

extern "C" fn power_callback(
    refcon: *mut c_void,
    _service: MachPort,
    message_type: u32,
    message_argument: *mut c_void,
) {
    let ctx = unsafe { &*(refcon as *const Context) };
    let (event, needs_ack) = map_power_message(message_type);
    if let Some(event) = event {
        (ctx.callback)(event);
    }
    if needs_ack {
        unsafe { IOAllowPowerChange(ctx.root_port, message_argument as isize) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_full_wake() {
        assert_eq!(
            classify_capabilities(K_IOPM_CAPABILITY_CPU | K_IOPM_CAPABILITY_VIDEO),
            PowerState::FullWake
        );
    }

    #[test]
    fn classify_dark_wake() {
        assert_eq!(
            classify_capabilities(K_IOPM_CAPABILITY_CPU),
            PowerState::DarkWake
        );
    }

    #[test]
    fn classify_unknown_without_cpu() {
        assert_eq!(classify_capabilities(0), PowerState::Unknown);
        assert_eq!(
            classify_capabilities(K_IOPM_CAPABILITY_VIDEO),
            PowerState::Unknown
        );
    }

    #[test]
    fn map_power_message_matrix() {
        assert_eq!(
            map_power_message(K_IO_MESSAGE_CAN_SYSTEM_SLEEP),
            (Some(PowerEvent::WillSleep), true)
        );
        assert_eq!(
            map_power_message(K_IO_MESSAGE_SYSTEM_WILL_SLEEP),
            (Some(PowerEvent::WillSleep), true)
        );
        assert_eq!(
            map_power_message(K_IO_MESSAGE_SYSTEM_WILL_NOT_SLEEP),
            (Some(PowerEvent::DidWake), false)
        );
        assert_eq!(
            map_power_message(K_IO_MESSAGE_SYSTEM_HAS_POWERED_ON),
            (Some(PowerEvent::DidWake), false)
        );
        assert_eq!(map_power_message(0xe000_0320), (None, false));
    }
}
