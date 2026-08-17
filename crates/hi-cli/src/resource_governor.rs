//! Cross-process capacity governor for expensive child and verifier processes.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};

const MAX_CONCURRENCY: usize = 16;
const DEFAULT_MIN_AVAILABLE_MEMORY_BYTES: u64 = 512 * 1024 * 1024;
const OVERLOAD_WINDOW: Duration = Duration::from_secs(10);
const RECOVERY_STEP: Duration = Duration::from_secs(5);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResourceClass {
    Setup,
    Model,
    Verifier,
    Merge,
}

impl ResourceClass {
    fn as_str(self) -> &'static str {
        match self {
            Self::Setup => "setup",
            Self::Model => "model",
            Self::Verifier => "verifier",
            Self::Merge => "merge",
        }
    }
}

pub(crate) struct ResourceLease {
    path: PathBuf,
    _file: File,
}

fn report_wait(class: ResourceClass, started: Instant, reason: &str) {
    if started.elapsed() >= Duration::from_secs(2) {
        eprintln!(
            "waiting for {} capacity ({reason}, {}ms elapsed)",
            class.as_str(),
            started.elapsed().as_millis()
        );
    }
}

impl Drop for ResourceLease {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

pub(crate) fn effective_capacity() -> usize {
    std::env::var("HI_GLOBAL_PROCESS_CONCURRENCY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or_else(|| crate::scheduler_ops::SchedulerPreset::from_env().process_capacity())
        .clamp(1, MAX_CONCURRENCY)
}

fn capacity_for(class: ResourceClass) -> usize {
    if class == ResourceClass::Merge {
        1
    } else {
        effective_capacity()
    }
}

fn aggregate_capacity() -> usize {
    std::env::var("HI_GLOBAL_AGGREGATE_PROCESS_CONCURRENCY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(effective_capacity)
        .clamp(1, MAX_CONCURRENCY)
}

fn adaptive_penalty(state_root: &Path, class: ResourceClass) -> usize {
    let path = state_root
        .join("resource-leases")
        .join(format!("{}-penalty", class.as_str()));
    let Ok(text) = std::fs::read_to_string(path) else {
        return 0;
    };
    let mut fields = text.split_whitespace();
    let Some(recorded_ms) = fields.next().and_then(|value| value.parse::<u128>().ok()) else {
        return 0;
    };
    let penalty = fields
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    adaptive_penalty_at(recorded_ms, penalty, unix_millis())
}

fn adaptive_penalty_at(recorded_ms: u128, penalty: usize, now_ms: u128) -> usize {
    let elapsed = now_ms.saturating_sub(recorded_ms);
    let hold_ms = OVERLOAD_WINDOW.as_millis();
    if elapsed < hold_ms {
        return penalty;
    }
    let recovered = ((elapsed - hold_ms) / RECOVERY_STEP.as_millis()) as usize + 1;
    penalty.saturating_sub(recovered)
}

fn unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

pub(crate) fn record_overload(state_root: &Path, class: ResourceClass) -> Result<()> {
    let lease_root = state_root.join("resource-leases");
    std::fs::create_dir_all(&lease_root)?;
    let recorded_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let penalty = capacity_for(class).div_ceil(2);
    std::fs::write(
        lease_root.join(format!("{}-penalty", class.as_str())),
        format!("{recorded_ms} {penalty}"),
    )?;
    Ok(())
}

fn available_memory_bytes() -> Option<u64> {
    available_memory_from_proc().or_else(available_memory_from_platform)
}

fn available_memory_from_proc() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kib = text.lines().find_map(|line| {
        line.strip_prefix("MemAvailable:")?
            .split_whitespace()
            .next()?
            .parse::<u64>()
            .ok()
    })?;
    Some(kib.saturating_mul(1024))
}

#[cfg(target_os = "macos")]
fn available_memory_from_platform() -> Option<u64> {
    let page_size = command_u64("sysctl", &["-n", "hw.pagesize"])?;
    let output = std::process::Command::new("vm_stat").output().ok()?;
    output.status.success().then_some(())?;
    let text = String::from_utf8(output.stdout).ok()?;
    let pages = [
        "Pages free",
        "Pages inactive",
        "Pages speculative",
        "Pages purgeable",
    ]
    .iter()
    .filter_map(|name| vm_stat_pages(&text, name))
    .fold(0_u64, u64::saturating_add);
    Some(pages.saturating_mul(page_size))
}

#[cfg(target_os = "macos")]
fn command_u64(command: &str, args: &[&str]) -> Option<u64> {
    let output = std::process::Command::new(command)
        .args(args)
        .output()
        .ok()?;
    output.status.success().then_some(())?;
    String::from_utf8(output.stdout).ok()?.trim().parse().ok()
}

#[cfg(target_os = "macos")]
fn vm_stat_pages(text: &str, name: &str) -> Option<u64> {
    text.lines().find_map(|line| {
        line.strip_prefix(name)?
            .trim_start_matches([' ', ':'])
            .trim_end_matches('.')
            .parse()
            .ok()
    })
}

#[cfg(target_os = "windows")]
fn available_memory_from_platform() -> Option<u64> {
    let output = std::process::Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory",
        ])
        .output()
        .ok()?;
    output.status.success().then_some(())?;
    String::from_utf8(output.stdout)
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(|kib| kib.saturating_mul(1024))
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
fn available_memory_from_platform() -> Option<u64> {
    None
}

fn memory_budget_allows(class: ResourceClass) -> bool {
    if class == ResourceClass::Merge {
        return true;
    }
    let minimum = std::env::var("HI_MIN_AVAILABLE_MEMORY_BYTES")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(DEFAULT_MIN_AVAILABLE_MEMORY_BYTES);
    available_memory_bytes().is_none_or(|available| available >= minimum)
}

pub(crate) fn acquire(
    state_root: &Path,
    class: ResourceClass,
    timeout: Duration,
) -> Result<ResourceLease> {
    acquire_while(state_root, class, timeout, &|| false)
}

pub(crate) fn acquire_while(
    state_root: &Path,
    class: ResourceClass,
    timeout: Duration,
    stop: &dyn Fn() -> bool,
) -> Result<ResourceLease> {
    let configured = capacity_for(class).min(aggregate_capacity());
    let limit = configured
        .saturating_sub(adaptive_penalty(state_root, class))
        .max(1);
    let lease_root = state_root.join("resource-leases");
    std::fs::create_dir_all(&lease_root)
        .with_context(|| format!("creating resource lease directory {}", lease_root.display()))?;
    let queue_path = create_waiter(&lease_root, class)?;
    let mut waiter = WaiterGuard(queue_path);
    let started = Instant::now();
    loop {
        if stop() {
            bail!("cancelled waiting for shared {} capacity", class.as_str());
        }
        if !waiter_is_eligible(&lease_root, class, &waiter.0, limit)? {
            report_wait(class, started, "queued behind earlier work");
            if started.elapsed() >= timeout {
                bail!("timed out waiting for shared {} capacity", class.as_str());
            }
            std::thread::sleep(Duration::from_millis(50));
            continue;
        }
        if !memory_budget_allows(class) {
            report_wait(class, started, "insufficient available memory");
            if started.elapsed() >= timeout {
                bail!(
                    "timed out waiting for memory capacity for {}",
                    class.as_str()
                );
            }
            std::thread::sleep(Duration::from_millis(100));
            continue;
        }
        for slot in 0..limit {
            let path = lease_root.join(format!("{}-slot-{slot}.lease", class.as_str()));
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut file) => {
                    writeln!(file, "{} {}", std::process::id(), class.as_str())?;
                    std::fs::remove_file(&waiter.0)?;
                    waiter.0 = PathBuf::new();
                    return Ok(ResourceLease { path, _file: file });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    if lease_is_stale(&path) {
                        let _ = std::fs::remove_file(&path);
                    }
                }
                Err(error) => return Err(error).context("acquiring process capacity"),
            }
        }
        if started.elapsed() >= timeout {
            bail!("timed out waiting for shared {} capacity", class.as_str());
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

struct WaiterGuard(PathBuf);

impl Drop for WaiterGuard {
    fn drop(&mut self) {
        if !self.0.as_os_str().is_empty() {
            let _ = std::fs::remove_file(&self.0);
        }
    }
}

fn create_waiter(lease_root: &Path, class: ResourceClass) -> Result<PathBuf> {
    for nonce in 0_u32.. {
        let path = lease_root.join(format!(
            "{}-wait-{:020}-{}-{nonce}.queue",
            class.as_str(),
            unix_millis(),
            std::process::id()
        ));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut file) => {
                writeln!(file, "{}", std::process::id())?;
                return Ok(path);
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error).context("joining process capacity queue"),
        }
    }
    unreachable!()
}

fn waiter_is_eligible(
    lease_root: &Path,
    class: ResourceClass,
    own: &Path,
    available_slots: usize,
) -> Result<bool> {
    let prefix = format!("{}-wait-", class.as_str());
    let mut waiters = Vec::new();
    for entry in std::fs::read_dir(lease_root)? {
        let path = entry?.path();
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with(&prefix) && name.ends_with(".queue"))
        {
            if lease_is_stale(&path) {
                let _ = std::fs::remove_file(path);
            } else {
                waiters.push(path);
            }
        }
    }
    waiters.sort();
    Ok(waiters
        .iter()
        .take(available_slots.max(1))
        .any(|path| path == own))
}

fn lease_is_stale(path: &Path) -> bool {
    let Ok(text) = std::fs::read_to_string(path) else {
        return false;
    };
    let Some(pid) = text
        .split_whitespace()
        .next()
        .and_then(|value| value.parse::<u32>().ok())
    else {
        return false;
    };
    !pid_is_alive(pid)
}

#[cfg(unix)]
fn pid_is_alive(pid: u32) -> bool {
    std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .status()
        .is_ok_and(|status| status.success())
}

#[cfg(not(unix))]
fn pid_is_alive(_pid: u32) -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_capacity_is_bounded() {
        let capacity = effective_capacity();
        assert!((1..=MAX_CONCURRENCY).contains(&capacity));
    }

    #[test]
    fn memory_admission_is_fail_open_without_platform_data() {
        assert!(memory_budget_allows(ResourceClass::Merge));
    }

    #[test]
    fn overload_penalty_reduces_but_never_exhausts_capacity() {
        let root =
            std::env::temp_dir().join(format!("hi-governor-overload-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        record_overload(&root, ResourceClass::Model).unwrap();
        let penalty = adaptive_penalty(&root, ResourceClass::Model);
        assert!(penalty >= 1);
        assert!(
            capacity_for(ResourceClass::Model)
                .saturating_sub(penalty)
                .max(1)
                >= 1
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn overload_penalty_recovers_one_slot_at_a_time() {
        let hold = OVERLOAD_WINDOW.as_millis();
        let step = RECOVERY_STEP.as_millis();
        assert_eq!(adaptive_penalty_at(1_000, 3, 1_000 + hold - 1), 3);
        assert_eq!(adaptive_penalty_at(1_000, 3, 1_000 + hold), 2);
        assert_eq!(adaptive_penalty_at(1_000, 3, 1_000 + hold + step), 1);
        assert_eq!(adaptive_penalty_at(1_000, 3, 1_000 + hold + 2 * step), 0);
    }

    #[test]
    fn queue_order_is_fifo_across_processes() {
        let root = std::env::temp_dir().join(format!(
            "hi-governor-fifo-{}-{}",
            std::process::id(),
            unix_millis()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let later = root.join(format!(
            "{}-wait-{:020}-2-0.queue",
            ResourceClass::Verifier.as_str(),
            2
        ));
        let earlier = root.join(format!(
            "{}-wait-{:020}-1-0.queue",
            ResourceClass::Verifier.as_str(),
            1
        ));
        std::fs::write(&later, format!("{}\n", std::process::id())).unwrap();
        std::fs::write(&earlier, format!("{}\n", std::process::id())).unwrap();
        assert!(waiter_is_eligible(&root, ResourceClass::Verifier, &earlier, 1).unwrap());
        assert!(!waiter_is_eligible(&root, ResourceClass::Verifier, &later, 1).unwrap());
        assert!(waiter_is_eligible(&root, ResourceClass::Verifier, &later, 2).unwrap());
        std::fs::remove_file(earlier).unwrap();
        assert!(waiter_is_eligible(&root, ResourceClass::Verifier, &later, 1).unwrap());
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn parses_vm_stat_pages() {
        let text = "Mach Virtual Memory Statistics: (page size of 4096 bytes)\nPages free: 123.\nPages inactive: 456.\n";
        assert_eq!(vm_stat_pages(text, "Pages free"), Some(123));
        assert_eq!(vm_stat_pages(text, "Pages inactive"), Some(456));
    }

    #[test]
    fn merge_is_single_writer_and_classes_are_independent() {
        assert_eq!(capacity_for(ResourceClass::Merge), 1);
        assert!(capacity_for(ResourceClass::Setup) >= 1);
        assert_ne!(ResourceClass::Setup.as_str(), ResourceClass::Model.as_str());
        assert_ne!(
            ResourceClass::Verifier.as_str(),
            ResourceClass::Merge.as_str()
        );
    }
}
