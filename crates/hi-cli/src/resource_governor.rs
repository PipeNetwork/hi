//! Cross-process capacity governor for expensive child and verifier processes.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};

const MAX_CONCURRENCY: usize = 16;
const DEFAULT_MIN_AVAILABLE_MEMORY_BYTES: u64 = 512 * 1024 * 1024;
const OVERLOAD_WINDOW: Duration = Duration::from_secs(10);
const RECOVERY_STEP: Duration = Duration::from_secs(5);
/// `create_new` wins ownership before the PID record is flushed. Give that
/// tiny critical section room, then recover a crash/write failure that left an
/// empty or malformed admission record instead of wedging unlimited waiters.
const INCOMPLETE_LEASE_GRACE: Duration = Duration::from_secs(60);
static CURRENT_PROCESS_BIRTH: OnceLock<Option<String>> = OnceLock::new();

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

/// The queue record is the caller's admission token. Losing the pathname or
/// seeing it replaced means this process can no longer prove its place in the
/// FIFO queue. This is an infrastructure error, not ordinary contention: with
/// an unlimited wait, treating it as `not eligible yet` would spin forever.
#[derive(Debug)]
pub(crate) struct WaiterOwnershipLost {
    class: ResourceClass,
    path: PathBuf,
}

impl std::fmt::Display for WaiterOwnershipLost {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "lost ownership of shared {} capacity waiter {}; retry acquisition",
            self.class.as_str(),
            self.path.display()
        )
    }
}

impl std::error::Error for WaiterOwnershipLost {}

#[derive(Debug)]
pub(crate) struct ResourceLease {
    path: PathBuf,
    owner: String,
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
        remove_owned_record(&self.path, &self.owner, &self._file);
    }
}

fn owner_token() -> String {
    uuid::Uuid::new_v4().to_string()
}

fn owner_record(owner: &str, class: ResourceClass, kind: &str) -> String {
    let birth = current_process_birth_identity().unwrap_or("unknown");
    format!(
        "owner={owner}\npid={}\nbirth={birth}\nclass={}\nkind={kind}\n",
        std::process::id(),
        class.as_str()
    )
}

fn record_owner(text: &str) -> Option<&str> {
    text.lines()
        .find_map(|line| line.strip_prefix("owner="))
        .map(str::trim)
        .filter(|owner| !owner.is_empty())
}

fn path_record_owner(path: &Path) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    record_owner(&text).map(str::to_owned)
}

fn open_record_text(file: &File) -> Option<String> {
    let mut file = file.try_clone().ok()?;
    file.rewind().ok()?;
    let mut text = String::new();
    file.read_to_string(&mut text).ok()?;
    Some(text)
}

#[cfg(unix)]
fn try_lock_owner_file(file: &File) -> std::io::Result<bool> {
    use std::os::fd::AsRawFd;

    let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if result == 0 {
        return Ok(true);
    }
    let error = std::io::Error::last_os_error();
    if matches!(
        error.raw_os_error(),
        Some(code) if code == libc::EWOULDBLOCK || code == libc::EAGAIN
    ) {
        Ok(false)
    } else {
        Err(error)
    }
}

#[cfg(not(unix))]
fn try_lock_owner_file(_file: &File) -> std::io::Result<bool> {
    Ok(true)
}

#[cfg(unix)]
fn file_still_names_open_inode(path: &Path, file: &File) -> bool {
    use std::os::unix::fs::MetadataExt;

    let Ok(open) = file.metadata() else {
        return false;
    };
    let Ok(named) = std::fs::metadata(path) else {
        return false;
    };
    open.dev() == named.dev() && open.ino() == named.ino()
}

#[cfg(not(unix))]
fn file_still_names_open_inode(path: &Path, _file: &File) -> bool {
    path.is_file()
}

fn remove_exact_open_file(path: &Path, file: &File) -> bool {
    file_still_names_open_inode(path, file) && std::fs::remove_file(path).is_ok()
}

fn remove_owned_record(path: &Path, owner: &str, file: &File) -> bool {
    open_record_text(file)
        .as_deref()
        .and_then(record_owner)
        .is_some_and(|recorded| recorded == owner)
        && path_record_owner(path).as_deref() == Some(owner)
        && remove_exact_open_file(path, file)
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

pub(crate) fn acquire_while_optional(
    state_root: &Path,
    class: ResourceClass,
    timeout: Option<Duration>,
    stop: &dyn Fn() -> bool,
) -> Result<ResourceLease> {
    let configured = capacity_for(class).min(aggregate_capacity());
    let limit = configured
        .saturating_sub(adaptive_penalty(state_root, class))
        .max(1);
    let lease_root = state_root.join("resource-leases");
    std::fs::create_dir_all(&lease_root)
        .with_context(|| format!("creating resource lease directory {}", lease_root.display()))?;
    let mut waiter = create_waiter(&lease_root, class)?;
    let started = Instant::now();
    loop {
        if stop() {
            bail!("cancelled waiting for shared {} capacity", class.as_str());
        }
        if !waiter_is_eligible(&lease_root, class, &waiter, limit)? {
            report_wait(class, started, "queued behind earlier work");
            if timeout.is_some_and(|timeout| started.elapsed() >= timeout) {
                bail!("timed out waiting for shared {} capacity", class.as_str());
            }
            std::thread::sleep(Duration::from_millis(50));
            continue;
        }
        if !memory_budget_allows(class) {
            report_wait(class, started, "insufficient available memory");
            if timeout.is_some_and(|timeout| started.elapsed() >= timeout) {
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
            match OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(mut file) => {
                    let owner = owner_token();
                    if !try_lock_owner_file(&file)
                        .context("locking process-capacity lease owner")?
                    {
                        let _ = remove_exact_open_file(&path, &file);
                        bail!("new process-capacity lease was already locked");
                    }
                    // Eligibility was checked before the slot scan. Recheck
                    // after claiming a slot so a waiter replacement cannot
                    // turn that check/use window into an unqueued admission.
                    if let Err(error) = ensure_waiter_owned(class, &waiter) {
                        let _ = remove_exact_open_file(&path, &file);
                        return Err(error);
                    }
                    if let Err(error) = file
                        .write_all(owner_record(&owner, class, "lease").as_bytes())
                        .and_then(|()| file.sync_all())
                    {
                        // Remove only the inode created above. If the write was
                        // partial and cleanup loses a race, the incomplete-record
                        // grace lets another process recover it safely later.
                        let _ = remove_exact_open_file(&path, &file);
                        return Err(error).context("recording process-capacity lease owner");
                    }
                    // Releasing the queue token is part of the ownership
                    // transfer. If it was replaced during the final window,
                    // relinquish the new slot and fail instead of silently
                    // admitting a caller that no longer owns its queue place.
                    if !waiter.release() {
                        let _ = remove_owned_record(&path, &owner, &file);
                        return Err(waiter_ownership_lost(class, &waiter));
                    }
                    return Ok(ResourceLease {
                        path,
                        owner,
                        _file: file,
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    reclaim_stale_record(&path);
                }
                Err(error) => return Err(error).context("acquiring process capacity"),
            }
        }
        if timeout.is_some_and(|timeout| started.elapsed() >= timeout) {
            bail!("timed out waiting for shared {} capacity", class.as_str());
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[derive(Debug)]
struct WaiterGuard {
    path: PathBuf,
    owner: String,
    file: File,
}

impl WaiterGuard {
    fn release(&mut self) -> bool {
        if !remove_owned_record(&self.path, &self.owner, &self.file) {
            return false;
        }
        self.path.clear();
        true
    }
}

impl Drop for WaiterGuard {
    fn drop(&mut self) {
        if !self.path.as_os_str().is_empty() {
            remove_owned_record(&self.path, &self.owner, &self.file);
        }
    }
}

fn create_waiter(lease_root: &Path, class: ResourceClass) -> Result<WaiterGuard> {
    for nonce in 0_u32.. {
        let path = lease_root.join(format!(
            "{}-wait-{:020}-{}-{nonce}.queue",
            class.as_str(),
            unix_millis(),
            std::process::id()
        ));
        match OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(mut file) => {
                let owner = owner_token();
                if !try_lock_owner_file(&file).context("locking process-capacity waiter owner")? {
                    let _ = remove_exact_open_file(&path, &file);
                    bail!("new process-capacity waiter was already locked");
                }
                if let Err(error) = file
                    .write_all(owner_record(&owner, class, "waiter").as_bytes())
                    .and_then(|()| file.sync_all())
                {
                    let _ = remove_exact_open_file(&path, &file);
                    return Err(error).context("recording process-capacity waiter owner");
                }
                return Ok(WaiterGuard { path, owner, file });
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
    own: &WaiterGuard,
    available_slots: usize,
) -> Result<bool> {
    ensure_waiter_owned(class, own)?;
    let prefix = format!("{}-wait-", class.as_str());
    let mut waiters = Vec::new();
    for entry in std::fs::read_dir(lease_root)? {
        let path = entry?.path();
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with(&prefix) && name.ends_with(".queue"))
            && !reclaim_stale_record(&path)
        {
            waiters.push(path);
        }
    }
    waiters.sort();
    // Close the scan window too: a pathname can be removed or replaced while
    // the directory is being inspected.
    ensure_waiter_owned(class, own)?;
    Ok(waiters
        .iter()
        .take(available_slots.max(1))
        .any(|path| path == &own.path))
}

fn ensure_waiter_owned(class: ResourceClass, own: &WaiterGuard) -> Result<()> {
    if !own.path.as_os_str().is_empty()
        && path_record_owner(&own.path).as_deref() == Some(own.owner.as_str())
        && file_still_names_open_inode(&own.path, &own.file)
    {
        return Ok(());
    }
    Err(waiter_ownership_lost(class, own))
}

fn waiter_ownership_lost(class: ResourceClass, own: &WaiterGuard) -> anyhow::Error {
    anyhow::Error::new(WaiterOwnershipLost {
        class,
        path: own.path.clone(),
    })
}

#[cfg(test)]
fn lease_is_stale(path: &Path) -> bool {
    let text = std::fs::read_to_string(path).ok();
    record_text_is_stale(text.as_deref(), file_age(path))
}

/// Reclaim an abandoned owner record without ever unlinking a replacement.
/// The open inode's advisory lock serializes reclaimers on Linux/macOS; every
/// live v2 owner holds that same lock for its full lease/wait lifetime.
fn reclaim_stale_record(path: &Path) -> bool {
    let Ok(file) = OpenOptions::new().read(true).write(true).open(path) else {
        return false;
    };
    if !matches!(try_lock_owner_file(&file), Ok(true)) {
        return false;
    }
    let Some(text) = open_record_text(&file) else {
        return false;
    };
    let age = file
        .metadata()
        .ok()
        .and_then(|metadata| metadata_age(&metadata));
    if !record_text_is_stale(Some(&text), age) || !file_still_names_open_inode(path, &file) {
        return false;
    }
    match record_owner(&text) {
        Some(owner) if path_record_owner(path).as_deref() != Some(owner) => return false,
        None if std::fs::read_to_string(path).ok().as_deref() != Some(text.as_str()) => {
            return false;
        }
        _ => {}
    }
    std::fs::remove_file(path).is_ok()
}

fn record_text_is_stale(text: Option<&str>, age: Option<Duration>) -> bool {
    let Some(text) = text else {
        return incomplete_lease_is_stale(age);
    };
    if text.lines().any(|line| line.starts_with("owner=")) {
        let owner = record_owner(text);
        let pid = text.lines().find_map(|line| {
            line.strip_prefix("pid=")?
                .trim()
                .parse::<u32>()
                .ok()
                .filter(|pid| *pid > 0)
        });
        let birth_field = text
            .lines()
            .find_map(|line| line.strip_prefix("birth="))
            .map(str::trim)
            .filter(|birth| !birth.is_empty());
        if owner.is_none() || pid.is_none() || birth_field.is_none() {
            return incomplete_lease_is_stale(age);
        }
        let birth = birth_field.filter(|birth| *birth != "unknown");
        return owner_record_is_stale_with_age(pid, birth, age);
    }

    // Backward compatibility for original PID-first records. File age versus
    // process uptime rejects a reused live PID when no birth identity exists.
    let pid = text
        .split_whitespace()
        .next()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|pid| *pid > 0);
    let birth = recorded_birth(text);
    owner_record_is_stale_with_age(pid, birth, age)
}

fn incomplete_lease_is_stale(age: Option<Duration>) -> bool {
    age.is_some_and(|age| age >= INCOMPLETE_LEASE_GRACE)
}

fn file_age(path: &Path) -> Option<Duration> {
    metadata_age(&std::fs::metadata(path).ok()?)
}

fn metadata_age(metadata: &std::fs::Metadata) -> Option<Duration> {
    metadata.modified().ok()?.elapsed().ok()
}

fn recorded_birth(text: &str) -> Option<&str> {
    text.lines()
        .find_map(|line| line.strip_prefix("birth="))
        .map(str::trim)
        .filter(|birth| !birth.is_empty())
}

pub(crate) fn current_process_birth_identity() -> Option<&'static str> {
    CURRENT_PROCESS_BIRTH
        .get_or_init(|| process_birth_identity(std::process::id()))
        .as_deref()
}

#[cfg(target_os = "linux")]
fn process_birth_identity(pid: u32) -> Option<String> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // `/proc/<pid>/stat` field 2 is a parenthesized command that may itself
    // contain spaces or `)`, so split at the final delimiter. Field 22
    // (`starttime`) is then index 19 in the remaining field-3-based slice and
    // has scheduler-tick resolution, avoiding same-second PID-reuse aliases.
    let (_, fields) = stat.rsplit_once(") ")?;
    let start_ticks = fields.split_whitespace().nth(19)?.parse::<u64>().ok()?;
    Some(format!("linux-start-ticks:{start_ticks}"))
}

#[cfg(target_os = "macos")]
fn process_birth_identity(pid: u32) -> Option<String> {
    let output = std::process::Command::new("/bin/ps")
        .args(["-o", "lstart=", "-p", &pid.to_string()])
        .env("LC_ALL", "C")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let birth = String::from_utf8(output.stdout).ok()?;
    let birth = birth.split_whitespace().collect::<Vec<_>>().join(" ");
    (!birth.is_empty()).then_some(birth)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn process_birth_identity(_pid: u32) -> Option<String> {
    None
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn process_uptime(pid: u32) -> Option<Duration> {
    let output = std::process::Command::new("/bin/ps")
        .args(["-o", "etime=", "-p", &pid.to_string()])
        .env("LC_ALL", "C")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    parse_process_elapsed(String::from_utf8(output.stdout).ok()?.trim())
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn process_uptime(_pid: u32) -> Option<Duration> {
    None
}

fn parse_process_elapsed(value: &str) -> Option<Duration> {
    let (days, clock) = match value.split_once('-') {
        Some((days, clock)) => (days.parse::<u64>().ok()?, clock),
        None => (0, value),
    };
    let fields = clock
        .split(':')
        .map(str::parse::<u64>)
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    let (hours, minutes, seconds) = match fields.as_slice() {
        [minutes, seconds] => (0, *minutes, *seconds),
        [hours, minutes, seconds] => (*hours, *minutes, *seconds),
        _ => return None,
    };
    let seconds = days
        .checked_mul(86_400)?
        .checked_add(hours.checked_mul(3_600)?)?
        .checked_add(minutes.checked_mul(60)?)?
        .checked_add(seconds)?;
    Some(Duration::from_secs(seconds))
}

fn owner_identity_is_stale(
    pid_alive: bool,
    recorded_birth: Option<&str>,
    observed_birth: Option<&str>,
    file_age: Option<Duration>,
    process_uptime: Option<Duration>,
) -> bool {
    if !pid_alive {
        return true;
    }
    if let Some(recorded_birth) = recorded_birth {
        return observed_birth.is_some_and(|observed| observed != recorded_birth);
    }
    // Legacy PID-only records remain valid while their process could have
    // created the file. If the current PID was born well after the file, it is
    // a reuse and cannot own this record.
    file_age
        .zip(process_uptime)
        .is_some_and(|(age, uptime)| age > uptime.saturating_add(Duration::from_secs(5)))
}

pub(crate) fn owner_record_is_stale(
    path: &Path,
    pid: Option<u32>,
    recorded_birth: Option<&str>,
) -> bool {
    owner_record_is_stale_with_age(pid, recorded_birth, file_age(path))
}

fn owner_record_is_stale_with_age(
    pid: Option<u32>,
    recorded_birth: Option<&str>,
    age: Option<Duration>,
) -> bool {
    let Some(pid) = pid else {
        return incomplete_lease_is_stale(age);
    };
    let observed_birth = process_birth_identity(pid);
    let alive = observed_birth.is_some() || pid_is_alive(pid);
    owner_identity_is_stale(
        alive,
        recorded_birth,
        observed_birth.as_deref(),
        age,
        recorded_birth
            .is_none()
            .then(|| process_uptime(pid))
            .flatten(),
    )
}

#[cfg(unix)]
fn pid_is_alive(pid: u32) -> bool {
    let Ok(pid) = i32::try_from(pid) else {
        return false;
    };
    let result = unsafe { libc::kill(pid, 0) };
    result == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(not(unix))]
fn pid_is_alive(_pid: u32) -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_root(label: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "hi-governor-{label}-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    fn guarded_waiter(path: PathBuf, class: ResourceClass) -> WaiterGuard {
        let owner = owner_token();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        assert!(try_lock_owner_file(&file).unwrap());
        file.write_all(owner_record(&owner, class, "waiter").as_bytes())
            .unwrap();
        file.sync_all().unwrap();
        WaiterGuard { path, owner, file }
    }

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
    fn malformed_capacity_records_age_out_after_creation_grace() {
        assert!(!incomplete_lease_is_stale(None));
        assert!(!incomplete_lease_is_stale(Some(Duration::from_secs(59))));
        assert!(incomplete_lease_is_stale(Some(Duration::from_secs(60))));
    }

    #[cfg(unix)]
    #[test]
    fn locked_incomplete_record_is_not_reclaimed_after_grace() {
        let root = test_root("incomplete-locked");
        let path = root.join("model-slot-0.lease");
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        assert!(try_lock_owner_file(&file).unwrap());
        file.write_all(b"owner=partially-written\n").unwrap();
        file.set_times(std::fs::FileTimes::new().set_modified(std::time::SystemTime::UNIX_EPOCH))
            .unwrap();

        assert!(lease_is_stale(&path), "the incomplete record is old enough");
        assert!(
            !reclaim_stale_record(&path),
            "the creator's lifetime lock protects an in-progress write"
        );
        assert!(path.exists());

        drop(file);
        let reclaimed = (0..20).any(|_| {
            let reclaimed = reclaim_stale_record(&path);
            if !reclaimed {
                std::thread::yield_now();
            }
            reclaimed
        });
        assert!(reclaimed, "the abandoned incomplete inode is reclaimable");
        assert!(!path.exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn stale_cleanup_cannot_unlink_a_locked_owner() {
        let root = test_root("stale-locked");
        let path = root.join("verifier-slot-0.lease");
        let owner = owner_token();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        assert!(try_lock_owner_file(&file).unwrap());
        file.write_all(
            format!(
                "owner={owner}\npid={}\nbirth=forged-stale-birth\nclass=verifier\nkind=lease\n",
                std::process::id()
            )
            .as_bytes(),
        )
        .unwrap();

        assert!(lease_is_stale(&path), "the forged identity is stale");
        assert!(!reclaim_stale_record(&path));
        assert!(path.exists());

        drop(file);
        assert!(reclaim_stale_record(&path));
        assert!(!path.exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn resource_lease_drop_preserves_path_replacement() {
        let root = test_root("lease-replacement");
        let path = root.join("setup-slot-0.lease");
        let owner = owner_token();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        assert!(try_lock_owner_file(&file).unwrap());
        file.write_all(owner_record(&owner, ResourceClass::Setup, "lease").as_bytes())
            .unwrap();
        let lease = ResourceLease {
            path: path.clone(),
            owner,
            _file: file,
        };

        std::fs::remove_file(&path).unwrap();
        let replacement_owner = owner_token();
        std::fs::write(
            &path,
            owner_record(&replacement_owner, ResourceClass::Setup, "lease"),
        )
        .unwrap();
        drop(lease);

        assert_eq!(
            path_record_owner(&path).as_deref(),
            Some(replacement_owner.as_str())
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn owner_identity_rejects_reused_pid_without_expiring_live_unlimited_owner() {
        let old_file = Some(Duration::from_secs(3_600));
        let young_process = Some(Duration::from_secs(10));
        assert!(owner_identity_is_stale(
            true,
            Some("old-birth"),
            Some("new-birth"),
            old_file,
            young_process,
        ));
        assert!(owner_identity_is_stale(
            true,
            None,
            Some("new-birth"),
            old_file,
            young_process,
        ));
        assert!(!owner_identity_is_stale(
            true,
            Some("same-birth"),
            Some("same-birth"),
            Some(Duration::from_secs(u32::MAX.into())),
            None,
        ));
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn forged_live_pid_record_is_stale_when_birth_identity_differs() {
        let root = std::env::temp_dir().join(format!(
            "hi-governor-forged-owner-{}-{}",
            std::process::id(),
            unix_millis()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let record = root.join("verifier-slot-0.lease");
        std::fs::write(
            &record,
            format!(
                "{} verifier\nbirth=definitely-not-this-process\n",
                std::process::id()
            ),
        )
        .unwrap();

        assert!(owner_record_is_stale(
            &record,
            Some(std::process::id()),
            Some("definitely-not-this-process")
        ));

        let birth = current_process_birth_identity().expect("ps exposes process birth time");
        std::fs::write(
            &record,
            format!("{} verifier\nbirth={birth}\n", std::process::id()),
        )
        .unwrap();
        assert!(!owner_record_is_stale(
            &record,
            Some(std::process::id()),
            Some(birth)
        ));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn process_elapsed_parser_supports_ps_formats() {
        assert_eq!(
            parse_process_elapsed("01:02"),
            Some(Duration::from_secs(62))
        );
        assert_eq!(
            parse_process_elapsed("03:04:05"),
            Some(Duration::from_secs(11_045))
        );
        assert_eq!(
            parse_process_elapsed("2-03:04:05"),
            Some(Duration::from_secs(183_845))
        );
        assert_eq!(parse_process_elapsed("n/a"), None);
    }

    #[cfg(unix)]
    #[test]
    fn out_of_range_pid_record_is_not_treated_as_a_live_process_group() {
        assert!(!pid_is_alive(u32::MAX));
    }

    #[test]
    fn queue_order_is_fifo_across_processes() {
        let root = test_root("fifo");
        let later_path = root.join(format!(
            "{}-wait-{:020}-2-0.queue",
            ResourceClass::Verifier.as_str(),
            2
        ));
        let earlier_path = root.join(format!(
            "{}-wait-{:020}-1-0.queue",
            ResourceClass::Verifier.as_str(),
            1
        ));
        let later = guarded_waiter(later_path, ResourceClass::Verifier);
        let mut earlier = guarded_waiter(earlier_path, ResourceClass::Verifier);
        assert!(waiter_is_eligible(&root, ResourceClass::Verifier, &earlier, 1).unwrap());
        assert!(!waiter_is_eligible(&root, ResourceClass::Verifier, &later, 1).unwrap());
        assert!(waiter_is_eligible(&root, ResourceClass::Verifier, &later, 2).unwrap());
        assert!(earlier.release());
        assert!(waiter_is_eligible(&root, ResourceClass::Verifier, &later, 1).unwrap());
        drop(earlier);
        drop(later);
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn replaced_waiter_returns_typed_ownership_error_and_old_guard_preserves_replacement() {
        let root = test_root("waiter-replacement");
        let path = root.join("merge-wait-00000000000000000001-1-0.queue");
        let waiter = guarded_waiter(path.clone(), ResourceClass::Merge);

        std::fs::remove_file(&path).unwrap();
        let replacement_owner = owner_token();
        std::fs::write(
            &path,
            owner_record(&replacement_owner, ResourceClass::Merge, "waiter"),
        )
        .unwrap();

        let error = waiter_is_eligible(&root, ResourceClass::Merge, &waiter, 1)
            .expect_err("a replacement must invalidate the old queue admission token");
        assert!(error.downcast_ref::<WaiterOwnershipLost>().is_some());
        drop(waiter);
        assert_eq!(
            path_record_owner(&path).as_deref(),
            Some(replacement_owner.as_str()),
            "the old guard must not unlink the replacement waiter"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    fn exercise_lost_waiter_during_unlimited_acquire(replace: bool) {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::mpsc;

        let root = test_root(if replace {
            "acquire-replaced-waiter"
        } else {
            "acquire-removed-waiter"
        });
        let held = acquire_while_optional(&root, ResourceClass::Merge, None, &|| false).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let worker_root = root.clone();
        let worker_stop = stop.clone();
        let (ready_tx, ready_rx) = mpsc::sync_channel(0);
        let (continue_tx, continue_rx) = mpsc::sync_channel(0);
        let (result_tx, result_rx) = mpsc::sync_channel(1);
        let worker = std::thread::spawn(move || {
            let first_stop_check = AtomicBool::new(true);
            let result = acquire_while_optional(&worker_root, ResourceClass::Merge, None, &|| {
                // `stop` is first consulted immediately after create_waiter
                // has flushed and locked the queue record. Hold that exact
                // point so the main thread can deterministically replace it
                // before the first eligibility check.
                if first_stop_check.swap(false, Ordering::Relaxed)
                    && (ready_tx.send(()).is_err() || continue_rx.recv().is_err())
                {
                    return true;
                }
                worker_stop.load(Ordering::Relaxed)
            })
            .map(drop);
            let _ = result_tx.send(result);
        });

        let lease_root = root.join("resource-leases");
        if let Err(error) = ready_rx.recv_timeout(Duration::from_secs(5)) {
            stop.store(true, Ordering::Relaxed);
            drop(ready_rx);
            drop(continue_tx);
            drop(held);
            worker.join().unwrap();
            panic!("unlimited acquisition did not publish its waiter record: {error}");
        }
        let waiter_paths = std::fs::read_dir(&lease_root)
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("merge-wait-") && name.ends_with(".queue"))
            })
            .collect::<Vec<_>>();
        if waiter_paths.len() != 1 {
            stop.store(true, Ordering::Relaxed);
            let _ = continue_tx.send(());
            drop(held);
            worker.join().unwrap();
            panic!("expected exactly one published waiter record, found {waiter_paths:?}");
        }
        let waiter_path = waiter_paths.into_iter().next().unwrap();

        std::fs::remove_file(&waiter_path).unwrap();
        let replacement_owner = replace.then(owner_token);
        if let Some(owner) = &replacement_owner {
            std::fs::write(
                &waiter_path,
                owner_record(owner, ResourceClass::Merge, "waiter"),
            )
            .unwrap();
        }
        continue_tx.send(()).unwrap();

        let result = match result_rx.recv_timeout(Duration::from_secs(2)) {
            Ok(result) => result,
            Err(error) => {
                stop.store(true, Ordering::Relaxed);
                drop(held);
                worker.join().unwrap();
                panic!("unlimited acquisition hung after losing its waiter: {error}");
            }
        };
        let error = result.expect_err("lost waiter ownership must fail acquisition");
        assert!(
            error.downcast_ref::<WaiterOwnershipLost>().is_some(),
            "unexpected acquisition error: {error:#}"
        );
        if let Some(owner) = replacement_owner {
            assert_eq!(
                path_record_owner(&waiter_path).as_deref(),
                Some(owner.as_str()),
                "the failed old acquisition must preserve the replacement waiter"
            );
        }

        drop(held);
        worker.join().unwrap();
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn replacement_during_unlimited_acquire_returns_instead_of_hanging() {
        exercise_lost_waiter_during_unlimited_acquire(true);
    }

    #[cfg(unix)]
    #[test]
    fn lost_waiter_stress_does_not_hang_unlimited_acquisition() {
        for replace in [false, true].into_iter().cycle().take(16) {
            exercise_lost_waiter_during_unlimited_acquire(replace);
        }
    }

    #[test]
    fn unlimited_capacity_wait_observes_cancellation_and_cleans_waiter() {
        let root = test_root("cancel");
        let held = acquire_while_optional(&root, ResourceClass::Merge, None, &|| false).unwrap();
        let started = Instant::now();
        let error = acquire_while_optional(&root, ResourceClass::Merge, None, &|| {
            started.elapsed() >= Duration::from_millis(75)
        })
        .expect_err("cancellation must stop an otherwise-unbounded capacity wait");

        assert!(format!("{error:#}").contains("cancelled"));
        assert!(started.elapsed() >= Duration::from_millis(75));
        let lease_root = root.join("resource-leases");
        let queued = std::fs::read_dir(&lease_root)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .is_some_and(|name| name.ends_with(".queue"))
            })
            .count();
        assert_eq!(queued, 0, "cancelled waiter cleaned its own queue record");

        drop(held);
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
