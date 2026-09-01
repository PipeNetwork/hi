//! Shared helpers for `gix` status scans.
//!
//! `gix-features` `in_parallel` does `spawn_scoped(...).expect("valid name")`.
//! Under `panic=abort` and a tight `RLIMIT_NPROC`, a failed spawn aborts the
//! whole process instead of becoming a recoverable `JoinError`. Cap
//! `index_worktree_options.thread_limit` so produce workers stay within
//! headroom. `Some(0)` means unlimited in gix — never pass 0.

/// Past 8 produce workers a status scan gains no speed, only spawn pressure.
const HARD_CAP: usize = 8;
/// Reserve for non-gix threads; nproc tests use `used + OUTER_RESERVE - 2`.
pub(crate) const OUTER_RESERVE: usize = 8;

const ENV_THREADS: &str = "HI_GIX_STATUS_THREADS";

/// Pure produce-worker budget. Always `n >= 1`. Caps at 8; shrinks under tight
/// soft nproc headroom (`headroom < 2` → 1).
pub fn compute_gix_status_thread_limit_from(
    cores: usize,
    soft_nproc: Option<usize>,
    threads_used: usize,
) -> usize {
    let cores = cores.max(1);
    let mut limit = cores.min(HARD_CAP);
    if let Some(soft) = soft_nproc {
        let headroom = soft
            .saturating_sub(threads_used)
            .saturating_sub(OUTER_RESERVE);
        if headroom < 2 {
            limit = 1;
        } else {
            limit = limit.min(headroom);
        }
    }
    limit.max(1)
}

/// Production budget (`n >= 1`). Honours `HI_GIX_STATUS_THREADS=N` for `N >= 1`
/// (forced dial; bypasses nproc). Else cores + soft nproc + thread usage.
pub fn compute_gix_status_thread_limit() -> usize {
    if let Ok(raw) = std::env::var(ENV_THREADS)
        && let Some(n) = parse_env_thread_override(&raw)
    {
        return n;
    }
    let cores = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    compute_gix_status_thread_limit_from(cores, soft_nproc_limit(), threads_used())
}

/// `N >= 1` only; reject `0` and garbage.
fn parse_env_thread_override(raw: &str) -> Option<usize> {
    raw.parse::<usize>().ok().filter(|&n| n >= 1)
}

/// Test helper: `None` = uncapped, `Some(n)` with `n >= 1`. Never `Some(0)`.
fn apply_thread_limit<'repo, P>(
    platform: gix::status::Platform<'repo, P>,
    limit: Option<usize>,
) -> gix::status::Platform<'repo, P>
where
    P: gix::Progress + 'static,
{
    debug_assert!(
        !matches!(limit, Some(0)),
        "Some(0) is unlimited in gix — never pass 0"
    );
    platform.index_worktree_options_mut(|opts| {
        opts.thread_limit = limit;
    })
}

/// Apply [`compute_gix_status_thread_limit`] as `Some(n)` on the status platform.
pub fn with_budgeted_thread_limit<'repo, P>(
    platform: gix::status::Platform<'repo, P>,
) -> gix::status::Platform<'repo, P>
where
    P: gix::Progress + 'static,
{
    apply_thread_limit(platform, Some(compute_gix_status_thread_limit()))
}

#[cfg(unix)]
fn soft_nproc_limit() -> Option<usize> {
    let mut lim = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: getrlimit writes only into local `lim`.
    if unsafe { libc::getrlimit(libc::RLIMIT_NPROC, &mut lim) } != 0 {
        return None;
    }
    if lim.rlim_cur == libc::RLIM_INFINITY {
        return None;
    }
    Some(
        lim.rlim_cur
            .min(usize::MAX as libc::rlim_t)
            .try_into()
            .unwrap_or(usize::MAX),
    )
}

#[cfg(not(unix))]
fn soft_nproc_limit() -> Option<usize> {
    None
}

fn threads_used() -> usize {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        status
            .lines()
            .find_map(|line| {
                line.strip_prefix("Threads:")
                    .and_then(|rest| rest.trim().parse().ok())
            })
            .unwrap_or(1)
    }
    #[cfg(not(target_os = "linux"))]
    {
        1
    }
}
