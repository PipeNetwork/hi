//! `hi loops daemon` — run the `/loop` manager headless, so loops keep firing
//! after you close the TUI. It reads the project's `loops.json`, fires every
//! loop on its cadence (auto-fixes and triggers included), records loud events
//! to `activity.jsonl`, and prints them, until interrupted.
//!
//! It holds the per-project fire-lock ([`crate::lock`]) so it and a TUI never
//! both fire the same loops. Come back to `hi` later and `/digest` shows you
//! everything the daemon noticed while you were away.

use std::io::Write;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, bail};

use crate::FleetLauncher;

/// Run the headless loop daemon until Ctrl-C. Fails fast if another firer (a TUI
/// or a second daemon) already owns this project's loops.
pub async fn run_loops_daemon(launcher: FleetLauncher) -> Result<()> {
    let Some(loops_file) = launcher.loops_file.clone() else {
        bail!("no project loops file — the daemon needs a persisted loops.json location");
    };
    let lock_path = crate::lock::lock_path(&loops_file);
    let Some(_lock) = crate::lock::try_acquire(&lock_path) else {
        let who = crate::lock::live_holder(&lock_path)
            .map(|p| p.to_string())
            .unwrap_or_else(|| "another process".into());
        bail!(
            "loop firing is already owned by pid {who} — only one firer runs per project; \
             stop it (or close the TUI) first"
        );
    };

    let count = crate::loops::persisted_count(&loops_file);
    println!(
        "⟳ hi loop daemon (pid {}) — firing {count} loop(s) for this project; Ctrl-C to stop",
        std::process::id()
    );
    if let Some(sinks) = crate::notify::NotifyConfig::from_env().describe() {
        println!("  notifications: {sinks}");
    }
    let _ = std::io::stdout().flush();

    let handle = crate::loops::start(Arc::new(launcher), Some(loops_file));

    loop {
        tokio::select! {
            // Ctrl-C (SIGINT) or SIGTERM (kill / systemd stop) → clean exit, so
            // the fire-lock is always released via `_lock`'s drop.
            _ = shutdown_signal() => {
                println!("⟳ daemon stopping — loops pause until a firer (TUI or daemon) next runs");
                let _ = std::io::stdout().flush();
                return Ok(());
            }
            _ = tokio::time::sleep(Duration::from_millis(500)) => {
                for (line, loud) in handle.drain() {
                    // Loud lines to stderr (so `| grep` on stdout can filter to
                    // routine ticks, or vice versa); quiet re-arm notes to stdout.
                    if loud {
                        eprintln!("{line}");
                    } else {
                        println!("{line}");
                    }
                }
                let _ = std::io::stdout().flush();
            }
        }
    }
}

/// Resolves when the daemon should stop: Ctrl-C anywhere, or SIGTERM on unix.
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        match signal(SignalKind::terminate()) {
            Ok(mut term) => {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {}
                    _ = term.recv() => {}
                }
            }
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
