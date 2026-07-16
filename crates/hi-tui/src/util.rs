//! Small formatting and side-effect helpers shared by `App`: token/time
//! humanizers, an error clipper, clipboard/notify escapes, base64, and the
//! `/goal` feedback wording.

use std::io::{self, Write};

/// One-line, length-capped form of an error message for the status bar:
/// whitespace/newlines collapsed, clipped with an ellipsis.
pub(crate) fn clip_reason(s: &str) -> String {
    let one_line = s.split_whitespace().collect::<Vec<_>>().join(" ");
    const MAX: usize = 60;
    if one_line.chars().count() > MAX {
        format!("{}…", one_line.chars().take(MAX).collect::<String>())
    } else {
        one_line
    }
}

/// Compact token count for the working line: `1234` → `1.2k`, `45000` → `45k`.
/// The live working line and the settled usage summary share one humanizer (in
/// `hi-agent`), so the same count never renders two different ways.
pub(crate) fn fmt_count(n: u64) -> String {
    hi_agent::humanize_count(n)
}

/// Format an elapsed-seconds count compactly: `45s`, `14m 28s`, `1h 02m`.
pub(crate) fn fmt_elapsed(secs: u64) -> String {
    let (h, m, s) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    if h > 0 {
        format!("{h}h {m:02}m")
    } else if m > 0 {
        format!("{m}m {s:02}s")
    } else {
        format!("{s}s")
    }
}

pub(crate) fn fmt_rate_limits(limits: Option<hi_ai::RateLimitState>) -> Option<String> {
    let limits = limits.filter(|limits| limits.has_data())?;
    let mut parts = Vec::new();
    if let Some(part) = fmt_bucket("req", limits.requests_min) {
        parts.push(part);
    } else if let Some(part) = fmt_bucket("req/hr", limits.requests_hour) {
        parts.push(part);
    }
    if let Some(part) = fmt_bucket("tok", limits.tokens_min) {
        parts.push(part);
    } else if let Some(part) = fmt_bucket("tok/hr", limits.tokens_hour) {
        parts.push(part);
    }
    (!parts.is_empty()).then(|| format!("limits {}", parts.join(" · ")))
}

fn fmt_bucket(label: &str, bucket: hi_ai::RateLimitBucket) -> Option<String> {
    if bucket.limit == 0 {
        return None;
    }
    let reset = if bucket.reset_seconds > 0 {
        format!(" reset {}", fmt_elapsed(bucket.reset_seconds))
    } else {
        String::new()
    };
    Some(format!(
        "{label} {}/{}{reset}",
        fmt_count(bucket.remaining),
        fmt_count(bucket.limit)
    ))
}

/// Nudge the terminal that a turn finished: the BEL (which most terminals turn
/// into a dock bounce / taskbar flash / audible ping when unfocused) plus an
/// OSC 9 desktop notification for terminals that support it (iTerm2, WezTerm, …).
/// Written straight to the tty; both are non-printing, so they don't disturb the
/// ratatui frame.
pub(crate) fn notify_done() {
    let mut out = io::stdout().lock();
    let _ = write!(out, "\x07\x1b]9;hi — turn complete\x07");
    let _ = out.flush();
}

/// Copy `text` to the system clipboard. Tries a local OS clipboard tool first
/// (the most reliable path, independent of terminal escape support), then always
/// also emits OSC 52 so remote/SSH sessions and OSC-52-capable terminals still
/// get it. Succeeds if either path did — notably fixing terminals like macOS
/// Terminal.app that silently ignore OSC 52.
pub(crate) fn copy_to_clipboard(text: &str) -> io::Result<()> {
    let native = copy_native(text);
    let osc = copy_osc52(text);
    // Prefer reporting the native result; fall back to the OSC 52 write status.
    native.or(osc)
}

fn copy_osc52(text: &str) -> io::Result<()> {
    let encoded = base64_encode(text.as_bytes());
    let mut out = io::stdout().lock();
    write!(out, "\x1b]52;c;{encoded}\x07")?;
    out.flush()
}

/// Pipe `text` into the first available OS clipboard tool. Returns `Ok` only if
/// one ran and exited successfully; `Err(NotFound)` if none is installed (e.g. a
/// bare SSH host), so the caller can rely on the OSC 52 path there.
fn copy_native(text: &str) -> io::Result<()> {
    use std::process::{Command, Stdio};
    // (command, args) in preference order: macOS, Wayland, X11 (xclip/xsel),
    // then Windows/WSL.
    const TOOLS: &[(&str, &[&str])] = &[
        ("pbcopy", &[]),
        ("wl-copy", &[]),
        ("xclip", &["-selection", "clipboard"]),
        ("xsel", &["--clipboard", "--input"]),
        ("clip.exe", &[]),
    ];
    for (cmd, args) in TOOLS {
        let mut child = match Command::new(cmd)
            .args(*args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(child) => child,
            Err(_) => continue, // not installed — try the next tool
        };
        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(text.as_bytes());
            // Drop stdin to send EOF so the tool finishes and sets the clipboard.
        }
        if let Ok(status) = child.wait()
            && status.success()
        {
            return Ok(());
        }
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "no OS clipboard tool found",
    ))
}

pub(crate) fn base64_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0];
        let b1 = *chunk.get(1).unwrap_or(&0);
        let b2 = *chunk.get(2).unwrap_or(&0);
        out.push(TABLE[(b0 >> 2) as usize] as char);
        out.push(TABLE[(((b0 & 0b0000_0011) << 4) | (b1 >> 4)) as usize] as char);
        if chunk.len() > 1 {
            out.push(TABLE[(((b1 & 0b0000_1111) << 2) | (b2 >> 6)) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(TABLE[(b2 & 0b0011_1111) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

/// The `/goal` feedback line, and whether it's a prominent applied-change
/// confirmation (set/clear) versus a dim status read-out (a bare `/goal`).
/// `goal` is the agent's goal *after* the action, so a set echoes exactly what's
/// stored. Pure, so the wording is unit-testable without an agent.
pub(crate) fn goal_feedback(arg: &str, goal: Option<&str>) -> (String, bool) {
    match arg.trim() {
        "" => match goal {
            Some(g) => (format!("goal: {g}"), false),
            None => ("goal: off (set one with /goal <text>)".to_string(), false),
        },
        "clear" | "off" | "none" => ("✓ goal cleared".to_string(), true),
        _ => (
            format!(
                "✓ goal set — steers every turn until cleared: \"{}\"",
                goal.unwrap_or_default()
            ),
            true,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmt_count_humanizes() {
        assert_eq!(fmt_count(0), "0");
        assert_eq!(fmt_count(999), "999");
        assert_eq!(fmt_count(1234), "1.2k");
        assert_eq!(fmt_count(45000), "45k");
    }

    #[test]
    fn working_and_summary_share_one_humanizer() {
        // The live working line (fmt_count) and the settled usage summary
        // (hi_agent::humanize_count) must format a count identically — else the
        // same number renders two ways as a turn finishes (the regression fixed).
        for n in [0u64, 999, 1234, 22_864, 12_000, 1_000_000, 1_500_000] {
            assert_eq!(fmt_count(n), hi_agent::humanize_count(n), "diverged at {n}");
        }
    }

    #[test]
    fn fmt_elapsed_shows_minutes_and_seconds() {
        assert_eq!(fmt_elapsed(0), "0s");
        assert_eq!(fmt_elapsed(45), "45s");
        assert_eq!(fmt_elapsed(60), "1m 00s");
        assert_eq!(fmt_elapsed(868), "14m 28s"); // the reported "868s"
        assert_eq!(fmt_elapsed(3600), "1h 00m");
        assert_eq!(fmt_elapsed(3661), "1h 01m");
    }

    #[test]
    fn clip_reason_collapses_and_truncates() {
        assert_eq!(clip_reason("a\n  b   c"), "a b c");
        assert!(clip_reason(&"x".repeat(200)).ends_with('…'));
    }

    #[test]
    fn base64_encoder_handles_padding() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
    }

    #[test]
    fn goal_feedback_is_prominent_on_change_quiet_on_read() {
        // Setting echoes the stored goal as a prominent ✓ confirmation that says
        // it persists — the visibility fix for "/goal seemed to do nothing".
        let (msg, prominent) = goal_feedback("ship it", Some("ship it"));
        assert!(prominent, "a set is an applied change, shown plainly");
        assert!(msg.starts_with("✓ goal set"), "got: {msg}");
        assert!(
            msg.contains("ship it") && msg.contains("until cleared"),
            "echoes the goal and that it persists: {msg}"
        );
        // Clearing is also a prominent ✓.
        assert_eq!(
            goal_feedback("clear", None),
            ("✓ goal cleared".to_string(), true)
        );
        // A bare /goal is a quiet read-out, not a ✓ confirmation.
        let (read, prominent) = goal_feedback("", Some("ship it"));
        assert_eq!((read.as_str(), prominent), ("goal: ship it", false));
        assert!(!goal_feedback("", None).1, "the off read-out stays dim too");
    }
}
