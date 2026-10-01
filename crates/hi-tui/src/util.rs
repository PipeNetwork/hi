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
pub(crate) fn fmt_count(n: u64) -> String {
    humanize_count(n)
}

pub(crate) fn humanize_count(n: u64) -> String {
    match n {
        0..=999 => n.to_string(),
        1_000..=9_999 => format!("{:.1}k", n as f64 / 1000.0),
        10_000..=999_999 => format!("{}k", n / 1000),
        _ => format!("{:.1}M", n as f64 / 1_000_000.0),
    }
}

/// Truncate to `max` characters, appending an ellipsis when shortened.
pub(crate) fn clip(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let kept: String = s.chars().take(max).collect();
        format!("{kept}…")
    }
}

/// Short tool-call label: name plus a salient argument when the JSON parses.
pub(crate) fn tool_label(name: &str, arguments: &str) -> String {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(arguments) else {
        return name.to_string();
    };
    let str_field = |key: &str| value.get(key).and_then(|v| v.as_str()).map(str::to_string);
    let arg = match name {
        "read" | "write" | "edit" => str_field("path"),
        "list" => Some(str_field("path").unwrap_or_else(|| ".".into())),
        "grep" => str_field("pattern"),
        "bash" => str_field("command").map(|command| hi_tools::shell_title(&command)),
        _ => None,
    };
    match arg {
        Some(arg) => format!("{name} {}", clip(&arg, 80)),
        None => name.to_string(),
    }
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

/// Grok-build "Worked for" marker: `5.0s`, `23m48s`, `1h2m`.
pub(crate) fn fmt_worked(elapsed: std::time::Duration) -> String {
    let secs = elapsed.as_secs();
    if secs < 10 {
        format!("{:.1}s", elapsed.as_secs_f64())
    } else if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m{}s", secs / 60, secs % 60)
    } else {
        format!("{}h{}m", secs / 3600, (secs % 3600) / 60)
    }
}

/// Right-aligned prompt timestamp, grok-build short form (`4:55 AM`).
pub(crate) fn fmt_clock(at: std::time::SystemTime) -> String {
    use chrono::Timelike;
    let dt = chrono::DateTime::<chrono::Local>::from(at);
    let h24 = dt.hour();
    let m = dt.minute();
    let (h12, ampm) = match h24 {
        0 => (12, "AM"),
        1..=11 => (h24, "AM"),
        12 => (12, "PM"),
        _ => (h24 - 12, "PM"),
    };
    format!("{h12}:{m:02} {ampm}")
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

/// Whether this process can hand text to a clipboard on the machine the
/// user is sitting at. Over SSH with no X11 or Wayland forwarding there is
/// nothing here to paste from; the local terminal's own highlight is the
/// clipboard, and capturing the mouse takes that highlight away.
pub(crate) fn host_clipboard_available() -> bool {
    #[cfg(any(
        target_os = "linux",
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "netbsd"
    ))]
    {
        host_clipboard_from(
            std::env::var("DISPLAY").ok().as_deref(),
            std::env::var("WAYLAND_DISPLAY").ok().as_deref(),
        )
    }
    #[cfg(not(any(
        target_os = "linux",
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "netbsd"
    )))]
    {
        true
    }
}

pub(crate) fn host_clipboard_from(display: Option<&str>, wayland: Option<&str>) -> bool {
    fn present(value: Option<&str>) -> bool {
        value.is_some_and(|text| !text.is_empty())
    }
    present(display) || present(wayland)
}

/// Copy `text` to the system clipboard a browser can paste.
///
/// The in-process clipboard (X11 / Wayland / the OS pasteboard) is what
/// Ctrl-V and middle-click read. External tools are the fallback when that
/// connection is missing. OSC 52 still goes out so an SSH session can land
/// on the local clipboard when the terminal honors it. Succeeds if any path
/// did.
pub(crate) fn copy_to_clipboard(text: &str) -> io::Result<()> {
    let owned = copy_arboard(text).or_else(|_| copy_native(text));
    let osc = copy_osc52(text);
    owned.or(osc)
}

/// OSC 52 payload. `ST` (`ESC \`) is the terminator xterm.js-based terminals
/// accept; BEL is dropped by some of them. Inside tmux the sequence has to be
/// wrapped in a DCS passthrough or tmux eats it.
fn osc52_sequence(text: &str, tmux: bool) -> String {
    let encoded = base64_encode(text.as_bytes());
    let osc = format!("\x1b]52;c;{encoded}\x1b\\");
    if tmux {
        format!("\x1bPtmux;{}\x1b\\", osc.replace('\x1b', "\x1b\x1b"))
    } else {
        osc
    }
}

fn copy_osc52(text: &str) -> io::Result<()> {
    let payload = osc52_sequence(text, std::env::var_os("TMUX").is_some());
    write_terminal(payload.as_bytes())
}

/// Write terminal bytes. A live TUI writes `/dev/tty` so the sequence is not
/// split across the draw buffer. Tests and piped stdout stay on `stdout`,
/// which the harness captures.
fn write_terminal(bytes: &[u8]) -> io::Result<()> {
    use std::io::IsTerminal;
    if io::stdout().is_terminal()
        && let Ok(mut tty) = std::fs::OpenOptions::new().write(true).open("/dev/tty")
    {
        tty.write_all(bytes)?;
        return tty.flush();
    }
    let mut out = io::stdout().lock();
    out.write_all(bytes)?;
    out.flush()
}

fn clip_err(err: impl std::fmt::Display) -> io::Error {
    io::Error::other(err.to_string())
}

/// One clipboard connection for the process. On Linux the app that last wrote
/// the clipboard has to stay alive and answer paste requests; dropping the
/// handle makes a later paste into a browser come back empty.
fn clipboard_slot() -> &'static std::sync::Mutex<Option<arboard::Clipboard>> {
    use std::sync::{Mutex, OnceLock};
    static CLIP: OnceLock<Mutex<Option<arboard::Clipboard>>> = OnceLock::new();
    CLIP.get_or_init(|| Mutex::new(None))
}

fn with_clipboard<T>(f: impl FnOnce(&mut arboard::Clipboard) -> io::Result<T>) -> io::Result<T> {
    let mut guard = clipboard_slot()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if guard.is_none() {
        *guard = Some(arboard::Clipboard::new().map_err(clip_err)?);
    }
    let Some(ctx) = guard.as_mut() else {
        return Err(io::Error::other("clipboard unavailable"));
    };
    f(ctx)
}

fn reset_clipboard() {
    if let Ok(mut guard) = clipboard_slot().lock() {
        *guard = None;
    }
}

fn copy_arboard(text: &str) -> io::Result<()> {
    if copy_arboard_once(text).is_ok() {
        return Ok(());
    }
    // A dead display connection (session restart) fails once; reconnect.
    reset_clipboard();
    copy_arboard_once(text)
}

fn copy_arboard_once(text: &str) -> io::Result<()> {
    with_clipboard(|ctx| {
        ctx.set_text(text).map_err(clip_err)?;
        set_primary(ctx, text);
        Ok(())
    })
}

fn set_primary(ctx: &mut arboard::Clipboard, text: &str) {
    #[cfg(target_os = "linux")]
    {
        use arboard::{LinuxClipboardKind, SetExtLinux};
        // Middle-click paste reads PRIMARY. Some Wayland compositors have no
        // primary selection; CLIPBOARD above is the one Ctrl-V uses.
        let _ = ctx
            .set()
            .clipboard(LinuxClipboardKind::Primary)
            .text(text.to_owned());
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ctx, text);
    }
}

fn arboard_clipboard_text() -> io::Result<String> {
    with_clipboard(|ctx| ctx.get_text().map_err(clip_err))
}

fn arboard_primary_text() -> io::Result<String> {
    #[cfg(target_os = "linux")]
    let result = {
        use arboard::{GetExtLinux, LinuxClipboardKind};
        with_clipboard(|ctx| {
            ctx.get()
                .clipboard(LinuxClipboardKind::Primary)
                .text()
                .map_err(clip_err)
        })
    };
    #[cfg(not(target_os = "linux"))]
    let result = Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "no primary selection",
    ));
    result
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

/// Max characters we'll pull from the OS clipboard into the prompt.
const PASTE_CHAR_CAP: usize = 512 * 1024;

/// Read CLIPBOARD (Ctrl+V / Cmd+V), grok-style: never falls back to PRIMARY.
pub(crate) fn read_clipboard() -> io::Result<String> {
    if let Ok(text) = arboard_clipboard_text()
        && !text.is_empty()
    {
        return Ok(clamp_paste(&text));
    }
    read_from_tools(&[
        ("pbpaste", &[] as &[&str]),
        ("wl-paste", &["--no-newline"]),
        ("xclip", &["-selection", "clipboard", "-o"]),
        ("xsel", &["--clipboard", "--output"]),
        (
            "powershell.exe",
            &["-NoProfile", "-Command", "Get-Clipboard"],
        ),
    ])
}

/// Read X11/Wayland PRIMARY (middle-click paste).
pub(crate) fn read_primary() -> io::Result<String> {
    if let Ok(text) = arboard_primary_text()
        && !text.is_empty()
    {
        return Ok(clamp_paste(&text));
    }
    read_from_tools(&[
        ("wl-paste", &["--primary", "--no-newline"]),
        ("xclip", &["-selection", "primary", "-o"]),
        ("xsel", &["--primary", "--output"]),
    ])
    .or_else(|_| read_clipboard())
}

fn read_from_tools(tools: &[(&str, &[&str])]) -> io::Result<String> {
    use std::process::{Command, Stdio};
    for (cmd, args) in tools {
        let output = Command::new(cmd)
            .args(*args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output();
        let Ok(output) = output else {
            continue;
        };
        if !output.status.success() {
            continue;
        }
        let text = String::from_utf8_lossy(&output.stdout).into_owned();
        if text.is_empty() {
            continue;
        }
        return Ok(clamp_paste(&text));
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "no clipboard contents",
    ))
}

pub(crate) fn clamp_paste(text: &str) -> String {
    let mut out: String = text.chars().take(PASTE_CHAR_CAP).collect();
    if text.chars().count() > PASTE_CHAR_CAP {
        out.push('…');
    }
    out
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
    fn clamp_paste_caps_huge_clipboard() {
        let huge: String = "x".repeat(PASTE_CHAR_CAP + 10);
        let out = clamp_paste(&huge);
        assert_eq!(out.chars().count(), PASTE_CHAR_CAP + 1);
        assert!(out.ends_with('…'));
        assert_eq!(clamp_paste("hi\nthere"), "hi\nthere");
    }

    #[test]
    fn fmt_count_humanizes() {
        assert_eq!(fmt_count(0), "0");
        assert_eq!(fmt_count(999), "999");
        assert_eq!(fmt_count(1234), "1.2k");
        assert_eq!(fmt_count(45000), "45k");
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
    fn fmt_worked_matches_grok_build_compact() {
        assert_eq!(fmt_worked(std::time::Duration::from_millis(2300)), "2.3s");
        assert_eq!(fmt_worked(std::time::Duration::from_secs(5)), "5.0s");
        assert_eq!(fmt_worked(std::time::Duration::from_secs(45)), "45s");
        assert_eq!(fmt_worked(std::time::Duration::from_secs(125)), "2m5s");
        assert_eq!(fmt_worked(std::time::Duration::from_secs(1428)), "23m48s");
        assert_eq!(fmt_worked(std::time::Duration::from_secs(3723)), "1h2m");
    }

    #[test]
    fn clip_reason_collapses_and_truncates() {
        assert_eq!(clip_reason("a\n  b   c"), "a b c");
        assert!(clip_reason(&"x".repeat(200)).ends_with('…'));
    }

    #[test]
    fn host_clipboard_requires_a_display_server() {
        assert!(!host_clipboard_from(None, None));
        assert!(!host_clipboard_from(Some(""), Some("")));
        assert!(host_clipboard_from(Some(":0"), None));
        assert!(host_clipboard_from(None, Some("wayland-0")));
    }

    fn osc52_uses_st_and_tmux_passthrough() {
        let plain = osc52_sequence("hi", false);
        assert!(plain.starts_with("\x1b]52;c;aGk="), "{plain:?}");
        assert!(plain.ends_with("\x1b\\"), "{plain:?}");
        assert!(
            !plain.contains('\u{7}'),
            "BEL is not a reliable OSC terminator"
        );
        let wrapped = osc52_sequence("hi", true);
        assert!(
            wrapped.starts_with("\x1bPtmux;\x1b\x1b]52;c;"),
            "{wrapped:?}"
        );
        assert!(wrapped.ends_with("\x1b\x1b\\\x1b\\"), "{wrapped:?}");
    }

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
