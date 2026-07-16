//! The TUI color theme: a named slot vocabulary so every rendered color maps to
//! a *role* (user, tool, error, running, …) rather than a hardcoded ANSI name.
//!
//! Modeled on grok-build's theme system. Two built-in palettes:
//! - `dark` — a GrokNight-style truecolor palette (neutral dark base, magenta
//!   assistant accent) shown on terminals that support 24-bit color.
//! - `ansi` — named ANSI colors that respect the user's own terminal theme;
//!   this reproduces hi's historical look and is the fallback on terminals
//!   without truecolor.
//!
//! Selection: `HI_THEME` (`dark` | `light` | `ansi` | `auto`, default `auto`).
//! `auto` picks `dark` on a truecolor terminal, `ansi` otherwise — so the
//! designed palette shows where it renders faithfully and never regresses a
//! basic terminal. A single global [`theme()`] accessor is read on every
//! render; [`set_mode`] switches at runtime (the `/theme` command), and
//! [`poll_auto_appearance`] follows the OS light/dark setting when mode = auto.

use std::sync::{OnceLock, RwLock};

use ratatui::style::Color;

/// Every color role the TUI draws. One field per semantic slot; renderers ask
/// for a role, never a raw `Color`, so the whole look restyles from one place.
///
/// The full palette is defined up front; call sites migrate onto it in phases
/// (transcript first, then chrome), so some slots have no reader yet.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub struct Theme {
    // Accents — the left gutter bar and headers take their color from the block
    // role, so a glance at the bar tells you what a block is.
    pub accent_user: Color,
    pub accent_assistant: Color,
    pub accent_thinking: Color,
    pub accent_tool: Color,
    pub accent_system: Color,
    pub accent_error: Color,
    pub accent_success: Color,
    pub accent_running: Color,
    pub accent_skill: Color,
    pub accent_plan: Color,
    pub accent_goal: Color,
    pub accent_verify: Color,

    // Text.
    pub text_primary: Color,
    pub text_secondary: Color,
    pub gray_dim: Color,
    pub gray: Color,
    pub gray_bright: Color,

    // Semantic.
    pub warning: Color,
    pub path: Color,
    pub command: Color,
    pub code: Color,
    pub link: Color,
    /// The `ui.status` stream ("🔍 skeptic approved") — informational, so it is
    /// muted rather than competing with the user's own prompt echo.
    pub status: Color,

    // Syntax highlighting for fenced code.
    pub syn_keyword: Color,
    pub syn_type: Color,
    pub syn_function: Color,
    pub syn_string: Color,
    pub syn_number: Color,
    pub syn_comment: Color,

    // Diffs.
    pub diff_add: Color,
    pub diff_del: Color,
    pub diff_hunk: Color,
    pub diff_context: Color,
    pub diff_gutter: Color,

    // Chrome.
    pub selection: Color,
    /// Background behind a mouse-dragged text selection — a muted, readable tint
    /// (unlike `selection`, which is a foreground accent). Painted on all themes,
    /// including ansi, so a drag-selection is always visible.
    pub selection_bg: Color,
    pub prompt_border: Color,
    pub prompt_border_active: Color,
    /// A subtle band behind a user prompt block (truecolor only; `Reset` on
    /// ansi so nothing paints a background the terminal theme won't match).
    pub band_user: Color,
    /// A sunken panel behind expanded tool output (truecolor only).
    pub panel: Color,
}

impl Theme {
    /// GrokNight-style truecolor dark palette. Neutral gray base, magenta
    /// assistant/thinking accent, blue system, standard green/red/yellow.
    pub const fn dark() -> Self {
        Self {
            accent_user: Color::Rgb(0xc8, 0xc8, 0xc8),
            accent_assistant: Color::Rgb(0xbb, 0x9a, 0xf7),
            accent_thinking: Color::Rgb(0x9d, 0x7c, 0xd8),
            accent_tool: Color::Rgb(0x78, 0x78, 0x78),
            accent_system: Color::Rgb(0x7a, 0xa2, 0xf7),
            accent_error: Color::Rgb(0xf7, 0x76, 0x8e),
            accent_success: Color::Rgb(0x9e, 0xce, 0x6a),
            accent_running: Color::Rgb(0xbb, 0x9a, 0xf7),
            accent_skill: Color::Rgb(0x7a, 0xa2, 0xf7),
            accent_plan: Color::Rgb(0x7d, 0xcf, 0xff),
            accent_goal: Color::Rgb(0xbb, 0x9a, 0xf7),
            accent_verify: Color::Rgb(0x7d, 0xcf, 0xff),
            text_primary: Color::Rgb(0xc0, 0xca, 0xf5),
            text_secondary: Color::Rgb(0x9a, 0xa5, 0xce),
            gray_dim: Color::Rgb(0x56, 0x5f, 0x89),
            gray: Color::Rgb(0x78, 0x7c, 0x99),
            gray_bright: Color::Rgb(0xa9, 0xb1, 0xd6),
            warning: Color::Rgb(0xe0, 0xaf, 0x68),
            path: Color::Rgb(0xff, 0x9e, 0x64),
            command: Color::Rgb(0x7d, 0xcf, 0xff),
            code: Color::Rgb(0x7d, 0xcf, 0xff),
            link: Color::Rgb(0x7a, 0xa2, 0xf7),
            status: Color::Rgb(0x9a, 0xa5, 0xce),
            syn_keyword: Color::Rgb(0xbb, 0x9a, 0xf7),
            syn_type: Color::Rgb(0x2a, 0xc3, 0xde),
            syn_function: Color::Rgb(0x7a, 0xa2, 0xf7),
            syn_string: Color::Rgb(0x9e, 0xce, 0x6a),
            syn_number: Color::Rgb(0xff, 0x9e, 0x64),
            syn_comment: Color::Rgb(0x56, 0x5f, 0x89),
            diff_add: Color::Rgb(0x9e, 0xce, 0x6a),
            diff_del: Color::Rgb(0xf7, 0x76, 0x8e),
            diff_hunk: Color::Rgb(0x7d, 0xcf, 0xff),
            diff_context: Color::Rgb(0x78, 0x7c, 0x99),
            diff_gutter: Color::Rgb(0x56, 0x5f, 0x89),
            selection: Color::Rgb(0x7d, 0xcf, 0xff),
            selection_bg: Color::Rgb(0x2d, 0x3f, 0x76),
            prompt_border: Color::Rgb(0x56, 0x5f, 0x89),
            prompt_border_active: Color::Rgb(0x7d, 0xcf, 0xff),
            band_user: Color::Rgb(0x1f, 0x23, 0x35),
            panel: Color::Rgb(0x1a, 0x1b, 0x26),
        }
    }

    /// A light truecolor palette for bright terminal backgrounds.
    pub const fn light() -> Self {
        Self {
            accent_user: Color::Rgb(0x34, 0x3b, 0x58),
            accent_assistant: Color::Rgb(0x7a, 0x4c, 0xc9),
            accent_thinking: Color::Rgb(0x8a, 0x5c, 0xd9),
            accent_tool: Color::Rgb(0x8c, 0x8c, 0x8c),
            accent_system: Color::Rgb(0x2e, 0x5c, 0xc9),
            accent_error: Color::Rgb(0xc0, 0x36, 0x4e),
            accent_success: Color::Rgb(0x38, 0x7a, 0x2c),
            accent_running: Color::Rgb(0x7a, 0x4c, 0xc9),
            accent_skill: Color::Rgb(0x2e, 0x5c, 0xc9),
            accent_plan: Color::Rgb(0x16, 0x7a, 0xa6),
            accent_goal: Color::Rgb(0x7a, 0x4c, 0xc9),
            accent_verify: Color::Rgb(0x16, 0x7a, 0xa6),
            text_primary: Color::Rgb(0x2a, 0x2e, 0x3a),
            text_secondary: Color::Rgb(0x50, 0x56, 0x6a),
            gray_dim: Color::Rgb(0x9a, 0xa0, 0xb0),
            gray: Color::Rgb(0x70, 0x76, 0x88),
            gray_bright: Color::Rgb(0x40, 0x46, 0x58),
            warning: Color::Rgb(0xa6, 0x6a, 0x00),
            path: Color::Rgb(0xc0, 0x54, 0x1a),
            command: Color::Rgb(0x16, 0x6a, 0xa6),
            code: Color::Rgb(0x16, 0x6a, 0xa6),
            link: Color::Rgb(0x2e, 0x5c, 0xc9),
            status: Color::Rgb(0x50, 0x56, 0x6a),
            syn_keyword: Color::Rgb(0x7a, 0x4c, 0xc9),
            syn_type: Color::Rgb(0x0f, 0x6b, 0x8a),
            syn_function: Color::Rgb(0x2e, 0x5c, 0xc9),
            syn_string: Color::Rgb(0x38, 0x7a, 0x2c),
            syn_number: Color::Rgb(0xc0, 0x54, 0x1a),
            syn_comment: Color::Rgb(0x8a, 0x90, 0xa0),
            diff_add: Color::Rgb(0x38, 0x7a, 0x2c),
            diff_del: Color::Rgb(0xc0, 0x36, 0x4e),
            diff_hunk: Color::Rgb(0x16, 0x6a, 0xa6),
            diff_context: Color::Rgb(0x70, 0x76, 0x88),
            diff_gutter: Color::Rgb(0x9a, 0xa0, 0xb0),
            selection: Color::Rgb(0x16, 0x6a, 0xa6),
            selection_bg: Color::Rgb(0xc6, 0xdd, 0xf7),
            prompt_border: Color::Rgb(0x9a, 0xa0, 0xb0),
            prompt_border_active: Color::Rgb(0x16, 0x6a, 0xa6),
            band_user: Color::Rgb(0xec, 0xee, 0xf5),
            panel: Color::Rgb(0xf0, 0xf2, 0xf8),
        }
    }

    /// Named-ANSI palette: reproduces hi's historical look and respects the
    /// user's own terminal colors. Backgrounds are `Reset` so nothing paints a
    /// band a terminal theme won't match. This is the non-truecolor fallback.
    pub const fn ansi() -> Self {
        Self {
            accent_user: Color::Blue,
            accent_assistant: Color::Magenta,
            accent_thinking: Color::DarkGray,
            accent_tool: Color::Cyan,
            accent_system: Color::Blue,
            accent_error: Color::Red,
            accent_success: Color::Green,
            accent_running: Color::Cyan,
            accent_skill: Color::Blue,
            accent_plan: Color::Cyan,
            accent_goal: Color::Magenta,
            accent_verify: Color::Cyan,
            text_primary: Color::Reset,
            text_secondary: Color::Gray,
            gray_dim: Color::DarkGray,
            gray: Color::Gray,
            gray_bright: Color::White,
            warning: Color::Yellow,
            path: Color::Cyan,
            command: Color::Cyan,
            code: Color::Cyan,
            link: Color::Blue,
            status: Color::Blue,
            syn_keyword: Color::Magenta,
            syn_type: Color::Cyan,
            syn_function: Color::Blue,
            syn_string: Color::Green,
            syn_number: Color::Yellow,
            syn_comment: Color::DarkGray,
            diff_add: Color::Green,
            diff_del: Color::Red,
            diff_hunk: Color::Cyan,
            diff_context: Color::DarkGray,
            diff_gutter: Color::DarkGray,
            selection: Color::Cyan,
            selection_bg: Color::Blue,
            prompt_border: Color::DarkGray,
            prompt_border_active: Color::Cyan,
            band_user: Color::Reset,
            panel: Color::Reset,
        }
    }

    /// Whether this theme paints real backgrounds (truecolor) or leaves them at
    /// the terminal default (ansi). Renderers use this to skip band/panel fills
    /// that would look wrong against an unknown terminal background.
    #[allow(dead_code)]
    pub fn paints_backgrounds(&self) -> bool {
        !matches!(self.band_user, Color::Reset)
    }
}

/// Which palette the user selected, decoupled from the resolved [`Theme`] so
/// `auto` can re-resolve when the OS appearance changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ThemeMode {
    Dark,
    Light,
    /// Named ANSI colors that respect the user's own terminal theme.
    Ansi,
    /// Follow the OS light/dark appearance (falls back to a truecolor-aware
    /// default when the OS can't be queried).
    Auto,
}

impl ThemeMode {
    /// Parse a `/theme <name>` / `HI_THEME` value. `None` for an unknown value.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "dark" => Some(Self::Dark),
            "light" => Some(Self::Light),
            "ansi" | "none" => Some(Self::Ansi),
            "auto" | "system" => Some(Self::Auto),
            _ => None,
        }
    }

    /// A short label for the status line / picker.
    pub fn label(self) -> &'static str {
        match self {
            Self::Dark => "dark",
            Self::Light => "light",
            Self::Ansi => "ansi",
            Self::Auto => "auto",
        }
    }

    /// The next mode when cycling with a bare `/theme`.
    pub fn next(self) -> Self {
        match self {
            Self::Dark => Self::Light,
            Self::Light => Self::Ansi,
            Self::Ansi => Self::Auto,
            Self::Auto => Self::Dark,
        }
    }

    /// Resolve this mode to a concrete palette, consulting the OS appearance for
    /// `Auto`.
    fn resolve(self) -> Theme {
        match self {
            Self::Dark => Theme::dark(),
            Self::Light => Theme::light(),
            Self::Ansi => Theme::ansi(),
            Self::Auto => match os_appearance() {
                Some(OsAppearance::Dark) => Theme::dark(),
                Some(OsAppearance::Light) => Theme::light(),
                // OS can't be queried: the designed palette on a truecolor
                // terminal, the terminal-respecting ANSI look otherwise.
                None if terminal_supports_truecolor() => Theme::dark(),
                None => Theme::ansi(),
            },
        }
    }
}

/// The OS's light/dark appearance, when it can be determined.
///
/// `#[allow(dead_code)]`: only the macOS `os_appearance` constructs these; on
/// platforms whose detector always returns `None` the variants are matched but
/// never built, which `-D dead-code` would otherwise reject.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OsAppearance {
    Dark,
    Light,
}

/// Query the OS light/dark appearance. macOS reads `AppleInterfaceStyle`
/// (present and "Dark" in dark mode; absent in light mode). Other platforms
/// return `None` for now (Linux XDG portal / Windows registry are a follow-up).
#[cfg(target_os = "macos")]
fn os_appearance() -> Option<OsAppearance> {
    let out = std::process::Command::new("defaults")
        .args(["read", "-g", "AppleInterfaceStyle"])
        .output()
        .ok()?;
    if out.status.success() && String::from_utf8_lossy(&out.stdout).trim() == "Dark" {
        Some(OsAppearance::Dark)
    } else {
        // A non-zero exit means the key is absent → light mode.
        Some(OsAppearance::Light)
    }
}

#[cfg(not(target_os = "macos"))]
fn os_appearance() -> Option<OsAppearance> {
    None
}

/// Resolve the initial mode from `HI_THEME` (default `auto`).
fn initial_mode() -> ThemeMode {
    std::env::var("HI_THEME")
        .ok()
        .and_then(|v| ThemeMode::parse(&v))
        .unwrap_or(ThemeMode::Auto)
}

/// Best-effort truecolor detection. `COLORTERM=truecolor|24bit` is the standard
/// signal; a few terminals advertise via `TERM`. Conservative: unknown → false
/// (fall back to the terminal-respecting ANSI palette).
fn terminal_supports_truecolor() -> bool {
    if let Ok(colorterm) = std::env::var("COLORTERM") {
        let c = colorterm.to_ascii_lowercase();
        if c.contains("truecolor") || c.contains("24bit") {
            return true;
        }
    }
    if let Ok(term) = std::env::var("TERM") {
        let t = term.to_ascii_lowercase();
        if t.contains("truecolor") || t.contains("24bit") || t == "xterm-kitty" {
            return true;
        }
    }
    // Modern terminal emulators that always support truecolor.
    matches!(
        std::env::var("TERM_PROGRAM").ok().as_deref(),
        Some("iTerm.app") | Some("WezTerm") | Some("ghostty") | Some("vscode")
    )
}

/// The selected mode and its currently-resolved palette, behind one lock so a
/// `/theme` switch or an OS-appearance change updates both atomically.
struct ThemeState {
    mode: ThemeMode,
    theme: Theme,
}

static STATE: OnceLock<RwLock<ThemeState>> = OnceLock::new();

fn cell() -> &'static RwLock<ThemeState> {
    STATE.get_or_init(|| {
        let mode = initial_mode();
        RwLock::new(ThemeState {
            mode,
            theme: mode.resolve(),
        })
    })
}

/// The active theme. Cheap `Copy`; read freely on every render.
pub fn theme() -> Theme {
    cell().read().unwrap().theme
}

/// The active mode (for the status line and the `/theme` cycle).
pub fn mode() -> ThemeMode {
    cell().read().unwrap().mode
}

/// Switch to `mode` and re-resolve its palette. Returns the resolved mode so
/// the caller can report it.
pub fn set_mode(mode: ThemeMode) {
    let mut state = cell().write().unwrap();
    state.mode = mode;
    state.theme = mode.resolve();
}

/// Cycle to the next mode (bare `/theme`), returning it for display.
pub fn cycle_mode() -> ThemeMode {
    let next = mode().next();
    set_mode(next);
    next
}

/// Re-resolve an `Auto` theme against the current OS appearance. Returns `true`
/// if the palette changed. A no-op for fixed modes. The caller (event loop)
/// rate-limits how often this runs since it may spawn a subprocess.
pub fn poll_auto_appearance() -> bool {
    let mut state = cell().write().unwrap();
    if state.mode != ThemeMode::Auto {
        return false;
    }
    let resolved = ThemeMode::Auto.resolve();
    // Compare a cheap discriminator (band_user is distinct per palette).
    if resolved.band_user != state.theme.band_user {
        state.theme = resolved;
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ansi_theme_leaves_backgrounds_at_terminal_default() {
        let t = Theme::ansi();
        assert!(!t.paints_backgrounds());
        assert_eq!(t.band_user, Color::Reset);
        assert_eq!(t.panel, Color::Reset);
    }

    #[test]
    fn truecolor_themes_paint_backgrounds() {
        assert!(Theme::dark().paints_backgrounds());
        assert!(Theme::light().paints_backgrounds());
    }

    #[test]
    fn every_role_is_distinct_enough_in_dark() {
        // The three most-overloaded historical roles (user, tool, status) must
        // not collapse to the same color in the designed palette.
        let t = Theme::dark();
        assert_ne!(t.accent_user, t.accent_tool);
        assert_ne!(t.accent_user, t.status);
        assert_ne!(t.accent_tool, t.accent_goal);
    }

    #[test]
    fn mode_parse_and_cycle() {
        assert_eq!(ThemeMode::parse("dark"), Some(ThemeMode::Dark));
        assert_eq!(ThemeMode::parse("LIGHT"), Some(ThemeMode::Light));
        assert_eq!(ThemeMode::parse("system"), Some(ThemeMode::Auto));
        assert_eq!(ThemeMode::parse("none"), Some(ThemeMode::Ansi));
        assert_eq!(ThemeMode::parse("nope"), None);
        // Cycle visits every mode and returns to the start.
        let mut m = ThemeMode::Dark;
        let mut seen = std::collections::HashSet::new();
        for _ in 0..4 {
            seen.insert(m);
            m = m.next();
        }
        assert_eq!(m, ThemeMode::Dark, "cycle is a 4-loop");
        assert_eq!(seen.len(), 4, "cycle visits all modes");
    }

    #[test]
    fn each_mode_resolves_to_the_expected_palette() {
        assert_eq!(ThemeMode::Dark.resolve().band_user, Theme::dark().band_user);
        assert_eq!(
            ThemeMode::Light.resolve().band_user,
            Theme::light().band_user
        );
        assert!(!ThemeMode::Ansi.resolve().paints_backgrounds());
    }

    // Note: the runtime `set_mode`/`theme` global is deliberately not unit-
    // tested here — mutating the process-wide theme would race color-asserting
    // render tests running in parallel. The pure constructors and mode logic
    // above cover the palette; the global is exercised by the app at startup.
}
