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
//! render; [`set_theme`] switches at runtime (e.g. a future `/theme`).

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

    // Diffs.
    pub diff_add: Color,
    pub diff_del: Color,
    pub diff_hunk: Color,
    pub diff_context: Color,
    pub diff_gutter: Color,

    // Chrome.
    pub selection: Color,
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
            diff_add: Color::Rgb(0x9e, 0xce, 0x6a),
            diff_del: Color::Rgb(0xf7, 0x76, 0x8e),
            diff_hunk: Color::Rgb(0x7d, 0xcf, 0xff),
            diff_context: Color::Rgb(0x78, 0x7c, 0x99),
            diff_gutter: Color::Rgb(0x56, 0x5f, 0x89),
            selection: Color::Rgb(0x7d, 0xcf, 0xff),
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
            diff_add: Color::Rgb(0x38, 0x7a, 0x2c),
            diff_del: Color::Rgb(0xc0, 0x36, 0x4e),
            diff_hunk: Color::Rgb(0x16, 0x6a, 0xa6),
            diff_context: Color::Rgb(0x70, 0x76, 0x88),
            diff_gutter: Color::Rgb(0x9a, 0xa0, 0xb0),
            selection: Color::Rgb(0x16, 0x6a, 0xa6),
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
            diff_add: Color::Green,
            diff_del: Color::Red,
            diff_hunk: Color::Cyan,
            diff_context: Color::DarkGray,
            diff_gutter: Color::DarkGray,
            selection: Color::Cyan,
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

/// Resolve the active theme from `HI_THEME` and terminal capability.
fn resolve_from_env() -> Theme {
    match std::env::var("HI_THEME").ok().as_deref().map(str::trim) {
        Some("dark") => Theme::dark(),
        Some("light") => Theme::light(),
        Some("ansi") | Some("none") => Theme::ansi(),
        // `auto` (and unset): the designed palette on a truecolor terminal,
        // the terminal-respecting ANSI look otherwise.
        _ => {
            if terminal_supports_truecolor() {
                Theme::dark()
            } else {
                Theme::ansi()
            }
        }
    }
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

static THEME: OnceLock<RwLock<Theme>> = OnceLock::new();

fn cell() -> &'static RwLock<Theme> {
    THEME.get_or_init(|| RwLock::new(resolve_from_env()))
}

/// The active theme. Cheap `Copy`; read freely on every render.
pub fn theme() -> Theme {
    *cell().read().unwrap()
}

/// Switch the active theme at runtime (wired to a future `/theme` command).
#[allow(dead_code)]
pub fn set_theme(theme: Theme) {
    *cell().write().unwrap() = theme;
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

    // Note: the runtime `set_theme`/`theme` global is deliberately not unit-
    // tested here — mutating the process-wide theme would race color-asserting
    // render tests running in parallel. The pure `Theme` constructors above
    // cover the palette; the global is exercised by the app at startup.
}
