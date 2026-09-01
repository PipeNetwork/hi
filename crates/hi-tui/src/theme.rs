//! The TUI color theme: a named slot vocabulary so every rendered color maps to
//! a *role* (user, tool, error, running, …) rather than a hardcoded ANSI name.
//!
//! Palettes match grok-build's pager:
//! - `groknight` (`dark`) — default: near-black gray base, Tokyo Night accents.
//! - `grokday` (`light`) — grok-build's light counterpart.
//! - `tokyonight` — blue-tinted Storm (hi's previous default).
//! - `oscura-midnight` / `rosepine-moon` — the other grok-build truecolor looks.
//! - `ansi` — named ANSI colors that respect the user's terminal theme.
//!
//! Selection: `HI_THEME` (canonical names above, plus `auto`; default
//! `groknight`). `auto` picks `groknight`/`grokday` from the OS appearance.

use std::sync::{OnceLock, RwLock};

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, BorderType};

/// Semantic visual roles shared by transcript, dashboard, picker, and status
/// renderers. Renderers choose a role; palettes decide the actual color.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum UiTone {
    Muted,
    Active,
    Info,
    Success,
    Warning,
    Error,
    User,
    Assistant,
    Tool,
}

/// The style family for one bordered surface or status row.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ChromeStyles {
    pub(crate) border: Style,
    pub(crate) title: Style,
    pub(crate) body: Style,
    pub(crate) hint: Style,
    pub(crate) selected: Style,
}

const fn rgb(r: u8, g: u8, b: u8) -> Color {
    Color::Rgb(r, g, b)
}

/// Every color role the TUI draws. One field per semantic slot; renderers ask
/// for a role, never a raw `Color`, so the whole look restyles from one place.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub struct Theme {
    /// Full-screen fill. `Reset` on ansi so the terminal background shows through.
    pub bg_base: Color,
    pub bg_highlight: Color,

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
    /// Model name inlined in the prompt's bottom divider.
    pub accent_model: Color,

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
    /// Insert-line background band (truecolor; `Reset` on ansi).
    pub diff_add_bg: Color,
    /// Delete-line background band (truecolor; `Reset` on ansi).
    pub diff_del_bg: Color,

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
    pub(crate) const fn tone_color(self, tone: UiTone) -> Color {
        match tone {
            UiTone::Muted => self.gray_dim,
            UiTone::Active => self.accent_running,
            UiTone::Info => self.accent_system,
            UiTone::Success => self.accent_success,
            UiTone::Warning => self.warning,
            UiTone::Error => self.accent_error,
            UiTone::User => self.accent_user,
            UiTone::Assistant => self.accent_assistant,
            UiTone::Tool => self.accent_tool,
        }
    }

    /// Resolve a semantic tone to the shared chrome styles used by all views.
    pub(crate) fn chrome(self, tone: UiTone) -> ChromeStyles {
        let accent = self.tone_color(tone);
        ChromeStyles {
            border: Style::default().fg(accent),
            title: Style::default().fg(accent).add_modifier(Modifier::BOLD),
            body: Style::default().fg(self.text_primary),
            hint: Style::default().fg(self.gray_dim),
            selected: Style::default()
                .fg(self.text_primary)
                .bg(self.selection_bg)
                .add_modifier(Modifier::BOLD),
        }
    }

    /// Build the common rounded panel shell. Callers can still add titles or
    /// bottom hints after this, but border shape and semantic color stay shared.
    pub(crate) fn panel_block(self, title: impl Into<String>, tone: UiTone) -> Block<'static> {
        let chrome = self.chrome(tone);
        let mut block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(chrome.border)
            .title(Line::styled(title.into(), chrome.title));
        if self.paints_backgrounds() {
            block = block.style(Style::default().bg(self.bg_base).fg(self.text_primary));
        }
        block
    }

    pub(crate) fn input_border(self, active: bool) -> Style {
        Style::default().fg(if active {
            self.prompt_border_active
        } else {
            self.prompt_border
        })
    }

    /// GrokNight — grok-build's default: neutral #141414 base, Tokyo Night accents.
    pub const fn groknight() -> Self {
        Self {
            bg_base: rgb(20, 20, 20),
            bg_highlight: rgb(36, 36, 36),
            accent_user: rgb(0xc8, 0xc8, 0xc8),
            accent_assistant: rgb(0xbb, 0x9a, 0xf7),
            accent_thinking: rgb(0xbb, 0x9a, 0xf7),
            accent_tool: rgb(0x78, 0x78, 0x78),
            accent_system: rgb(0x7a, 0xa2, 0xf7),
            accent_error: rgb(0xf7, 0x76, 0x8e),
            accent_success: rgb(0x9e, 0xce, 0x6a),
            accent_running: rgb(0xbb, 0x9a, 0xf7),
            accent_skill: rgb(0x7a, 0xa2, 0xf7),
            accent_plan: rgb(0xff, 0xdb, 0x8d),
            accent_goal: rgb(0xbb, 0x9a, 0xf7),
            accent_verify: rgb(0xbb, 0x9a, 0xf7),
            accent_model: rgb(0x1a, 0xbc, 0x9c),
            text_primary: rgb(0xe1, 0xe1, 0xe1),
            text_secondary: rgb(0xc8, 0xc8, 0xc8),
            gray_dim: rgb(0x58, 0x58, 0x58),
            gray: rgb(0x6c, 0x6c, 0x6c),
            gray_bright: rgb(0x78, 0x78, 0x78),
            warning: rgb(0xe0, 0xaf, 0x68),
            path: rgb(0xff, 0x9e, 0x64),
            command: rgb(0xe0, 0xaf, 0x68),
            code: rgb(0x3a, 0x95, 0xab),
            link: rgb(0x7a, 0xa6, 0xda),
            status: rgb(0x6c, 0x6c, 0x6c),
            syn_keyword: rgb(0xbb, 0x9a, 0xf7),
            syn_type: rgb(0x2a, 0xc3, 0xde),
            syn_function: rgb(0x7a, 0xa2, 0xf7),
            syn_string: rgb(0x9e, 0xce, 0x6a),
            syn_number: rgb(0xff, 0x9e, 0x64),
            syn_comment: rgb(0x6c, 0x6c, 0x6c),
            diff_add: rgb(0x9e, 0xce, 0x6a),
            diff_del: rgb(0xf7, 0x76, 0x8e),
            diff_hunk: rgb(0x7d, 0xcf, 0xff),
            diff_context: rgb(0x78, 0x78, 0x78),
            diff_gutter: rgb(0x58, 0x58, 0x58),
            diff_add_bg: rgb(6, 56, 6),
            diff_del_bg: rgb(66, 14, 20),
            selection: rgb(0x7d, 0xcf, 0xff),
            selection_bg: rgb(0x36, 0x36, 0x36),
            prompt_border: rgb(0x32, 0x32, 0x37),
            prompt_border_active: rgb(0x50, 0x50, 0x58),
            band_user: rgb(0x24, 0x24, 0x24),
            panel: rgb(0x1c, 0x1c, 0x1c),
        }
    }

    /// Alias used by `/theme dark` and historical call sites.
    pub const fn dark() -> Self {
        Self::groknight()
    }

    /// GrokDay — grok-build's light counterpart: neutral gray, deepened accents.
    pub const fn grokday() -> Self {
        Self {
            bg_base: rgb(238, 238, 238),
            bg_highlight: rgb(222, 222, 222),
            accent_user: rgb(0x44, 0x44, 0x44),
            accent_assistant: rgb(0x7d, 0x4b, 0xc6),
            accent_thinking: rgb(0x7d, 0x4b, 0xc6),
            accent_tool: rgb(0x62, 0x62, 0x62),
            accent_system: rgb(0x2f, 0x64, 0xd2),
            accent_error: rgb(0xcd, 0x30, 0x48),
            accent_success: rgb(0x37, 0x8e, 0x23),
            accent_running: rgb(0x7d, 0x4b, 0xc6),
            accent_skill: rgb(0x2f, 0x64, 0xd2),
            accent_plan: rgb(0xa8, 0x78, 0x0a),
            accent_goal: rgb(0x7d, 0x4b, 0xc6),
            accent_verify: rgb(0x78, 0x50, 0xa0),
            accent_model: rgb(0x0a, 0x8e, 0x70),
            text_primary: rgb(0x26, 0x26, 0x26),
            text_secondary: rgb(0x44, 0x44, 0x44),
            gray_dim: rgb(0xa5, 0xa5, 0xa5),
            gray: rgb(0x76, 0x76, 0x76),
            gray_bright: rgb(0x62, 0x62, 0x62),
            warning: rgb(0xa2, 0x76, 0x12),
            path: rgb(0xc3, 0x69, 0x1e),
            command: rgb(0xa2, 0x76, 0x12),
            code: rgb(0x0f, 0x87, 0xa2),
            link: rgb(0x2f, 0x64, 0xd2),
            status: rgb(0x76, 0x76, 0x76),
            syn_keyword: rgb(0x7d, 0x4b, 0xc6),
            syn_type: rgb(0x0f, 0x87, 0xa2),
            syn_function: rgb(0x2f, 0x64, 0xd2),
            syn_string: rgb(0x37, 0x8e, 0x23),
            syn_number: rgb(0xc3, 0x69, 0x1e),
            syn_comment: rgb(0x76, 0x76, 0x76),
            diff_add: rgb(0x37, 0x8e, 0x23),
            diff_del: rgb(0xcd, 0x30, 0x48),
            diff_hunk: rgb(0x00, 0x82, 0xaa),
            diff_context: rgb(0x62, 0x62, 0x62),
            diff_gutter: rgb(0xa5, 0xa5, 0xa5),
            diff_add_bg: rgb(218, 242, 220),
            diff_del_bg: rgb(245, 218, 222),
            selection: rgb(0x00, 0x82, 0xaa),
            selection_bg: rgb(0xc6, 0xc6, 0xc6),
            prompt_border: rgb(0xc8, 0xc8, 0xcd),
            prompt_border_active: rgb(0xa5, 0xa5, 0xaf),
            band_user: rgb(0xde, 0xde, 0xde),
            panel: rgb(0xe4, 0xe4, 0xe4),
        }
    }

    /// Alias used by `/theme light` and historical call sites.
    pub const fn light() -> Self {
        Self::grokday()
    }

    /// Tokyo Night Storm — hi's previous default, kept as a named option.
    pub const fn tokyonight() -> Self {
        Self {
            bg_base: rgb(0x24, 0x28, 0x3b),
            bg_highlight: rgb(0x29, 0x2e, 0x42),
            accent_user: rgb(0xc8, 0xc8, 0xc8),
            accent_assistant: rgb(0xbb, 0x9a, 0xf7),
            accent_thinking: rgb(0x9d, 0x7c, 0xd8),
            accent_tool: rgb(0x78, 0x78, 0x78),
            accent_system: rgb(0x7a, 0xa2, 0xf7),
            accent_error: rgb(0xf7, 0x76, 0x8e),
            accent_success: rgb(0x9e, 0xce, 0x6a),
            accent_running: rgb(0xbb, 0x9a, 0xf7),
            accent_skill: rgb(0x7a, 0xa2, 0xf7),
            accent_plan: rgb(0x7d, 0xcf, 0xff),
            accent_goal: rgb(0xbb, 0x9a, 0xf7),
            accent_verify: rgb(0x7d, 0xcf, 0xff),
            accent_model: rgb(0x1a, 0xbc, 0x9c),
            text_primary: rgb(0xc0, 0xca, 0xf5),
            text_secondary: rgb(0x9a, 0xa5, 0xce),
            gray_dim: rgb(0x56, 0x5f, 0x89),
            gray: rgb(0x78, 0x7c, 0x99),
            gray_bright: rgb(0xa9, 0xb1, 0xd6),
            warning: rgb(0xe0, 0xaf, 0x68),
            path: rgb(0xff, 0x9e, 0x64),
            command: rgb(0x7d, 0xcf, 0xff),
            code: rgb(0x7d, 0xcf, 0xff),
            link: rgb(0x7a, 0xa2, 0xf7),
            status: rgb(0x9a, 0xa5, 0xce),
            syn_keyword: rgb(0xbb, 0x9a, 0xf7),
            syn_type: rgb(0x2a, 0xc3, 0xde),
            syn_function: rgb(0x7a, 0xa2, 0xf7),
            syn_string: rgb(0x9e, 0xce, 0x6a),
            syn_number: rgb(0xff, 0x9e, 0x64),
            syn_comment: rgb(0x56, 0x5f, 0x89),
            diff_add: rgb(0x9e, 0xce, 0x6a),
            diff_del: rgb(0xf7, 0x76, 0x8e),
            diff_hunk: rgb(0x7d, 0xcf, 0xff),
            diff_context: rgb(0x78, 0x7c, 0x99),
            diff_gutter: rgb(0x56, 0x5f, 0x89),
            diff_add_bg: rgb(15, 65, 20),
            diff_del_bg: rgb(85, 15, 20),
            selection: rgb(0x7d, 0xcf, 0xff),
            selection_bg: rgb(0x28, 0x34, 0x57),
            prompt_border: rgb(0x3c, 0x4b, 0x78),
            prompt_border_active: rgb(0x4b, 0x5c, 0x8c),
            band_user: rgb(0x1f, 0x23, 0x35),
            panel: rgb(0x1a, 0x1b, 0x26),
        }
    }

    /// Oscura Midnight — grok-build's deep purple-tinted dark palette.
    pub const fn oscura() -> Self {
        Self {
            bg_base: rgb(3, 3, 4),
            bg_highlight: rgb(15, 18, 22),
            accent_user: rgb(0xc4, 0xa7, 0xe7),
            accent_assistant: rgb(0x9b, 0x7e, 0xce),
            accent_thinking: rgb(0x81, 0x86, 0x8f),
            accent_tool: rgb(0x5e, 0x64, 0x6c),
            accent_system: rgb(0x7d, 0xcf, 0xdf),
            accent_error: rgb(0xdc, 0x5a, 0x64),
            accent_success: rgb(0x50, 0xb4, 0x8c),
            accent_running: rgb(0x6e, 0x5a, 0x9a),
            accent_skill: rgb(0x9b, 0x7e, 0xce),
            accent_plan: rgb(0xeb, 0xd9, 0x6e),
            accent_goal: rgb(0x9b, 0x7e, 0xce),
            accent_verify: rgb(0x9b, 0x7e, 0xce),
            accent_model: rgb(0x7d, 0xcf, 0xdf),
            text_primary: rgb(0xe4, 0xe4, 0xe4),
            text_secondary: rgb(0xbe, 0xbe, 0xbe),
            gray_dim: rgb(0x5e, 0x64, 0x6c),
            gray: rgb(0x81, 0x86, 0x8f),
            gray_bright: rgb(0xbe, 0xbe, 0xbe),
            warning: rgb(0xeb, 0xd9, 0x6e),
            path: rgb(0xf1, 0xbd, 0x00),
            command: rgb(0xeb, 0xd9, 0x6e),
            code: rgb(0x7d, 0xcf, 0xdf),
            link: rgb(0x7d, 0xcf, 0xdf),
            status: rgb(0x81, 0x86, 0x8f),
            syn_keyword: rgb(0xc4, 0xa7, 0xe7),
            syn_type: rgb(0x7d, 0xcf, 0xdf),
            syn_function: rgb(0x9b, 0x7e, 0xce),
            syn_string: rgb(0x50, 0xb4, 0x8c),
            syn_number: rgb(0xf1, 0xbd, 0x00),
            syn_comment: rgb(0x5e, 0x64, 0x6c),
            diff_add: rgb(0x50, 0xb4, 0x8c),
            diff_del: rgb(0xdc, 0x5a, 0x64),
            diff_hunk: rgb(0x7d, 0xcf, 0xdf),
            diff_context: rgb(0x81, 0x86, 0x8f),
            diff_gutter: rgb(0x5e, 0x64, 0x6c),
            diff_add_bg: rgb(10, 35, 30),
            diff_del_bg: rgb(45, 15, 25),
            selection: rgb(0xc4, 0xa7, 0xe7),
            selection_bg: rgb(0x24, 0x20, 0x34),
            prompt_border: rgb(0x24, 0x20, 0x34),
            prompt_border_active: rgb(0x34, 0x30, 0x48),
            band_user: rgb(0x12, 0x10, 0x1c),
            panel: rgb(0x04, 0x05, 0x07),
        }
    }

    /// Rose Pine Moon — grok-build's warmer purple dark palette.
    pub const fn rosepine() -> Self {
        Self {
            bg_base: rgb(35, 33, 54),
            bg_highlight: rgb(57, 53, 82),
            accent_user: rgb(0xe0, 0xde, 0xf4),
            accent_assistant: rgb(0xc4, 0xa7, 0xe7),
            accent_thinking: rgb(0x6e, 0x6a, 0x86),
            accent_tool: rgb(0x90, 0x8c, 0xaa),
            accent_system: rgb(0x3e, 0x8f, 0xb0),
            accent_error: rgb(0xeb, 0x6f, 0x92),
            accent_success: rgb(0x9c, 0xcf, 0xd8),
            accent_running: rgb(0x6e, 0x6a, 0x86),
            accent_skill: rgb(0x90, 0x8c, 0xaa),
            accent_plan: rgb(0xf6, 0xc1, 0x77),
            accent_goal: rgb(0xc4, 0xa7, 0xe7),
            accent_verify: rgb(0x3e, 0x8f, 0xb0),
            accent_model: rgb(0x3e, 0x8f, 0xb0),
            text_primary: rgb(0xe0, 0xde, 0xf4),
            text_secondary: rgb(0x90, 0x8c, 0xaa),
            gray_dim: rgb(0x44, 0x41, 0x5a),
            gray: rgb(0x6e, 0x6a, 0x86),
            gray_bright: rgb(0x90, 0x8c, 0xaa),
            warning: rgb(0xf6, 0xc1, 0x77),
            path: rgb(0xea, 0x9a, 0x97),
            command: rgb(0xf6, 0xc1, 0x77),
            code: rgb(0x9c, 0xcf, 0xd8),
            link: rgb(0x3e, 0x8f, 0xb0),
            status: rgb(0x6e, 0x6a, 0x86),
            syn_keyword: rgb(0xc4, 0xa7, 0xe7),
            syn_type: rgb(0x9c, 0xcf, 0xd8),
            syn_function: rgb(0x3e, 0x8f, 0xb0),
            syn_string: rgb(0x9c, 0xcf, 0xd8),
            syn_number: rgb(0xf6, 0xc1, 0x77),
            syn_comment: rgb(0x6e, 0x6a, 0x86),
            diff_add: rgb(0x9c, 0xcf, 0xd8),
            diff_del: rgb(0xeb, 0x6f, 0x92),
            diff_hunk: rgb(0x3e, 0x8f, 0xb0),
            diff_context: rgb(0x90, 0x8c, 0xaa),
            diff_gutter: rgb(0x44, 0x41, 0x5a),
            diff_add_bg: rgb(25, 45, 55),
            diff_del_bg: rgb(55, 30, 40),
            selection: rgb(0xc4, 0xa7, 0xe7),
            selection_bg: rgb(0x44, 0x41, 0x5a),
            prompt_border: rgb(0x44, 0x41, 0x5a),
            prompt_border_active: rgb(0x56, 0x52, 0x6e),
            band_user: rgb(0x2a, 0x27, 0x3f),
            panel: rgb(0x2a, 0x27, 0x3f),
        }
    }

    /// Named-ANSI palette: respects the user's own terminal colors. Backgrounds
    /// are `Reset` so nothing paints a band a terminal theme won't match.
    pub const fn ansi() -> Self {
        Self {
            bg_base: Color::Reset,
            bg_highlight: Color::Reset,
            accent_user: Color::LightBlue,
            accent_assistant: Color::Magenta,
            accent_thinking: Color::DarkGray,
            accent_tool: Color::Cyan,
            accent_system: Color::LightBlue,
            accent_error: Color::LightRed,
            accent_success: Color::Green,
            accent_running: Color::Cyan,
            accent_skill: Color::LightBlue,
            accent_plan: Color::Cyan,
            accent_goal: Color::Magenta,
            accent_verify: Color::Cyan,
            accent_model: Color::Cyan,
            text_primary: Color::Reset,
            text_secondary: Color::Gray,
            gray_dim: Color::DarkGray,
            gray: Color::Gray,
            gray_bright: Color::White,
            warning: Color::Yellow,
            path: Color::Cyan,
            command: Color::Cyan,
            code: Color::Cyan,
            link: Color::LightBlue,
            status: Color::LightBlue,
            syn_keyword: Color::Magenta,
            syn_type: Color::Cyan,
            syn_function: Color::LightBlue,
            syn_string: Color::Green,
            syn_number: Color::Yellow,
            syn_comment: Color::DarkGray,
            diff_add: Color::Green,
            diff_del: Color::LightRed,
            diff_hunk: Color::Cyan,
            diff_context: Color::DarkGray,
            diff_gutter: Color::DarkGray,
            diff_add_bg: Color::Reset,
            diff_del_bg: Color::Reset,
            selection: Color::Cyan,
            selection_bg: Color::Blue,
            prompt_border: Color::DarkGray,
            prompt_border_active: Color::Gray,
            band_user: Color::Reset,
            panel: Color::Reset,
        }
    }

    /// Whether this theme paints real backgrounds (truecolor) or leaves them at
    /// the terminal default (ansi). Renderers use this to skip band/panel fills
    /// that would look wrong against an unknown terminal background.
    pub fn paints_backgrounds(&self) -> bool {
        !matches!(self.bg_base, Color::Reset)
    }
}

/// Which palette the user selected, decoupled from the resolved [`Theme`] so
/// `auto` can re-resolve when the OS appearance changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ThemeMode {
    /// GrokNight (canonical grok-build dark). `/theme dark` is an alias.
    Dark,
    /// GrokDay (canonical grok-build light). `/theme light` is an alias.
    Light,
    TokyoNight,
    Oscura,
    RosePine,
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
            "dark" | "groknight" | "grok-night" => Some(Self::Dark),
            "light" | "grokday" | "grok-day" | "day" => Some(Self::Light),
            "tokyonight" | "tokyo-night" | "tokyo" => Some(Self::TokyoNight),
            "oscura" | "oscura-midnight" => Some(Self::Oscura),
            "rosepine" | "rose-pine" | "rosepine-moon" | "rose-pine-moon" => Some(Self::RosePine),
            "ansi" | "none" => Some(Self::Ansi),
            "auto" | "system" => Some(Self::Auto),
            _ => None,
        }
    }

    /// A short label for the status line / picker.
    pub fn label(self) -> &'static str {
        match self {
            Self::Dark => "groknight",
            Self::Light => "grokday",
            Self::TokyoNight => "tokyonight",
            Self::Oscura => "oscura-midnight",
            Self::RosePine => "rosepine-moon",
            Self::Ansi => "ansi",
            Self::Auto => "auto",
        }
    }

    /// The next mode when cycling with a bare `/theme`.
    pub fn next(self) -> Self {
        match self {
            Self::Dark => Self::Light,
            Self::Light => Self::TokyoNight,
            Self::TokyoNight => Self::Oscura,
            Self::Oscura => Self::RosePine,
            Self::RosePine => Self::Ansi,
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
            Self::TokyoNight => Theme::tokyonight(),
            Self::Oscura => Theme::oscura(),
            Self::RosePine => Theme::rosepine(),
            Self::Ansi => Theme::ansi(),
            Self::Auto => match os_appearance() {
                Some(OsAppearance::Dark) => Theme::groknight(),
                Some(OsAppearance::Light) => Theme::grokday(),
                None if terminal_supports_truecolor() => Theme::groknight(),
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

/// Resolve a `HI_THEME` value. Missing or unknown values are GrokNight.
fn mode_from_env(raw: Option<&str>) -> ThemeMode {
    raw.and_then(ThemeMode::parse).unwrap_or(ThemeMode::Dark)
}

/// Resolve the initial mode from `HI_THEME` (default `groknight`).
fn initial_mode() -> ThemeMode {
    mode_from_env(std::env::var("HI_THEME").ok().as_deref())
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
    revision: u64,
}

static STATE: OnceLock<RwLock<ThemeState>> = OnceLock::new();

fn cell() -> &'static RwLock<ThemeState> {
    STATE.get_or_init(|| {
        let mode = initial_mode();
        RwLock::new(ThemeState {
            mode,
            theme: mode.resolve(),
            revision: 0,
        })
    })
}

/// The active theme. Cheap `Copy`; read freely on every render.
pub fn theme() -> Theme {
    cell().read().unwrap().theme
}

/// Read the resolved palette and its cache identity under one shared lock.
pub(crate) fn snapshot() -> (Theme, u64) {
    let state = cell().read().unwrap();
    (state.theme, state.revision)
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
    state.revision = state.revision.wrapping_add(1);
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
    if resolved.bg_base != state.theme.bg_base {
        state.theme = resolved;
        state.revision = state.revision.wrapping_add(1);
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
        assert_eq!(t.bg_base, Color::Reset);
    }

    #[test]
    fn truecolor_themes_paint_backgrounds() {
        assert!(Theme::dark().paints_backgrounds());
        assert!(Theme::light().paints_backgrounds());
        assert!(Theme::tokyonight().paints_backgrounds());
        assert!(Theme::oscura().paints_backgrounds());
        assert!(Theme::rosepine().paints_backgrounds());
    }

    #[test]
    fn groknight_is_neutral_gray_not_tokyo_blue() {
        let t = Theme::groknight();
        assert_eq!(t.bg_base, Color::Rgb(20, 20, 20));
        assert_eq!(t.text_primary, Color::Rgb(0xe1, 0xe1, 0xe1));
        assert_ne!(t.bg_base, Theme::tokyonight().bg_base);
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
        assert_eq!(ThemeMode::parse("groknight"), Some(ThemeMode::Dark));
        assert_eq!(ThemeMode::parse("LIGHT"), Some(ThemeMode::Light));
        assert_eq!(ThemeMode::parse("grokday"), Some(ThemeMode::Light));
        assert_eq!(ThemeMode::parse("tokyonight"), Some(ThemeMode::TokyoNight));
        assert_eq!(ThemeMode::parse("oscura"), Some(ThemeMode::Oscura));
        assert_eq!(ThemeMode::parse("rosepine-moon"), Some(ThemeMode::RosePine));
        assert_eq!(ThemeMode::parse("system"), Some(ThemeMode::Auto));
        assert_eq!(ThemeMode::parse("none"), Some(ThemeMode::Ansi));
        assert_eq!(ThemeMode::parse("nope"), None);
        let mut m = ThemeMode::Dark;
        let mut seen = std::collections::HashSet::new();
        for _ in 0..7 {
            seen.insert(m);
            m = m.next();
        }
        assert_eq!(m, ThemeMode::Dark, "cycle is a 7-loop");
        assert_eq!(seen.len(), 7, "cycle visits all modes");
    }

    #[test]
    fn each_mode_resolves_to_the_expected_palette() {
        assert_eq!(
            ThemeMode::Dark.resolve().bg_base,
            Theme::groknight().bg_base
        );
        assert_eq!(ThemeMode::Light.resolve().bg_base, Theme::grokday().bg_base);
        assert_eq!(
            ThemeMode::TokyoNight.resolve().bg_base,
            Theme::tokyonight().bg_base
        );
        assert!(!ThemeMode::Ansi.resolve().paints_backgrounds());
    }

    #[test]
    fn missing_hi_theme_defaults_to_groknight() {
        assert_eq!(mode_from_env(None), ThemeMode::Dark);
        assert_eq!(mode_from_env(Some("")), ThemeMode::Dark);
        assert_eq!(mode_from_env(Some("nope")), ThemeMode::Dark);
        assert_eq!(mode_from_env(Some("auto")), ThemeMode::Auto);
        assert_eq!(mode_from_env(Some("grokday")), ThemeMode::Light);
    }
}
