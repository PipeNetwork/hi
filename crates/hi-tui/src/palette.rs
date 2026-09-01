//! Command palette (Ctrl-K): fuzzy-filter slash commands and actions.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use hi_agent::help::{CORE_COMMANDS, HelpSection, command_section};

/// One row in the command palette.
#[derive(Clone, Debug)]
pub(crate) struct PaletteItem {
    /// Text inserted/run when accepted (e.g. `/density` or `/help`).
    pub command: String,
    /// Display label.
    pub label: String,
    /// Short help blurb.
    pub help: String,
    /// Disclosure group. `None` for uncategorized builtins.
    pub section: Option<HelpSection>,
}

/// Interactive Ctrl-K palette state.
#[derive(Clone, Debug, Default)]
pub(crate) struct CommandPalette {
    pub query: String,
    pub items: Vec<PaletteItem>,
    pub selected: usize,
}

impl CommandPalette {
    pub fn open() -> Self {
        let mut p = Self::default();
        p.refilter();
        p
    }

    pub fn refilter(&mut self) {
        let needle = self.query.to_ascii_lowercase();
        let mut items = builtin_items();
        for spec in hi_agent::command::COMMANDS {
            items.push(PaletteItem {
                command: if spec.args.is_empty() {
                    format!("/{}", spec.name)
                } else {
                    format!("/{} ", spec.name)
                },
                label: format!("/{}", spec.name),
                help: spec.help.to_string(),
                section: command_section(spec.name),
            });
        }
        // De-dupe by label (builtins may overlap).
        let mut seen = std::collections::HashSet::new();
        items.retain(|i| seen.insert(i.label.clone()));

        if needle.is_empty() {
            // Empty palette is the front door: core, then project, then modes.
            items.retain(|i| {
                matches!(
                    i.section,
                    Some(HelpSection::Core | HelpSection::Project | HelpSection::Modes)
                )
            });
            items.sort_by_key(empty_palette_rank);
        } else {
            items.retain(|i| {
                i.label.to_ascii_lowercase().contains(&needle)
                    || i.help.to_ascii_lowercase().contains(&needle)
                    || i.command.to_ascii_lowercase().contains(&needle)
            });
            items.sort_by_key(|i| {
                let l = i.label.to_ascii_lowercase();
                (
                    !l.contains(&needle),
                    !l.trim_start_matches('/').starts_with(&needle),
                    l,
                )
            });
        }
        self.items = items;
        self.selected = 0;
    }

    pub fn insert(&mut self, c: char) {
        self.query.push(c);
        self.refilter();
    }

    pub fn backspace(&mut self) {
        self.query.pop();
        self.refilter();
    }

    pub fn up(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }

    pub fn down(&mut self) {
        if self.selected + 1 < self.items.len() {
            self.selected += 1;
        }
    }

    pub fn current(&self) -> Option<&PaletteItem> {
        self.items.get(self.selected)
    }

    /// Handle a key while the palette is open. Returns `Some(command)` when the
    /// user accepts a row (caller runs/queues it), `None` if still open, and
    /// sets `closed` when Esc dismisses.
    pub fn handle_key(&mut self, key: &KeyEvent) -> PaletteOutcome {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Esc => PaletteOutcome::Closed,
            KeyCode::Char('c') if ctrl => PaletteOutcome::Closed,
            KeyCode::Char('k') if ctrl => PaletteOutcome::Closed,
            KeyCode::Up => {
                self.up();
                PaletteOutcome::Continue
            }
            KeyCode::Down => {
                self.down();
                PaletteOutcome::Continue
            }
            KeyCode::Enter => {
                if let Some(item) = self.current() {
                    PaletteOutcome::Accept(item.command.clone())
                } else {
                    PaletteOutcome::Closed
                }
            }
            KeyCode::Backspace => {
                self.backspace();
                PaletteOutcome::Continue
            }
            KeyCode::Char(c) if !ctrl => {
                self.insert(c);
                PaletteOutcome::Continue
            }
            _ => PaletteOutcome::Continue,
        }
    }
}

#[derive(Debug)]
pub(crate) enum PaletteOutcome {
    Continue,
    Closed,
    Accept(String),
}

fn empty_palette_rank(item: &PaletteItem) -> (u8, usize, String) {
    let section_ord = match item.section {
        None => 0,
        Some(HelpSection::Core) => 1,
        Some(HelpSection::Project) => 2,
        Some(HelpSection::Modes) => 3,
        Some(HelpSection::Platform) => 4,
    };
    let core_idx = item
        .label
        .strip_prefix('/')
        .and_then(|name| CORE_COMMANDS.iter().position(|n| *n == name))
        .unwrap_or(usize::MAX);
    (section_ord, core_idx, item.label.clone())
}

fn builtin_items() -> Vec<PaletteItem> {
    vec![
        PaletteItem {
            command: "/tutorial".into(),
            label: "/tutorial".into(),
            help: "interactive tour".into(),
            section: Some(HelpSection::Core),
        },
        PaletteItem {
            command: "/density".into(),
            label: "/density".into(),
            help: "cycle transcript density".into(),
            section: None,
        },
        PaletteItem {
            command: "/theme".into(),
            label: "/theme".into(),
            help: "cycle color theme".into(),
            section: None,
        },
        PaletteItem {
            command: "/help".into(),
            label: "/help".into(),
            help: "core commands; /help all for the rest".into(),
            section: Some(HelpSection::Core),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_lists_everyday_commands() {
        let p = CommandPalette::open();
        assert!(p.items.len() > 5);
        assert!(p.items.iter().any(|i| i.label == "/help"));
        assert!(p.items.iter().any(|i| i.label == "/verify"));
        assert!(p.items.iter().any(|i| i.label == "/fleet"));
        assert!(
            !p.items.iter().any(|i| i.label == "/rsi"),
            "empty palette hides platform commands"
        );
        assert!(
            !p.items.iter().any(|i| i.label == "/dashboard"),
            "empty palette hides /dashboard alias"
        );
    }

    #[test]
    fn filter_finds_platform_commands() {
        let mut p = CommandPalette::open();
        p.insert('r');
        p.insert('s');
        p.insert('i');
        assert!(
            p.items.iter().any(|i| i.label == "/rsi"),
            "search still finds platform commands: {:?}",
            p.items.iter().map(|i| &i.label).collect::<Vec<_>>()
        );
    }

    #[test]
    fn filter_narrows() {
        let mut p = CommandPalette::open();
        p.insert('d');
        p.insert('e');
        p.insert('n');
        assert!(
            p.items.iter().any(|i| i.label.contains("density")),
            "density should match: {:?}",
            p.items.iter().map(|i| &i.label).collect::<Vec<_>>()
        );
    }
}
