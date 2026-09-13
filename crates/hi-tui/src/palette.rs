//! Command palette (Ctrl-K): fuzzy-filter slash commands and actions.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use hi_harness::{COMMANDS, CORE_COMMANDS};

/// One row in the command palette.
#[derive(Clone, Debug)]
pub(crate) struct PaletteItem {
    /// Text inserted/run when accepted (e.g. `/density` or `/help`).
    pub command: String,
    /// Display label.
    pub label: String,
    /// Short help blurb.
    pub help: String,
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
        for spec in COMMANDS {
            items.push(PaletteItem {
                command: if spec.args.is_empty() {
                    format!("/{}", spec.name)
                } else {
                    format!("/{} ", spec.name)
                },
                label: format!("/{}", spec.name),
                help: spec.help.to_string(),
            });
        }
        let mut seen = std::collections::HashSet::new();
        items.retain(|i| seen.insert(i.label.clone()));

        if needle.is_empty() {
            items.retain(|i| {
                i.label
                    .strip_prefix('/')
                    .is_some_and(|name| CORE_COMMANDS.contains(&name))
                    || matches!(i.label.as_str(), "/help" | "/tutorial")
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

fn empty_palette_rank(item: &PaletteItem) -> (usize, String) {
    let core_idx = item
        .label
        .strip_prefix('/')
        .and_then(|name| CORE_COMMANDS.iter().position(|n| *n == name))
        .unwrap_or(usize::MAX);
    (core_idx, item.label.clone())
}

fn builtin_items() -> Vec<PaletteItem> {
    vec![
        PaletteItem {
            command: "/tutorial".into(),
            label: "/tutorial".into(),
            help: "interactive tour".into(),
        },
        PaletteItem {
            command: "/density".into(),
            label: "/density".into(),
            help: "cycle transcript density".into(),
        },
        PaletteItem {
            command: "/theme".into(),
            label: "/theme".into(),
            help: "cycle color theme".into(),
        },
        PaletteItem {
            command: "/help".into(),
            label: "/help".into(),
            help: "core commands".into(),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_lists_everyday_commands() {
        let p = CommandPalette::open();
        assert!(p.items.iter().any(|i| i.label == "/help"));
        assert!(p.items.iter().any(|i| i.label == "/login"));
        assert!(p.items.iter().any(|i| i.label == "/model"));
        assert!(p.items.iter().any(|i| i.label == "/effort"));
        assert!(p.items.iter().any(|i| i.label == "/yolo"));
        assert!(p.items.iter().any(|i| i.label == "/auto"));
        assert!(p.items.iter().any(|i| i.label == "/verify"));
        assert!(p.items.iter().any(|i| i.label == "/dashboard"));
        assert!(!p.items.iter().any(|i| i.label == "/fleet"));
    }

    #[test]
    fn filter_finds_model() {
        let mut p = CommandPalette::open();
        for c in ['m', 'o', 'd', 'e', 'l'] {
            p.insert(c);
        }
        assert!(
            p.items.iter().any(|i| i.label == "/model"),
            "search still finds /model: {:?}",
            p.items.iter().map(|i| &i.label).collect::<Vec<_>>()
        );
    }
}
