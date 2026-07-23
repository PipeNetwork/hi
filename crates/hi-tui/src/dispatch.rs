//! Apply [`Action`]s to [`App`] and mode-local key handlers.
//!
//! Keeps `run.rs` thin: resolve → apply, with specialized fallthrough for
//! text editing and normal-mode search.

use crossterm::event::KeyEvent;

use crate::App;
use crate::action::{self, Action, KeySurface};
use crate::domain::{ComposerDomain, OverlayDomain, TurnChromeDomain};
use crate::mode::UiMode;

/// Result of handling a key through the action/mode pipeline.
#[derive(Debug)]
pub(crate) enum DispatchResult {
    /// Key fully handled; redraw and continue.
    Handled,
    /// Caller should run text-editing / submit path (`edit_key`).
    Fallthrough,
    /// Open the command palette overlay.
    OpenPalette,
}

impl App {
    /// Whether a hard keyboard-owning overlay is up (confirmation, pickers, forms, palette).
    pub(crate) fn has_hard_overlay(&self) -> bool {
        OverlayDomain::any_hard(self)
    }

    /// Current key surface for action resolution.
    pub(crate) fn key_surface(&self) -> KeySurface {
        KeySurface::from_app(&self.mode, self.has_hard_overlay())
    }

    /// Resolve and apply a key. Returns [`DispatchResult::Fallthrough`] when the
    /// caller should still run `edit_key` / normal-mode search typing.
    pub(crate) fn dispatch_key(&mut self, key: &KeyEvent) -> DispatchResult {
        let surface = self.key_surface();

        // Hard overlays are owned by specialized handlers in `run.rs`. If we
        // ever reach dispatch while one is up, swallow — never fall through to
        // edit_key (would type into the composer under a modal).
        if surface == KeySurface::Overlay {
            return DispatchResult::Handled;
        }

        // Normal-mode search typing is specialized (not action-table driven).
        if matches!(self.mode, UiMode::Normal { search: Some(_) }) {
            return DispatchResult::Fallthrough;
        }

        let mut action = action::resolve_key(surface, key);

        // ToggleHelp only when the input is empty (otherwise `?` is a character).
        if action == Action::ToggleHelp && !ComposerDomain::input_empty(self) {
            action = Action::None;
        }

        // Queue actions no-op when empty — still "handled" so they don't type.
        if matches!(
            action,
            Action::QueueSelectPrev
                | Action::QueueSelectNext
                | Action::QueueRemoveSelected
                | Action::QueueMoveSelected { .. }
        ) && ComposerDomain::queue_is_empty(self)
        {
            return DispatchResult::Handled;
        }

        if action == Action::None {
            // Block-nav / review with unmatched keys stay swallowed.
            if matches!(surface, KeySurface::BlockNav | KeySurface::Review) {
                return DispatchResult::Handled;
            }
            return DispatchResult::Fallthrough;
        }

        if action == Action::OpenPalette {
            return DispatchResult::OpenPalette;
        }

        self.apply_action(action);
        DispatchResult::Handled
    }

    /// Apply a resolved action. Palette open is handled by the caller via
    /// [`DispatchResult::OpenPalette`]; calling this with [`Action::OpenPalette`]
    /// is a no-op by design (use `dispatch_key`).
    pub(crate) fn apply_action(&mut self, action: Action) {
        match action {
            Action::None | Action::OpenPalette => {}
            Action::ToggleHelp => {
                self.show_help = !self.show_help;
            }
            Action::VoiceToggle => self.toggle_voice(),
            Action::ToggleDebug => {
                self.show_debug = !self.show_debug;
            }
            Action::ToggleDiff => {
                self.show_diff = !self.show_diff;
                if self.show_diff {
                    self.diff_text = Some(crate::working_tree_diff_sync(&self.workspace_root));
                } else {
                    self.diff_text = None;
                }
            }
            Action::ToggleReview => {
                if self.mode.is_review() {
                    self.mode.to_insert();
                } else {
                    self.open_review(None);
                }
            }
            Action::ToggleReasoning => {
                self.show_reasoning = !self.show_reasoning;
                self.bump_transcript();
            }
            Action::ToggleToolOutput => {
                self.show_tool_output = !self.show_tool_output;
                self.bump_transcript();
            }
            Action::ToggleBlockNav => {
                let n = self.tool_block_count();
                if n > 0 {
                    if self.mode.is_block_nav() {
                        self.mode.to_insert();
                    } else {
                        self.mode = UiMode::BlockNav;
                        self.block_cursor = n - 1;
                    }
                }
            }
            Action::EnterNormal => {
                self.mode = UiMode::Normal { search: None };
            }
            Action::ExitToInsert => {
                self.mode.to_insert();
            }
            Action::CopyLastCode => {
                self.copy_last_code_block();
            }
            Action::ExternalEdit => {
                self.edit_in_external_editor();
            }
            Action::CycleDensity => {
                self.density = self.density.next();
                self.bump_transcript();
                self.push(ratatui::text::Line::styled(
                    TurnChromeDomain::density_status(self),
                    ratatui::style::Style::default().fg(crate::theme::theme().accent_success),
                ));
                self.follow();
            }
            Action::JumpPrompt { dir } => {
                self.jump_transcript_marker(TranscriptMarker::UserPrompt, dir);
            }
            Action::JumpError { dir } => {
                self.jump_transcript_marker(TranscriptMarker::Error, dir);
            }
            Action::QueueSelectPrev => self.queue_select_prev(),
            Action::QueueSelectNext => self.queue_select_next(),
            Action::QueueRemoveSelected => {
                let _ = self.queue_remove_selected();
            }
            Action::QueueMoveSelected { delta } => self.queue_move_selected(delta),
            Action::ReviewClose => {
                self.mode.to_insert();
            }
            Action::ReviewScroll { delta } => {
                if delta == Action::REVIEW_SCROLL_END {
                    let total = self
                        .diff_text
                        .as_deref()
                        .map(|t| t.lines().count())
                        .unwrap_or(0);
                    self.review_scroll = total;
                } else if delta > 0 {
                    self.review_scroll = self.review_scroll.saturating_add(delta as usize);
                } else {
                    self.review_scroll = self
                        .review_scroll
                        .saturating_sub(delta.unsigned_abs() as usize);
                }
            }
            Action::ReviewHunk { dir } => {
                self.review_scroll = crate::app::review_next_hunk(
                    self.diff_text.as_deref(),
                    self.review_scroll,
                    dir,
                );
            }
            Action::BlockNavUp => {
                self.block_cursor = self.selected_block_ord().saturating_sub(1);
            }
            Action::BlockNavDown => {
                let n = self.tool_block_count();
                if n > 0 {
                    self.block_cursor = (self.selected_block_ord() + 1).min(n - 1);
                }
            }
            Action::BlockNavToggle => self.toggle_selected_block(),
            Action::BlockNavExit => self.mode.to_insert(),
        }
    }

    /// Jump the transcript scroll to the next/prev marker of `kind`.
    pub(crate) fn jump_transcript_marker(&mut self, kind: TranscriptMarker, dir: i32) {
        // Ensure cache is warm enough to have prompt indices; use last known width.
        let width = self.view_cache.width.max(40);
        let nav = self.mode.is_block_nav().then(|| self.selected_block_ord());
        self.ensure_view_cache(width, nav);

        let targets: Vec<u16> = match kind {
            TranscriptMarker::UserPrompt => self
                .view_cache
                .prompt_line_starts
                .iter()
                .filter_map(|&idx| self.view_cache.prefix.get(idx).copied())
                .map(|r| r.min(u16::MAX as u32) as u16)
                .collect(),
            TranscriptMarker::Error => {
                // Scan flattened line texts for error markers.
                let mut rows = Vec::new();
                for (i, line) in self.view_cache.lines.iter().enumerate() {
                    let t = crate::render::line_text(line);
                    if t.contains('✗')
                        || t.contains("failed ·")
                        || t.contains("error:")
                        || t.starts_with('⚠')
                    {
                        if let Some(&row) = self.view_cache.prefix.get(i) {
                            rows.push(row.min(u16::MAX as u32) as u16);
                        }
                    }
                }
                rows
            }
        };
        if targets.is_empty() {
            return;
        }
        let cur = self.scroll;
        let next = if dir > 0 {
            targets
                .iter()
                .copied()
                .find(|&r| r > cur)
                .or_else(|| targets.first().copied())
        } else {
            targets
                .iter()
                .copied()
                .rev()
                .find(|&r| r < cur)
                .or_else(|| targets.last().copied())
        };
        if let Some(row) = next {
            self.scroll_to(row);
        }
    }
}

/// Kinds of transcript landmarks the jump commands seek.
#[derive(Clone, Copy, Debug)]
pub(crate) enum TranscriptMarker {
    UserPrompt,
    Error,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::{Action, KeySurface, resolve_key};
    use crate::tests::test_app;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn key(code: KeyCode, mods: KeyModifiers) -> KeyEvent {
        KeyEvent::new(code, mods)
    }

    #[test]
    fn overlay_surface_is_swallowed_not_fallthrough() {
        let mut app = test_app("openai", "gpt-4o");
        app.palette = Some(crate::palette::CommandPalette::open());
        assert!(app.has_hard_overlay());
        assert_eq!(app.key_surface(), KeySurface::Overlay);
        let r = app.dispatch_key(&key(KeyCode::Char('x'), KeyModifiers::NONE));
        assert!(matches!(r, DispatchResult::Handled));
    }

    #[test]
    fn toggle_help_only_when_input_empty() {
        let mut app = test_app("openai", "gpt-4o");
        assert!(matches!(
            app.dispatch_key(&key(KeyCode::Char('?'), KeyModifiers::NONE)),
            DispatchResult::Handled
        ));
        assert!(app.show_help);

        app.input.set("has text");
        app.show_help = false;
        let r = app.dispatch_key(&key(KeyCode::Char('?'), KeyModifiers::NONE));
        assert!(
            matches!(r, DispatchResult::Fallthrough),
            "? with non-empty input should type"
        );
        assert!(!app.show_help);
    }

    #[test]
    fn ctrl_space_resolves_to_voice_toggle_while_typing() {
        // Ctrl+Space is the dictation trigger. It must resolve on the insert
        // surface even with text already in the prompt — dictation appends to
        // what you have typed rather than replacing it.
        let app = test_app("openai", "gpt-4o");
        assert_eq!(app.key_surface(), KeySurface::Insert);
        assert_eq!(
            crate::keys::resolve_from_table(
                KeySurface::Insert,
                &key(KeyCode::Char(' '), KeyModifiers::CONTROL)
            ),
            Action::VoiceToggle
        );
        // Plain space must still type a space, not start recording.
        assert_eq!(
            crate::keys::resolve_from_table(
                KeySurface::Insert,
                &key(KeyCode::Char(' '), KeyModifiers::NONE)
            ),
            Action::None
        );
    }

    #[test]
    fn open_palette_returns_caller_result() {
        let mut app = test_app("openai", "gpt-4o");
        let r = app.dispatch_key(&key(KeyCode::Char('k'), KeyModifiers::CONTROL));
        assert!(matches!(r, DispatchResult::OpenPalette));
        // apply_action alone must not open it
        app.apply_action(Action::OpenPalette);
        assert!(app.palette.is_none());
    }

    #[test]
    fn cycle_density_via_action() {
        let mut app = test_app("openai", "gpt-4o");
        let before = app.density;
        app.apply_action(Action::CycleDensity);
        assert_eq!(app.density, before.next());
    }

    #[test]
    fn enter_normal_via_action() {
        let mut app = test_app("openai", "gpt-4o");
        app.apply_action(Action::EnterNormal);
        assert!(app.mode.is_normal());
    }

    #[test]
    fn resolve_and_table_agree_on_ctrl_d() {
        assert_eq!(
            resolve_key(
                KeySurface::Insert,
                &key(KeyCode::Char('d'), KeyModifiers::CONTROL)
            ),
            Action::ToggleDiff
        );
    }

    #[test]
    fn normal_search_fallthrough_suppresses_globals_until_handler() {
        let mut app = test_app("openai", "gpt-4o");
        app.mode = UiMode::Normal {
            search: Some("q".into()),
        };
        // dispatch_key returns Fallthrough before resolve; globals don't fire.
        let r = app.dispatch_key(&key(KeyCode::Char('d'), KeyModifiers::CONTROL));
        assert!(matches!(r, DispatchResult::Fallthrough));
        assert!(!app.show_diff);
    }

    #[test]
    fn empty_queue_chords_are_handled() {
        let mut app = test_app("openai", "gpt-4o");
        assert!(app.queue.is_empty());
        let r = app.dispatch_key(&key(KeyCode::Up, KeyModifiers::ALT));
        assert!(matches!(r, DispatchResult::Handled));
    }

    #[test]
    fn block_nav_unmatched_is_handled() {
        let mut app = test_app("openai", "gpt-4o");
        app.mode = UiMode::BlockNav;
        let r = app.dispatch_key(&key(KeyCode::Char('x'), KeyModifiers::NONE));
        assert!(matches!(r, DispatchResult::Handled));
    }
}
