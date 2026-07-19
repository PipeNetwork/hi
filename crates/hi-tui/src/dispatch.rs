//! Apply [`Action`]s to [`App`] and mode-local key handlers.
//!
//! Keeps `run.rs` thin: resolve → apply, with specialized fallthrough for
//! text editing and normal-mode search.

use crossterm::event::KeyEvent;

use crate::action::{self, Action, KeySurface};
use crate::mode::UiMode;
use crate::App;

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
    /// Whether a hard keyboard-owning overlay is up (confirmation, pickers, forms).
    pub(crate) fn has_hard_overlay(&self) -> bool {
        // Keep OverlayDomain as the documented seam even while fields live on App.
        let _ = crate::domain::OverlayDomain::palette_open(self);
        self.confirmation.is_some()
            || self.picker.is_some()
            || self.provider_picker.is_some()
            || self.provider_form.is_some()
            || self.palette.is_some()
    }

    /// Current key surface for action resolution.
    pub(crate) fn key_surface(&self) -> KeySurface {
        KeySurface::from_app(&self.mode, self.has_hard_overlay())
    }

    /// Resolve and apply a key. Returns [`DispatchResult::Fallthrough`] when the
    /// caller should still run `edit_key` / normal-mode search typing.
    pub(crate) fn dispatch_key(&mut self, key: &KeyEvent) -> DispatchResult {
        let surface = self.key_surface();

        // Normal-mode search typing is specialized (not action-table driven).
        if matches!(self.mode, UiMode::Normal { search: Some(_) }) {
            return DispatchResult::Fallthrough;
        }

        let mut action = action::resolve_key(surface, key);

        // ToggleHelp only when the input is empty (otherwise `?` is a character).
        if action == Action::ToggleHelp && !self.input.is_empty() {
            action = Action::None;
        }

        // Queue actions no-op when empty — still "handled" so they don't type.
        if matches!(
            action,
            Action::QueueSelectPrev
                | Action::QueueSelectNext
                | Action::QueueRemoveSelected
                | Action::QueueMoveSelected { .. }
        ) && self.queue.is_empty()
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

    /// Apply a resolved action. Palette open is handled by the caller.
    pub(crate) fn apply_action(&mut self, action: Action) {
        match action {
            Action::None => {}
            Action::ToggleHelp => {
                self.show_help = !self.show_help;
            }
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
                    format!(
                        "density: {}",
                        crate::domain::TurnChromeDomain::density_status(self)
                    ),
                    ratatui::style::Style::default().fg(crate::theme::theme().accent_success),
                ));
                self.follow();
            }
            Action::OpenPalette => {
                // Caller opens the palette (needs to set state after this returns).
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
                if delta == i32::MAX / 4 {
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
                    if t.contains("✗")
                        || t.contains("failed ·")
                        || t.contains("error:")
                        || t.starts_with("⚠")
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
            targets.iter().copied().find(|&r| r > cur).or_else(|| targets.first().copied())
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
