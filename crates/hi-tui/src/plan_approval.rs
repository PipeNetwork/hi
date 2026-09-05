//! Plan-approval card shown automatically after a successful plan draft.
//!
//! Approve starts leftover drive. Request changes returns to plan mode so the
//! user can type feedback (line comments are included). Quit turns plan mode
//! off and pauses auto-drive. Esc parks the card so keys return to the
//! composer; `/view-plan` or a click on the turn-status row reopens it.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use hi_agent::Agent;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{BorderType, Paragraph, Wrap};

use crate::App;
use crate::render::dim;
use crate::theme::{UiTone, theme};

const CHOICES: [&str; 3] = [
    "Approve — leave plan mode and start implementing",
    "Request changes — stay in plan mode and edit the plan",
    "Quit — turn plan mode off without driving",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PlanApprovalFocus {
    Preview,
    Choices,
    Commenting,
}

#[derive(Clone, Debug)]
pub(crate) struct PlanComment {
    pub step: usize,
    pub text: String,
}

#[derive(Clone, Debug)]
pub(crate) struct PlanApproval {
    pub selected: usize,
    pub preview_sel: usize,
    pub focus: PlanApprovalFocus,
    pub parked: bool,
    review_after_draft: bool,
    pub comments: Vec<PlanComment>,
    pub comment_draft: String,
}

impl PlanApproval {
    pub(crate) fn new() -> Self {
        Self {
            selected: 0,
            preview_sel: 0,
            focus: PlanApprovalFocus::Preview,
            parked: false,
            review_after_draft: false,
            comments: Vec::new(),
            comment_draft: String::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PlanApprovalOutcome {
    Continue,
    Park,
    Approve,
    RequestChanges,
    Quit,
}

pub(crate) fn leftover_indices(app: &App) -> Vec<usize> {
    app.plan
        .iter()
        .enumerate()
        .filter(|(_, step)| {
            matches!(
                step.status,
                hi_agent::PlanStatus::Pending | hi_agent::PlanStatus::Active
            )
        })
        .map(|(i, _)| i)
        .collect()
}

pub(crate) fn handle_key(app: &mut App, key: &KeyEvent) -> PlanApprovalOutcome {
    let leftover = leftover_indices(app);
    let Some(card) = app.plan_approval.as_mut() else {
        return PlanApprovalOutcome::Continue;
    };
    if card.parked {
        return PlanApprovalOutcome::Continue;
    }
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    match card.focus {
        PlanApprovalFocus::Commenting => match key.code {
            KeyCode::Esc => {
                card.comment_draft.clear();
                card.focus = PlanApprovalFocus::Preview;
                PlanApprovalOutcome::Continue
            }
            KeyCode::Enter => {
                let text = card.comment_draft.trim().to_string();
                if !text.is_empty() {
                    let step = leftover
                        .get(card.preview_sel)
                        .copied()
                        .unwrap_or(card.preview_sel);
                    card.comments.push(PlanComment { step, text });
                }
                card.comment_draft.clear();
                card.focus = PlanApprovalFocus::Preview;
                PlanApprovalOutcome::Continue
            }
            KeyCode::Backspace => {
                card.comment_draft.pop();
                PlanApprovalOutcome::Continue
            }
            KeyCode::Char(c) if !ctrl => {
                card.comment_draft.push(c);
                PlanApprovalOutcome::Continue
            }
            _ => PlanApprovalOutcome::Continue,
        },
        PlanApprovalFocus::Preview => match key.code {
            KeyCode::Up | KeyCode::Char('k') if !ctrl => {
                card.preview_sel = card.preview_sel.saturating_sub(1);
                PlanApprovalOutcome::Continue
            }
            KeyCode::Down | KeyCode::Char('j') if !ctrl => {
                let last = leftover.len().saturating_sub(1);
                card.preview_sel = (card.preview_sel + 1).min(last);
                PlanApprovalOutcome::Continue
            }
            KeyCode::Tab => {
                card.focus = PlanApprovalFocus::Choices;
                PlanApprovalOutcome::Continue
            }
            KeyCode::Char('c') if !ctrl => {
                card.comment_draft.clear();
                card.focus = PlanApprovalFocus::Commenting;
                PlanApprovalOutcome::Continue
            }
            KeyCode::Char('1') | KeyCode::Char('a') if !ctrl => PlanApprovalOutcome::Approve,
            KeyCode::Char('2') | KeyCode::Char('r') if !ctrl => PlanApprovalOutcome::RequestChanges,
            KeyCode::Char('3') | KeyCode::Char('q') if !ctrl => PlanApprovalOutcome::Quit,
            KeyCode::Esc => PlanApprovalOutcome::Park,
            KeyCode::Enter => {
                card.focus = PlanApprovalFocus::Choices;
                PlanApprovalOutcome::Continue
            }
            _ => PlanApprovalOutcome::Continue,
        },
        PlanApprovalFocus::Choices => match key.code {
            KeyCode::Up | KeyCode::Char('k') if !ctrl => {
                card.selected = card.selected.saturating_sub(1);
                PlanApprovalOutcome::Continue
            }
            KeyCode::Down | KeyCode::Char('j') if !ctrl => {
                card.selected = (card.selected + 1).min(CHOICES.len() - 1);
                PlanApprovalOutcome::Continue
            }
            KeyCode::BackTab | KeyCode::Tab => {
                card.focus = PlanApprovalFocus::Preview;
                PlanApprovalOutcome::Continue
            }
            KeyCode::Char('1') | KeyCode::Char('a') if !ctrl => PlanApprovalOutcome::Approve,
            KeyCode::Char('2') | KeyCode::Char('r') if !ctrl => PlanApprovalOutcome::RequestChanges,
            KeyCode::Char('3') | KeyCode::Char('q') if !ctrl => PlanApprovalOutcome::Quit,
            KeyCode::Esc => PlanApprovalOutcome::Park,
            KeyCode::Enter => match card.selected {
                1 => PlanApprovalOutcome::RequestChanges,
                2 => PlanApprovalOutcome::Quit,
                _ => PlanApprovalOutcome::Approve,
            },
            _ => PlanApprovalOutcome::Continue,
        },
    }
}

pub(crate) fn render(frame: &mut ratatui::Frame, area: Rect, app: &App) {
    let Some(card) = &app.plan_approval else {
        return;
    };
    if card.parked {
        return;
    }
    let th = theme();
    let leftover_idx = leftover_indices(app);
    let mut body = vec![
        Line::styled(
            "Review the plan before execution.",
            Style::default()
                .fg(th.accent_plan)
                .add_modifier(Modifier::BOLD),
        ),
        Line::raw(""),
    ];
    if leftover_idx.is_empty() {
        body.push(Line::styled("No checklist steps left.", dim()));
    } else {
        body.push(Line::styled(
            format!("{} leftover step(s):", leftover_idx.len()),
            Style::default().fg(th.text_secondary),
        ));
        for (i, &step_i) in leftover_idx.iter().enumerate().take(8) {
            let title = app.plan.get(step_i).map(|s| s.title.as_str()).unwrap_or("");
            let notes: Vec<_> = card.comments.iter().filter(|c| c.step == step_i).collect();
            let selected = card.focus != PlanApprovalFocus::Choices && i == card.preview_sel;
            let mark = if selected { "▶ " } else { "  " };
            if selected {
                let mut fence = None;
                body.push(crate::render::markdown_line(
                    &format!("{mark}▸ {title}"),
                    &mut fence,
                ));
                for comment in &notes {
                    body.extend(crate::render::markdown_body_lines(&format!(
                        "> {}",
                        comment.text
                    )));
                }
            } else {
                let mut line = format!("{mark}▸ {title}");
                if !notes.is_empty() {
                    let n = notes.len();
                    line.push_str(&format!("  ({n} comment{})", if n == 1 { "" } else { "s" }));
                }
                body.push(Line::styled(line, dim()));
            }
        }
        if leftover_idx.len() > 8 {
            body.push(Line::styled(
                format!("  … +{} more", leftover_idx.len() - 8),
                dim(),
            ));
        }
    }
    body.push(Line::raw(""));
    if card.focus == PlanApprovalFocus::Commenting {
        body.push(Line::styled(
            "Comment on this step:",
            Style::default().fg(th.text_secondary),
        ));
        body.push(Line::from(vec![
            Span::styled("  ", Style::default()),
            Span::raw(card.comment_draft.clone()),
            Span::styled("▏", dim()),
        ]));
        body.push(Line::styled(" Enter save · Esc cancel", dim()));
    } else {
        for (i, label) in CHOICES.iter().enumerate() {
            if card.focus == PlanApprovalFocus::Choices && i == card.selected {
                body.push(Line::styled(
                    format!("▶ {label}"),
                    Style::default()
                        .fg(th.accent_plan)
                        .add_modifier(Modifier::BOLD),
                ));
            } else {
                body.push(Line::styled(format!("  {label}"), dim()));
            }
        }
    }
    let hint = match card.focus {
        PlanApprovalFocus::Preview => {
            " j/k steps · c comment · Tab choices · a approve · Esc park "
        }
        PlanApprovalFocus::Choices => {
            " j/k · Enter · a approve · r request changes · q quit · Esc park "
        }
        PlanApprovalFocus::Commenting => " type a comment · Enter save · Esc back ",
    };
    let block = th
        .panel_block(" Plan approval ", UiTone::Warning)
        .border_type(BorderType::Rounded)
        .title_bottom(Line::styled(hint, dim()));
    frame.render_widget(
        Paragraph::new(body).block(block).wrap(Wrap { trim: false }),
        area,
    );
}

impl App {
    pub(crate) fn plan_approval_capturing(&self) -> bool {
        self.plan_approval.as_ref().is_some_and(|p| !p.parked)
    }

    /// Input must target the surface actually rendered. A completed draft can
    /// open its card while a full-screen overlay or tool question is still up.
    /// Keep the approval gate pending, but never accept a hidden plan decision.
    pub(crate) fn plan_approval_visible(&self) -> bool {
        self.plan_approval_capturing()
            && self.confirmation.is_none()
            && self.tutorial.is_none()
            && self.workflow_overlay.is_none()
            && self.inspect_subagent.is_none()
            && self.tasks_overlay.is_none()
            && self.block_viewer.is_none()
            && self.jump_picker.is_none()
            && self.rewind_picker.is_none()
            && self.memory_browser.is_none()
            && self.diff_lab.is_none()
            && self.race.is_none()
            && !self.mode.is_review()
            && self.local_download_confirmation.is_none()
            && self.local_directory_prompt.is_none()
            && self.local_picker.is_none()
            && !((self.local_startup_blocked || self.local_startup_error.is_some())
                && self.provider_picker.is_none()
                && self.provider_form.is_none()
                && self.picker.is_none())
    }

    /// Reopen a parked card. Returns `true` only for the transition, so repeated
    /// `/view-plan` dispatches cannot consume or emit the unpark twice.
    pub(crate) fn unpark_plan_approval(&mut self) -> bool {
        if let Some(card) = self.plan_approval.as_mut() {
            if !card.parked {
                return false;
            }
            card.parked = false;
            card.review_after_draft = false;
            card.focus = PlanApprovalFocus::Preview;
            self.completion = None;
            self.session_face_dirty = true;
            self.status = "Waiting on plan approval".into();
            let _ = self.trace_approval_shown("plan");
            return true;
        }
        false
    }

    pub(crate) fn park_plan_approval_local(&mut self) {
        let Some(card) = self.plan_approval.as_mut() else {
            return;
        };
        if card.parked {
            return;
        }
        card.parked = true;
        card.review_after_draft = false;
        self.session_face_dirty = true;
        self.status = "plan approval parked — /view-plan".into();
        self.trace_approval_decided("plan", "parked");
    }

    /// Bracketed paste belongs to the visible comment editor, just like typed
    /// characters. Keep a parked card from consuming the main composer's paste.
    pub(crate) fn paste_plan_comment(&mut self, text: &str) -> bool {
        if !self.plan_approval_visible() {
            return false;
        }
        let Some(card) = self
            .plan_approval
            .as_mut()
            .filter(|card| !card.parked && card.focus == PlanApprovalFocus::Commenting)
        else {
            return false;
        };
        card.comment_draft
            .push_str(&text.replace("\r\n", "\n").replace('\r', "\n"));
        true
    }

    pub(crate) fn park_plan_approval(&mut self, agent: &mut Agent) {
        self.park_plan_approval_local();
        self.push_session_face(agent);
    }

    pub(crate) fn open_plan_approval(&mut self) {
        self.plan_approval = Some(PlanApproval::new());
        self.completion = None;
        self.session_face_dirty = true;
        self.status = "Waiting on plan approval".into();
        let _ = self.trace_approval_shown("plan");
    }

    pub(crate) fn restore_parked_plan_approval(&mut self) {
        let mut card = PlanApproval::new();
        card.parked = true;
        self.plan_approval = Some(card);
        self.status = "plan approval parked — /view-plan".into();
    }

    pub(crate) fn maybe_open_plan_approval(&mut self) {
        if self.plan_approval.is_some() {
            return;
        }
        if self.plan_mode {
            return;
        }
        if !self.plan_has_leftover() {
            return;
        }
        if self
            .goal
            .as_ref()
            .is_some_and(hi_agent::Goal::has_drive_work)
        {
            return;
        }
        self.open_plan_approval();
    }

    /// A new user revision supersedes an earlier decision to park review. Keep
    /// the card and feedback until the draft succeeds; a fresh mid-turn park
    /// remains an explicit choice and clears this pending review.
    pub(crate) fn begin_plan_draft(&mut self, started_in_plan_mode: bool) {
        if let Some(card) = self.plan_approval.as_mut() {
            card.review_after_draft = started_in_plan_mode && card.parked;
        }
    }

    /// Drafting stays read-only until the user approves. The end of a
    /// successful draft presents that decision without requiring a mode key.
    /// An explicit mid-turn approval already left plan mode and must not open
    /// a second card when the original turn finishes.
    pub(crate) fn finish_plan_draft(
        &mut self,
        started_in_plan_mode: bool,
        outcome: Option<&hi_agent::TurnOutcome>,
    ) {
        let review_parked = self
            .plan_approval
            .as_mut()
            .is_some_and(|card| std::mem::take(&mut card.review_after_draft) && card.parked);
        if started_in_plan_mode
            && self.plan_mode
            && self.plan_has_leftover()
            && outcome.is_some_and(|outcome| outcome.status == hi_agent::TurnStatus::Completed)
        {
            if self.plan_approval.is_none() {
                self.open_plan_approval();
            } else if review_parked {
                self.unpark_plan_approval();
            }
        }
    }

    fn comment_prompt(&self) -> Option<String> {
        let card = self.plan_approval.as_ref()?;
        if card.comments.is_empty() {
            return None;
        }
        let mut out = String::from("Please revise the plan:\n");
        for comment in &card.comments {
            let title = self
                .plan
                .get(comment.step)
                .map(|s| s.title.as_str())
                .unwrap_or("step");
            out.push_str(&format!("- [{title}]: {}\n", comment.text));
        }
        let draft = self.input.text();
        if !draft.trim().is_empty() {
            out.push('\n');
            out.push_str(&draft);
        }
        Some(out)
    }

    pub(crate) fn apply_plan_approve(&mut self, agent: &mut Agent) -> bool {
        let card = self.plan_approval.take();
        self.plan_mode = false;
        self.plan_drive_paused = false;
        self.plan_drive_pause_dirty = true;
        self.session_face_dirty = true;
        if !self.push_session_face(agent) {
            if card.is_some() {
                self.plan_approval = card;
            }
            return false;
        }
        self.trace_approval_decided("plan", "approved");
        self.status = "plan approved — driving leftover work".into();
        true
    }

    pub(crate) fn apply_plan_request_changes(&mut self, agent: &mut Agent) {
        self.trace_approval_decided("plan", "request_changes");
        let prompt = self.comment_prompt();
        self.plan_approval = None;
        self.plan_mode = true;
        self.permission_mode = hi_agent::PermissionMode::Ask;
        self.session_face_dirty = true;
        if !self.push_session_face(agent) {
            return;
        }
        if let Some(prompt) = prompt {
            self.input.set(&prompt);
        }
        self.status = "back in plan mode — type changes, then Enter".into();
    }

    pub(crate) fn apply_plan_quit(&mut self, agent: &mut Agent) {
        self.trace_approval_decided("plan", "quit");
        self.plan_approval = None;
        self.plan_mode = false;
        self.plan_drive_paused = true;
        self.plan_drive_pause_dirty = true;
        self.session_face_dirty = true;
        if !self.push_session_face(agent) {
            return;
        }
        self.status = "plan drive paused".into();
        self.trace_drive_state("drive_paused", "plan_drive", "plan_approval_quit");
    }

    /// Mid-turn / no-agent path: flip App flags; [`App::push_session_face`]
    /// applies them when the agent is free again.
    pub(crate) fn apply_plan_approve_local(&mut self) {
        self.trace_approval_decided("plan", "approved");
        self.plan_approval = None;
        self.plan_mode = false;
        self.plan_drive_paused = false;
        self.plan_drive_pause_dirty = true;
        self.session_face_dirty = true;
        self.status = "plan approved — driving leftover work".into();
    }

    pub(crate) fn apply_plan_request_changes_local(&mut self) {
        self.trace_approval_decided("plan", "request_changes");
        let prompt = self.comment_prompt();
        self.plan_approval = None;
        self.plan_mode = true;
        self.permission_mode = hi_agent::PermissionMode::Ask;
        self.session_face_dirty = true;
        if let Some(prompt) = prompt {
            self.input.set(&prompt);
        }
        self.status = "back in plan mode — type changes, then Enter".into();
    }

    pub(crate) fn apply_plan_quit_local(&mut self) {
        self.trace_approval_decided("plan", "quit");
        self.plan_approval = None;
        self.plan_mode = false;
        self.plan_drive_paused = true;
        self.plan_drive_pause_dirty = true;
        self.session_face_dirty = true;
        self.status = "plan drive paused".into();
        self.trace_drive_state("drive_paused", "plan_drive", "plan_approval_quit");
    }
}

#[cfg(test)]
#[path = "plan_approval_tests.rs"]
mod tests;
