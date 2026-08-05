//! TUI surface for a provider-backed coding race.

use std::time::Instant;

use anyhow::Result;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use hi_race::{CandidateState, RaceSnapshot, RaceSpec, RaceStatus};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph};

use crate::{RaceDefaults, RaceRunRequest, RaceRunner};

pub(crate) struct RaceOverlay {
    pub(crate) snapshot: RaceSnapshot,
    pub(crate) selected: usize,
    pub(crate) message: String,
    pub(crate) started_at: Option<Instant>,
    expanded: bool,
    pub(crate) task: Option<tokio::task::JoinHandle<Result<RaceSnapshot>>>,
    task_text: String,
    defaults: RaceDefaults,
    runner: Option<RaceRunner>,
}

impl RaceOverlay {
    pub(crate) fn open(task: &str, defaults: RaceDefaults, runner: Option<RaceRunner>) -> Self {
        let spec = RaceSpec::new(task.trim(), defaults.targets.clone());
        let mut snapshot = RaceSnapshot::pending(&spec);
        snapshot.artifact_root = None;
        let message = if defaults.targets.len() < 2 {
            "configure at least two targets with /race setup".into()
        } else if defaults.verify_commands.is_empty() {
            "set /verify before starting a coding race".into()
        } else {
            "Enter: start · ↑↓: inspect candidate · a: apply reviewed winner · Esc: close".into()
        };
        Self {
            snapshot,
            selected: 0,
            message,
            started_at: None,
            expanded: false,
            task: None,
            task_text: task.trim().to_string(),
            defaults,
            runner,
        }
    }

    pub(crate) fn can_start(&self) -> bool {
        self.task.is_none()
            && self.defaults.targets.len() >= 2
            && !self.defaults.verify_commands.is_empty()
            && !self.task_text.trim().is_empty()
    }

    pub(crate) fn start(&mut self, apply: bool) {
        if !self.can_start() {
            self.message = if self.task_text.trim().is_empty() {
                "use /race <task> to describe the coding change".into()
            } else if self.defaults.targets.len() < 2 {
                "configure at least two targets with /race setup".into()
            } else if self.defaults.verify_commands.is_empty() {
                "set /verify before starting a coding race".into()
            } else {
                "a race is already active".into()
            };
            return;
        }
        let Some(runner) = self.runner.clone() else {
            self.message = "race runner unavailable in this TUI session".into();
            return;
        };
        let selected_candidate = apply
            .then(|| {
                self.snapshot
                    .candidates
                    .get(self.selected)
                    .map(|candidate| candidate.candidate_id.clone())
                    .or_else(|| self.snapshot.selected_candidate.clone())
            })
            .flatten();
        let request = RaceRunRequest {
            task: self.task_text.clone(),
            targets: self.defaults.targets.clone(),
            max_candidates: self.defaults.max_candidates,
            max_concurrency: self.defaults.max_concurrency,
            verify_commands: self.defaults.verify_commands.clone(),
            fuzz: self.defaults.fuzz.clone(),
            apply,
            source_run_id: apply.then(|| self.snapshot.run_id.clone()),
            artifact_root: apply.then(|| self.snapshot.artifact_root.clone()).flatten(),
            selected_candidate,
            expected_workspace_digest: apply.then(|| self.snapshot.workspace_digest.clone()),
        };
        let spec = RaceSpec::new(&request.task, request.targets.clone());
        self.snapshot = RaceSnapshot::pending(&spec);
        self.snapshot.status = RaceStatus::Running;
        self.started_at = Some(Instant::now());
        self.message = if apply {
            "applying the reviewed winner and re-verifying the destination…".into()
        } else {
            "running isolated candidates and verification stages…".into()
        };
        self.task = Some(tokio::spawn(async move { runner(request).await }));
    }

    pub(crate) fn handle_key(&mut self, key: KeyEvent) -> bool {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        if key.code == KeyCode::Esc || (ctrl && key.code == KeyCode::Char('c')) {
            if let Some(task) = self.task.take() {
                task.abort();
                self.snapshot.status = RaceStatus::Cancelled;
                self.message = "race cancelled".into();
            } else {
                return true;
            }
            return false;
        }
        match key.code {
            KeyCode::Enter if self.snapshot.status == RaceStatus::Ready && self.task.is_none() => {
                self.expanded = !self.expanded;
            }
            KeyCode::Char('d') if !ctrl => self.expanded = !self.expanded,
            KeyCode::Enter | KeyCode::Char('r') if !ctrl => self.start(false),
            KeyCode::Char('a') if !ctrl && self.snapshot.status == RaceStatus::Ready => {
                self.start(true)
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.selected = self.selected.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.selected = self
                    .selected
                    .saturating_add(1)
                    .min(self.snapshot.candidates.len().saturating_sub(1));
            }
            _ => {}
        }
        false
    }

    pub(crate) fn render(&self, frame: &mut ratatui::Frame, area: Rect) {
        frame.render_widget(Clear, area);
        let outer = Block::default()
            .title(" Coding Race ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(crate::theme::theme().accent_system));
        frame.render_widget(outer, area);
        let inner = area.inner(ratatui::layout::Margin::new(1, 1));
        let constraints = if self.expanded {
            vec![
                Constraint::Length(5),
                Constraint::Min(8),
                Constraint::Length(8),
                Constraint::Length(3),
            ]
        } else {
            vec![
                Constraint::Length(5),
                Constraint::Min(8),
                Constraint::Length(3),
            ]
        };
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(constraints)
            .split(inner);

        let elapsed = self
            .started_at
            .map(|started| format!(" · {:.1}s", started.elapsed().as_secs_f32()))
            .unwrap_or_default();
        let status = format!(
            "status: {:?}{elapsed}\ntask: {}\ntargets: {} · fuzz: {}",
            self.snapshot.status,
            if self.task_text.is_empty() {
                "<missing>"
            } else {
                &self.task_text
            },
            self.snapshot.candidates.len(),
            if self.defaults.fuzz.is_some() {
                "on"
            } else {
                "off"
            },
        );
        frame.render_widget(
            Paragraph::new(status).block(Block::bordered().title("run")),
            chunks[0],
        );

        let rows = self
            .snapshot
            .candidates
            .iter()
            .enumerate()
            .map(|(index, candidate)| {
                let marker = if index == self.selected { "▶" } else { " " };
                let state = match candidate.state {
                    CandidateState::Passed => "PASS",
                    CandidateState::Failed => "FAIL",
                    CandidateState::Running => "run ",
                    CandidateState::Verifying => "test",
                    CandidateState::Fuzzing => "fuzz",
                    CandidateState::Cancelled => "stop",
                    CandidateState::Abandoned => "gone",
                    CandidateState::Pending => "wait",
                };
                let detail = candidate
                    .failure_reason
                    .as_deref()
                    .or_else(|| candidate.fuzz.as_ref().map(|fuzz| fuzz.name.as_str()))
                    .unwrap_or("");
                ListItem::new(Line::from(vec![
                    Span::raw(format!(
                        "{marker} {:<5} {:<16} {:<18}",
                        state,
                        format!(
                            "{}@{}",
                            candidate.target.name,
                            if candidate.target.model.is_empty() {
                                "model"
                            } else {
                                &candidate.target.model
                            }
                        ),
                        detail
                    )),
                    Span::styled(
                        format!(
                            " {} file(s), {} lines, {}ms",
                            candidate.actual_changes.len(),
                            candidate.changed_lines,
                            candidate.wall_clock_ms
                        ),
                        Style::default().add_modifier(Modifier::DIM),
                    ),
                ]))
            })
            .collect::<Vec<_>>();
        frame.render_widget(
            List::new(rows).block(Block::bordered().title("candidates")),
            chunks[1],
        );
        let footer = if self.expanded {
            let detail = self
                .snapshot
                .candidates
                .get(self.selected)
                .map(|candidate| {
                    let verification = if candidate.verify.is_empty() {
                        "not run".to_string()
                    } else {
                        candidate
                            .verify
                            .iter()
                            .map(|stage| {
                                format!(
                                    "{} {}",
                                    stage.name,
                                    if stage.passed { "PASS" } else { "FAIL" }
                                )
                            })
                            .collect::<Vec<_>>()
                            .join(", ")
                    };
                    format!(
                        "target: {} / {}\nverify: {} · fuzz: {}\nfiles: {} · lines: {} · runtime: {}ms\npatch artifact: {}",
                        candidate.target.profile,
                        candidate.target.model,
                        verification,
                        candidate
                            .fuzz
                            .as_ref()
                            .map(|stage| if stage.passed { "PASS" } else { "FAIL" })
                            .unwrap_or("not configured"),
                        candidate.actual_changes.len(),
                        candidate.changed_lines,
                        candidate.wall_clock_ms,
                        candidate.artifact_ref.as_deref().unwrap_or("none"),
                    )
                })
                .unwrap_or_else(|| "no candidate selected".into());
            frame.render_widget(
                Paragraph::new(detail).block(Block::bordered().title("selected candidate")),
                chunks[2],
            );
            chunks[3]
        } else {
            chunks[2]
        };
        frame.render_widget(
            Paragraph::new(format!("{} · Enter/d: details", self.message)).block(Block::bordered()),
            footer,
        );
    }
}
