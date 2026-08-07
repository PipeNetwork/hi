//! The interactive Diff Lab shell.
//!
//! The overlay owns the small API comparison wizard. Provider construction is
//! injected by `hi-cli`; this module only edits non-secret target selections and
//! a deliberately explicit prompt.

use std::time::Instant;

use anyhow::Result;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use hi_diff::{
    BackendKind, DiffMode, DiffRunSnapshot, DiffRunSpec, LocalTarget, RunStatus, TargetSpec,
};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph};

use crate::{DiffApiRunRequest, DiffApiRunner, DiffApiTarget, ProfileInfo};

pub(crate) struct DiffLabOverlay {
    pub(crate) mode: DiffMode,
    pub(crate) snapshot: DiffRunSnapshot,
    /// 0 = prompt, 1 = first target, 2 = second target.
    pub(crate) selected: usize,
    pub(crate) message: String,
    pub(crate) started_at: Option<Instant>,
    pub(crate) task: Option<tokio::task::JoinHandle<Result<DiffRunSnapshot>>>,
    prompt: String,
    api_target_names: Vec<String>,
    api_target_buffers: Vec<String>,
    api_runner: Option<DiffApiRunner>,
}

impl DiffLabOverlay {
    pub(crate) fn open(
        arg: &str,
        profiles: Vec<ProfileInfo>,
        api_runner: Option<DiffApiRunner>,
    ) -> Self {
        let trimmed = arg.trim();
        let mut parts = trimmed.split_whitespace();
        let mode = match parts
            .next()
            .unwrap_or("local")
            .to_ascii_lowercase()
            .as_str()
        {
            "api" | "response" => DiffMode::ApiResponse,
            "agent" | "agents" => DiffMode::AgentOutcome,
            _ => DiffMode::LocalParity,
        };
        let mode_len = trimmed.split_whitespace().next().map_or(0, str::len);
        let prompt = if mode == DiffMode::ApiResponse {
            {
                trimmed
                    .get(mode_len..)
                    .unwrap_or_default()
                    .trim()
                    .to_string()
            }
        } else {
            Default::default()
        };
        let targets = targets_for(mode);
        let mut spec = DiffRunSpec::new(mode, 42, targets);
        spec.case_count = 1;
        spec.max_concurrency = 2;
        Self {
            mode,
            snapshot: DiffRunSnapshot::pending(&spec),
            selected: 0,
            message: if mode == DiffMode::ApiResponse {
                "type a prompt · Tab changes field · Enter runs · Esc closes".into()
            } else {
                "n: run engine smoke test · Esc: close".into()
            },
            started_at: None,
            task: None,
            prompt,
            api_target_names: default_api_targets(&profiles)
                .iter()
                .map(|target| target.name.clone())
                .collect(),
            api_target_buffers: default_api_targets(&profiles)
                .iter()
                .map(target_display)
                .collect(),
            api_runner,
        }
    }

    pub(crate) fn start(&mut self) {
        if self.task.is_some() {
            self.message = "a Diff Lab run is already active".into();
            return;
        }
        if self.mode == DiffMode::ApiResponse {
            if self.prompt.trim().is_empty() {
                self.message = "enter an explicit prompt before running".into();
                self.selected = 0;
                return;
            }
            let Some(runner) = self.api_runner.clone() else {
                self.message = "API runner unavailable in this TUI session".into();
                return;
            };
            let Some(targets) = self.edited_api_targets() else {
                self.message = "targets must use PROFILE:MODEL".into();
                self.selected = 1;
                return;
            };
            let request = DiffApiRunRequest {
                prompt: self.prompt.trim().to_string(),
                targets,
                seed: 42,
                cases: 1,
                max_concurrency: 2,
                max_requests: 2,
                max_tokens: 4096,
            };
            self.snapshot.status = RunStatus::Running;
            self.snapshot.cases_total = request.cases;
            self.snapshot.cases_completed = 0;
            self.snapshot.mismatches = 0;
            self.snapshot.errors = 0;
            self.started_at = Some(Instant::now());
            self.message = "running the same request against both API targets…".into();
            self.task = Some(tokio::spawn(async move { runner(request).await }));
            return;
        }

        let targets = targets_for(self.mode);
        let mut spec = DiffRunSpec::new(self.mode, 42, targets);
        spec.case_count = 256;
        let root = hi_diff::default_root();
        spec.artifact_root = Some(root.clone());
        self.snapshot = DiffRunSnapshot::pending(&spec);
        self.snapshot.status = RunStatus::Running;
        self.started_at = Some(Instant::now());
        self.message = "running deterministic smoke cases…".into();
        self.task = Some(tokio::spawn(async move {
            let store = hi_diff::RunStore::new(root)?;
            store.write_spec(&spec)?;
            let started = hi_diff::DiffEvent::Started(hi_diff::DiffRunSnapshot::pending(&spec));
            store.append_event(&spec.run_id, &started)?;
            let snapshot = hi_diff::run_smoke(&spec)?;
            store.write_named_snapshot(&snapshot)?;
            store.append_event(
                &spec.run_id,
                &hi_diff::DiffEvent::Finished(snapshot.clone()),
            )?;
            Ok(snapshot)
        }));
    }

    pub(crate) fn handle_key(&mut self, key: KeyEvent) -> bool {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        if key.code == KeyCode::Esc || (ctrl && key.code == KeyCode::Char('c')) {
            return true;
        }
        if self.mode == DiffMode::ApiResponse {
            match key.code {
                KeyCode::Tab | KeyCode::Down => {
                    self.selected = (self.selected + 1) % 3;
                }
                KeyCode::BackTab | KeyCode::Up => {
                    self.selected = self.selected.checked_sub(1).unwrap_or(2);
                }
                KeyCode::Backspace => self.backspace_selected(),
                KeyCode::Enter => self.start(),
                KeyCode::Char(ch) if !ctrl => self.insert_selected(ch),
                _ => {}
            }
            return false;
        }
        match key.code {
            KeyCode::Char('n') | KeyCode::Char('r') if !ctrl => {
                self.start();
                false
            }
            KeyCode::Char('q') if !ctrl => true,
            KeyCode::Up => {
                self.selected = self.selected.saturating_sub(1);
                false
            }
            KeyCode::Down => {
                self.selected = self.selected.saturating_add(1).min(3);
                false
            }
            _ => false,
        }
    }

    fn insert_selected(&mut self, ch: char) {
        match self.selected {
            0 => self.prompt.push(ch),
            1 | 2 => self.api_target_buffers[self.selected - 1].push(ch),
            _ => {}
        }
    }

    fn backspace_selected(&mut self) {
        match self.selected {
            0 => {
                self.prompt.pop();
            }
            1 | 2 => {
                self.api_target_buffers[self.selected - 1].pop();
            }
            _ => {}
        }
    }

    pub(crate) fn handle_paste(&mut self, text: &str) {
        match self.selected {
            0 => self.prompt.push_str(text),
            1 | 2 => self.api_target_buffers[self.selected - 1].push_str(text),
            _ => {}
        }
    }

    fn edited_api_targets(&self) -> Option<Vec<DiffApiTarget>> {
        self.api_target_buffers
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let (profile, model) = value.split_once(':')?;
                let profile = profile.trim();
                let model = model.trim();
                (!profile.is_empty() && !model.is_empty()).then(|| DiffApiTarget {
                    name: self.api_target_names[index].clone(),
                    profile: profile.to_string(),
                    model: model.to_string(),
                })
            })
            .collect()
    }

    pub(crate) fn render(&self, frame: &mut ratatui::Frame, area: Rect) {
        frame.render_widget(Clear, area);
        let outer = Block::default()
            .title(" Diff Lab ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(crate::theme::theme().accent_system));
        frame.render_widget(outer, area);
        let inner = area.inner(ratatui::layout::Margin::new(1, 1));
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(if self.mode == DiffMode::ApiResponse {
                    6
                } else {
                    4
                }),
                Constraint::Min(8),
                Constraint::Length(3),
            ])
            .split(inner);

        let status = if self.mode == DiffMode::ApiResponse {
            format!(
                "mode: API response\nstatus: {:?} · {}/{} cases · {} mismatches · {:.0} cases/s\nprompt: {}",
                self.snapshot.status,
                self.snapshot.cases_completed,
                self.snapshot.cases_total,
                self.snapshot.mismatches,
                self.snapshot.cases_per_second,
                self.prompt_or_placeholder(),
            )
        } else {
            format!(
                "mode: {}\nstatus: {:?} · {}/{} cases · {} mismatches · {:.0} cases/s\n{}",
                self.mode.label(),
                self.snapshot.status,
                self.snapshot.cases_completed,
                self.snapshot.cases_total,
                self.snapshot.mismatches,
                self.snapshot.cases_per_second,
                self.message,
            )
        };
        frame.render_widget(
            Paragraph::new(status).block(Block::bordered().title("run")),
            chunks[0],
        );

        let rows = if self.mode == DiffMode::ApiResponse {
            vec![
                editable_row(self.selected == 1, "target 1", &self.api_target_buffers[0]),
                editable_row(self.selected == 2, "target 2", &self.api_target_buffers[1]),
                ListItem::new("────────────────────────────────────────"),
                ListItem::new(format!(
                    "recent failures: {}",
                    self.snapshot.recent_failures.len()
                )),
                ListItem::new(self.message.clone()),
            ]
        } else {
            vec![
                ListItem::new(Line::from(vec![
                    Span::styled(
                        "● ",
                        Style::default().fg(crate::theme::theme().accent_success),
                    ),
                    Span::raw("reference implementation"),
                ])),
                ListItem::new(Line::from(vec![
                    Span::styled(
                        "● ",
                        Style::default().fg(crate::theme::theme().accent_success),
                    ),
                    Span::raw("candidate implementation"),
                ])),
                ListItem::new("────────────────────────────────────────"),
                ListItem::new(format!(
                    "recent failures: {}",
                    self.snapshot.recent_failures.len()
                )),
            ]
        };
        frame.render_widget(
            List::new(rows).block(Block::bordered().title("targets / observations")),
            chunks[1],
        );

        let footer = if self.mode == DiffMode::ApiResponse {
            Paragraph::new(Line::from(vec![
                Span::styled(" Tab/↑↓ ", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw("edit prompt/targets   "),
                Span::styled("Enter ", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw("run   "),
                Span::styled("Esc ", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw("close"),
            ]))
        } else {
            Paragraph::new(Line::from(vec![
                Span::styled(" n/r ", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw("start/replay smoke run   "),
                Span::styled("Esc/q ", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw("close   "),
                Span::styled(
                    "/diff-lab api|agent ",
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::raw("select mode"),
            ]))
        }
        .block(Block::bordered());
        frame.render_widget(footer, chunks[2]);
    }

    fn prompt_or_placeholder(&self) -> String {
        if self.prompt.is_empty() {
            "<type an explicit request>".into()
        } else {
            self.prompt.clone()
        }
    }
}

fn editable_row(selected: bool, label: &str, value: &str) -> ListItem<'static> {
    let prefix = if selected { "▶ " } else { "  " };
    ListItem::new(format!("{prefix}{label}: {value}"))
}

fn target_display(target: &DiffApiTarget) -> String {
    format!("{}:{}", target.profile, target.model)
}

fn default_api_targets(profiles: &[ProfileInfo]) -> Vec<DiffApiTarget> {
    let profile = profiles
        .iter()
        .find(|profile| profile.provider.eq_ignore_ascii_case("pipenetwork"))
        .map(|profile| profile.name.clone())
        .unwrap_or_else(|| "pipenetwork".into());
    vec![
        DiffApiTarget {
            name: "glm-5.2".into(),
            profile: profile.clone(),
            model: "pipe/glm-5.2".into(),
        },
        DiffApiTarget {
            name: "kimi3".into(),
            profile,
            model: "pipe/kimi3".into(),
        },
    ]
}

fn targets_for(mode: DiffMode) -> Vec<TargetSpec> {
    match mode {
        DiffMode::LocalParity => vec![
            TargetSpec::Local(LocalTarget {
                name: "reference".into(),
                backend: BackendKind::Cpu,
                model_path: ".".into(),
                model_fingerprint: None,
            }),
            TargetSpec::Local(LocalTarget {
                name: "candidate".into(),
                backend: BackendKind::Custom,
                model_path: ".".into(),
                model_fingerprint: None,
            }),
        ],
        DiffMode::ApiResponse => vec![
            TargetSpec::Api(hi_diff::ApiTarget {
                name: "glm-5.2".into(),
                profile: "pipenetwork".into(),
                model: "pipe/glm-5.2".into(),
                provider: "pipenetwork".into(),
            }),
            TargetSpec::Api(hi_diff::ApiTarget {
                name: "kimi3".into(),
                profile: "pipenetwork".into(),
                model: "pipe/kimi3".into(),
                provider: "pipenetwork".into(),
            }),
        ],
        DiffMode::AgentOutcome => vec![
            TargetSpec::Agent(hi_diff::AgentTarget {
                name: "agent-a".into(),
                profile: "default".into(),
                model: "current".into(),
                provider: "configured".into(),
                verify_commands: Vec::new(),
            }),
            TargetSpec::Agent(hi_diff::AgentTarget {
                name: "agent-b".into(),
                profile: "default".into(),
                model: "current".into(),
                provider: "configured".into(),
                verify_commands: Vec::new(),
            }),
        ],
    }
}
