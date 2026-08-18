//! `App` methods: render.

use hi_agent::{Agent, PlanStatus};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Paragraph, Wrap};

use crate::chrome::{self, ShortcutHint};
use crate::layout::{UiLayout, display_width, truncate_display};
use crate::model_picker::{display_capabilities, display_price, display_window};
use crate::render::{diff_lines, dim, lerp_color, markdown_line, wrapped_line_height};
use crate::theme::UiTone;
use crate::util::{fmt_count, fmt_elapsed, fmt_rate_limits};
use crate::{FORM_LABEL_WIDTH, PICKER_ROWS, SPINNER, TurnEventKind, TurnState};

/// Clip ghost suffix to `max_width` cells without trimming leading spaces
/// (those spaces are part of the remaining suggestion).
fn clip_ghost(text: &str, max_width: usize) -> String {
    if display_width(text) <= max_width {
        return text.to_string();
    }
    let mut width = 0;
    let mut out = String::new();
    for ch in text.chars() {
        let ch_width = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        if width + ch_width > max_width {
            break;
        }
        width += ch_width;
        out.push(ch);
    }
    out
}

/// Render a confirmation request's details as styled lines so the user can
/// review a diff or command with real coloring instead of a wall of plain text.
/// - `FileEdit` / `DelegateApply`: the embedded unified diff is colored (added
///   lines green, removed lines red, hunk headers cyan); the `file:`/summary
///   header above it stays in secondary text.
/// - `ShellMutation`: the `$ command` line is highlighted bold so the exact
///   command being approved stands out.
fn confirmation_lines(
    request: &hi_agent::ConfirmationRequest,
    details: &str,
) -> Vec<Line<'static>> {
    use hi_agent::ConfirmationRequest;
    let th = crate::theme::theme();
    match request {
        ConfirmationRequest::FileEdit { .. } | ConfirmationRequest::DelegateApply { .. } => {
            // The details are "file: <path>\n\n<diff>" or "<summary>\n\n<diff>".
            // Split off the diff portion (after the blank line) and color it.
            let (header, diff) = match details.split_once("\n\n") {
                Some((h, d)) => (h, d),
                None => (details, ""),
            };
            let mut lines: Vec<Line<'static>> = header
                .lines()
                .map(|l| Line::styled(l.to_string(), Style::default().fg(th.text_secondary)))
                .collect();
            if !diff.is_empty() {
                lines.push(Line::raw(""));
                // `diff_lines` colors unified diffs; fall back to plain lines.
                if crate::render::looks_like_diff(diff) {
                    lines.extend(crate::render::diff_lines(diff));
                } else {
                    lines.extend(diff.lines().map(|l| Line::raw(l.to_string())));
                }
            }
            lines
        }
        ConfirmationRequest::ShellMutation { .. } => {
            // Highlight the `$ command` line bold so the exact command stands out.
            details
                .lines()
                .map(|l| {
                    if let Some(cmd) = l.strip_prefix("$ ") {
                        Line::from(vec![
                            Span::styled("$ ", Style::default().fg(th.accent_tool)),
                            Span::styled(
                                cmd.to_string(),
                                Style::default()
                                    .fg(th.text_primary)
                                    .add_modifier(Modifier::BOLD),
                            ),
                        ])
                    } else if l.starts_with("working directory:") || l.starts_with("warning:") {
                        Line::styled(l.to_string(), Style::default().fg(th.text_secondary))
                    } else {
                        Line::raw(l.to_string())
                    }
                })
                .collect()
        }
        ConfirmationRequest::AskUser { question, options } => {
            let mut lines = vec![Line::styled(
                question.clone(),
                Style::default()
                    .fg(th.text_primary)
                    .add_modifier(Modifier::BOLD),
            )];
            if !options.is_empty() {
                lines.push(Line::raw(""));
                for (i, option) in options.iter().enumerate() {
                    lines.push(Line::from(vec![
                        Span::styled(format!("  {} ", i + 1), Style::default().fg(th.accent_tool)),
                        Span::raw(option.clone()),
                    ]));
                }
            }
            lines
        }
    }
}

/// Paint the selection background over just the character range `[lo, hi)` of a
/// line, splitting spans at the range boundaries so only the selected glyphs are
/// highlighted (character-precise selection within one line).
fn highlight_char_range(line: &mut Line<'static>, lo: usize, hi: usize, bg: Color) {
    let mut out: Vec<Span<'static>> = Vec::new();
    let mut pos = 0usize;
    for span in std::mem::take(&mut line.spans) {
        let chars: Vec<char> = span.content.chars().collect();
        let (s0, s1) = (pos, pos + chars.len());
        pos = s1;
        let a = lo.max(s0);
        let b = hi.min(s1);
        if b <= a {
            out.push(span);
            continue;
        }
        let base = span.style;
        if a > s0 {
            out.push(Span::styled(
                chars[..a - s0].iter().collect::<String>(),
                base,
            ));
        }
        out.push(Span::styled(
            chars[a - s0..b - s0].iter().collect::<String>(),
            base.bg(bg),
        ));
        if b < s1 {
            out.push(Span::styled(
                chars[b - s0..].iter().collect::<String>(),
                base,
            ));
        }
    }
    line.spans = out;
}

fn review_repair_summary(t: &hi_agent::TurnTelemetry) -> Option<String> {
    if t.quality_repair_nudges == 0
        && t.review_repair_counts.is_empty()
        && t.review_repair_exhaustion_reason.is_empty()
    {
        return None;
    }

    let mut parts = vec![format!("total {}", t.quality_repair_nudges)];
    let mut counts = t.review_repair_counts.iter().collect::<Vec<_>>();
    counts.sort_by(|(left_mode, left_count), (right_mode, right_count)| {
        right_count
            .cmp(left_count)
            .then_with(|| left_mode.cmp(right_mode))
    });
    let top_modes = counts
        .into_iter()
        .take(2)
        .map(|(mode, count)| format!("{}={count}", hi_agent::compact_review_repair_label(mode)))
        .collect::<Vec<_>>();
    if !top_modes.is_empty() {
        parts.push(format!("top {}", top_modes.join(", ")));
    }
    let mut summary = format!("review repair: {}", parts.join(" · "));
    if !t.review_repair_exhaustion_reason.is_empty() {
        summary.push_str(&format!(
            "\nexhausted {}",
            hi_agent::compact_review_repair_label(&t.review_repair_exhaustion_reason)
        ));
    }
    Some(summary)
}

impl crate::App {
    /// Quiet model-phase lead for the turn-status row: `thinking…`,
    /// `responding…`, or `Working…` plus the turn timer. Command identity
    /// lives on transcript `Run` rows — this line never dumps tool ids,
    /// `bash_output`, round counts, or call counts.
    pub(crate) fn activity_line(&self) -> String {
        let secs = self.started.map(|t| t.elapsed().as_secs()).unwrap_or(0);
        let verb = if self.current_tool.is_some() {
            "Working"
        } else {
            match self.last_turn_event {
                Some(TurnEventKind::Reasoning) => "thinking",
                Some(TurnEventKind::Assistant) => "responding",
                _ => "Working",
            }
        };
        format!("{verb}… {}", fmt_elapsed(secs))
    }

    /// The `Working` lead rendered as a rolling wave: every letter starts gray,
    /// and one letter at a time lights up white (bold) sweeping across the word
    /// and back, like the Codex app's animation. Driven by the per-redraw
    /// `spinner` tick so it advances whenever the UI redraws.
    ///
    /// Returns the styled spans for the word `Working` (no trailing `…`/timer);
    /// the caller appends those so the wave stays on the verb itself.
    pub(crate) fn working_spans(&self) -> Vec<Span<'static>> {
        const WORD: &str = "Working";
        let chars: Vec<char> = WORD.chars().collect();
        let n = chars.len();
        // Sweep forward 0..n-1 then back n-1..0, giving a 2*(n-1) step cycle.
        let cycle = 2 * (n - 1).max(1);
        let step = self.spinner % cycle;
        let lit = if step < n { step } else { cycle - step };
        let th = crate::theme::theme();
        let gray = Style::default().fg(th.gray_dim);
        let lit_style = Style::default()
            .fg(th.accent_running)
            .add_modifier(Modifier::BOLD);
        chars
            .iter()
            .enumerate()
            .map(|(i, &c)| Span::styled(c.to_string(), if i == lit { lit_style } else { gray }))
            .collect()
    }

    fn live_subagent_tick(&self) -> u64 {
        self.subagents
            .values()
            .filter(|info| info.live())
            .map(|info| info.started_at.elapsed().as_secs())
            .max()
            .unwrap_or(0)
    }

    /// Turn activity sits between scrollback and the prompt. Idle and done
    /// hide the row; latency folds into the header. Warnings and failures stay.
    fn turn_status_line(&self, width: u16) -> Option<Line<'static>> {
        crate::turn_status::build(self, width)
    }

    /// Keep the shortcuts row whenever the terminal is tall enough. Grok-build
    /// always shows `Shift+Tab:mode` here; `?` still opens the cheat sheet.
    fn show_shortcuts_row(&self) -> bool {
        true
    }

    pub(crate) fn report_status(&mut self, agent: &Agent) {
        let (input, output) = self.usage;
        let state = match &self.last_turn_state {
            TurnState::Idle => "idle".to_string(),
            TurnState::Running => "running".to_string(),
            TurnState::Done(s) if s == "done" => "done".to_string(),
            TurnState::Done(s) => format!("done ({s})"),
            TurnState::Warning(s) => format!("warning ({s})"),
            TurnState::Failed(s) => format!("failed ({s})"),
            TurnState::Cancelled => "cancelled".to_string(),
        };
        let ctx = self
            .context_pct()
            .map(|p| format!("{}{p}%", if self.usage_estimated { "~" } else { "" }))
            .unwrap_or_else(|| "unknown".to_string());
        let goal = agent.goal_summary();
        let verify = agent.verify_summary();
        let tel = agent.last_turn_telemetry();
        let error = self.last_error.as_deref().unwrap_or("none");
        for line in [
            format!("status: {state}"),
            format!("execution: {}", agent.execution_mode().as_str()),
            format!("provider/model: {} · {}", self.provider, self.model),
            format!(
                "local MLX: {}",
                self.local_runtime
                    .as_ref()
                    .map(|runtime| format!(
                        "{} · quant={} · source={} · endpoint={} · {}",
                        runtime.model_id,
                        runtime.quantization.as_deref().unwrap_or("unknown"),
                        runtime.source,
                        runtime.endpoint.as_deref().unwrap_or("not bound"),
                        if runtime.ready {
                            "runtime ready"
                        } else {
                            "runtime loading"
                        }
                    ))
                    .unwrap_or_else(|| "inactive".to_string())
            ),
            format!(
                "context: {ctx}; user prompt estimate: {input}; turn output across all model calls: {}{output}",
                if self.usage_estimated { "~" } else { "" }
            ),
            format!("goal: {goal}"),
            format!("verify: {verify}"),
            format!(
                "evidence: {} (reads {}, searches {}, listing_only {}, repair nudges {})",
                tel.discovery_depth,
                tel.file_reads,
                tel.targeted_searches,
                tel.listing_only,
                tel.quality_repair_nudges
            ),
            format!("last error: {error}"),
            format!(
                "startup notice: {}",
                self.startup_notice.as_deref().unwrap_or("none")
            ),
            format!(
                "queued: {}; checkpoints: {}",
                self.queue.len(),
                agent.checkpoint_count()
            ),
        ] {
            self.push(Line::styled(line, dim()));
        }
        self.follow();
    }

    /// The editable input rendered as one or more lines (the prompt may hold a
    /// pasted multi-line block), plus the cursor's (row, col) within them. Long
    /// inputs show only their last [`MAX_PROMPT_ROWS`] lines with a "… more above"
    /// note so they can't swallow the screen.
    ///
    /// `width` is the inner width of the input box (borders already subtracted).
    /// Each logical line is soft-wrapped to that width so a long single-line
    /// prompt stays visible and the cursor tracks the wrap instead of running off
    /// the right edge.
    /// Colour of the recording dot at a given redraw tick (test accessor).
    ///
    /// A triangle wave over a 20-tick cycle (0 → 1 → 0), so the dot breathes
    /// between muted and the error accent instead of sitting static.
    #[cfg(test)]
    pub(crate) fn recording_dot_color_at(tick: usize) -> ratatui::style::Color {
        recording_dot_color(tick)
    }

    /// One-line status for voice dictation, or `None` when idle.
    ///
    /// The recording dot pulses off the redraw spinner so an open microphone
    /// reads as live rather than as a static glyph that might be stale.
    pub(crate) fn voice_indicator(&self) -> Option<Line<'static>> {
        let th = crate::theme::theme();
        if self.voice.is_recording() {
            return Some(Line::from(vec![
                Span::styled("● ", Style::default().fg(recording_dot_color(self.spinner))),
                Span::styled(
                    "recording — Ctrl+Space to stop",
                    Style::default().fg(th.text_primary),
                ),
            ]));
        }
        if self.voice.is_transcribing() {
            return Some(Line::styled("◌ transcribing…".to_string(), dim()));
        }
        self.voice.download_percent().map(|percent| {
            Line::styled(format!("↓ downloading the voice model… {percent}%"), dim())
        })
    }

    pub(crate) fn input_view(&self, width: u16) -> (Vec<Line<'static>>, u16, u16) {
        const MAX_PROMPT_ROWS: usize = 10;
        const PREFIX: usize = 2; // "❯ " or "  "
        let text = self.input.text();
        let before: String = text.chars().take(self.input.cursor()).collect();
        let cursor_col_logical = display_width(
            before
                .rsplit_once('\n')
                .map(|(_, line)| line)
                .unwrap_or(&before),
        );

        // Inner text width per line (prefix occupies the first 3 columns).
        let wrap_w = width.saturating_sub(PREFIX as u16).max(1) as usize;

        // Split into logical lines, then soft-wrap each to `wrap_w` columns.
        // Each entry is (display_lines, cursor_offset_within_this_logical_line)
        // where cursor_offset is Some(col) if the cursor sits in this logical
        // line, else None.
        let all: Vec<&str> = text.split('\n').collect();
        let cursor_logical_row = before.matches('\n').count();

        // Build wrapped display lines and track the cursor's display (row, col).
        // Each entry: (chunk_text, cursor_col_within_chunk_if_cursor_here).
        let mut wrapped: Vec<(String, Option<usize>)> = Vec::new();
        for (li, seg) in all.iter().enumerate() {
            let cursor_in_this = if li == cursor_logical_row {
                Some(cursor_col_logical)
            } else {
                None
            };
            if seg.is_empty() {
                wrapped.push((String::new(), cursor_in_this));
                continue;
            }
            let chars: Vec<char> = seg.chars().collect();
            let mut start = 0;
            let mut start_width = 0;
            while start < chars.len() {
                let mut end = start;
                let mut chunk_width = 0;
                while end < chars.len() {
                    let char_width =
                        unicode_width::UnicodeWidthChar::width(chars[end]).unwrap_or(0);
                    if end > start && chunk_width + char_width > wrap_w {
                        break;
                    }
                    chunk_width += char_width;
                    end += 1;
                }
                // A single wide glyph still gets a line when the available
                // width is narrower than that glyph.
                if end == start {
                    end += 1;
                    chunk_width = unicode_width::UnicodeWidthChar::width(chars[start]).unwrap_or(0);
                }
                let chunk: String = chars[start..end].iter().collect();
                // The cursor is in this display line if its logical column falls
                // within [start_width, start_width + chunk_width]. A cursor exactly
                // at the end of a wrapped
                // chunk) stays on this line's last column rather than jumping to
                // the next line's column 0 — matches how terminals render it.
                let cursor_here = cursor_in_this.and_then(|c| {
                    if c >= start_width && c <= start_width + chunk_width {
                        Some(c - start_width)
                    } else {
                        None
                    }
                });
                wrapped.push((chunk, cursor_here));
                start = end;
                start_width += chunk_width;
            }
        }

        let truncated = wrapped.len() > MAX_PROMPT_ROWS;
        let start = if truncated {
            wrapped.len() - MAX_PROMPT_ROWS
        } else {
            0
        };

        let mut lines: Vec<Line<'static>> = Vec::new();
        let mut cursor_row: u16 = 0;
        let mut cursor_col: u16 = 0;
        let mut found_cursor = false;
        if truncated {
            lines.push(Line::styled(
                format!("  ⋮ {} more line(s) above", start),
                dim(),
            ));
        }
        let cursor_at_end = self.input.cursor() == self.input.chars.len();
        let ghost = if cursor_at_end {
            self.ghost_suffix().map(str::to_owned)
        } else {
            None
        };
        let last_visible = wrapped[start..].len().saturating_sub(1);
        for (i, (chunk, cursor_here)) in wrapped[start..].iter().enumerate() {
            // Match Grok's compact prompt: `❯ ` on the first row and continuation
            // rows aligned directly under the editable text.
            let first = i == 0 && !truncated;
            let prefix_span = if first {
                Span::styled(
                    "❯ ",
                    Style::default()
                        .fg(crate::theme::theme().accent_user)
                        .add_modifier(Modifier::BOLD),
                )
            } else {
                Span::raw("  ")
            };
            let suffix = if i == last_visible {
                ghost.as_deref()
            } else {
                None
            };
            let mut spans = vec![prefix_span];
            if first && chunk.is_empty() {
                if let Some(suggestion) = suffix {
                    let shown = clip_ghost(suggestion, wrap_w);
                    spans.push(Span::styled(shown, dim()));
                }
            } else {
                spans.extend(crate::file_mentions::mention_spans(chunk));
                if let Some(suggestion) = suffix {
                    let remain = wrap_w.saturating_sub(display_width(chunk));
                    if remain > 0 {
                        spans.push(Span::styled(clip_ghost(suggestion, remain), dim()));
                    }
                }
            }
            lines.push(Line::from(spans));
            if let Some(col) = cursor_here
                && !found_cursor
            {
                cursor_row = u16::from(truncated) + i as u16;
                cursor_col = (PREFIX + col) as u16;
                found_cursor = true;
            }
        }
        // Cursor past the very end (e.g. empty input): place at end of last line.
        if !found_cursor {
            cursor_row = lines.len().saturating_sub(1) as u16;
            cursor_col = PREFIX as u16;
        }
        (lines, cursor_row, cursor_col)
    }

    /// The pinned plan checklist shown just above the input, or empty when no
    /// plan has been posted. Done steps dim out; the active step is bold cyan.
    /// `max_steps` caps how many step lines are rendered (on top of the header)
    /// so a long plan can't swallow the input area or overflow the screen.
    pub(crate) fn plan_lines(&self, max_steps: usize) -> Vec<Line<'static>> {
        // Prefer the structured-goal view when a long-horizon goal is active: it's
        // the authoritative decomposition the executor's `update_plan` maps onto, so
        // showing both would be redundant.
        if let Some(goal) = &self.goal
            && !goal.sub_goals.is_empty()
        {
            if !self.plan_pane_expanded {
                return vec![Line::styled(
                    format!(
                        "▸ goal · {}/{}  Ctrl-L",
                        goal.sub_goals
                            .iter()
                            .filter(|s| s.status == hi_agent::GoalStatus::Done)
                            .count(),
                        goal.sub_goals.len()
                    ),
                    Style::default()
                        .fg(crate::theme::theme().accent_plan)
                        .add_modifier(Modifier::BOLD),
                )];
            }
            return self.goal_lines(goal, max_steps);
        }
        if self.plan.is_empty() {
            return Vec::new();
        }
        const HARD_CAP: usize = 8;
        let max_steps = if self.plan_pane_expanded {
            max_steps.min(HARD_CAP)
        } else {
            0
        };
        let total = self.plan.len();
        let done = self
            .plan
            .iter()
            .filter(|s| s.status == PlanStatus::Done)
            .count();
        let th = crate::theme::theme();
        let mut out = vec![Line::styled(
            {
                let mut header = format!("plan · {done}/{total}");
                if self.plan_drive_paused {
                    header.push_str(" · paused");
                } else if matches!(
                    self.last_drive,
                    hi_agent::DriveAction::Idle {
                        reason: hi_agent::DriveIdleReason::PlanParked
                    }
                ) {
                    header.push_str(" · parked");
                }
                if !self.plan_pane_expanded {
                    header.push_str("  Ctrl-L");
                }
                header
            },
            Style::default()
                .fg(th.accent_plan)
                .add_modifier(Modifier::BOLD),
        )];
        for s in self.plan.iter().take(max_steps) {
            let (glyph, glyph_style, title_style) = match s.status {
                PlanStatus::Done => ('✓', Style::default().fg(th.accent_success), dim()),
                PlanStatus::Active => (
                    '▸',
                    Style::default()
                        .fg(th.accent_plan)
                        .add_modifier(Modifier::BOLD),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                PlanStatus::Pending => ('☐', dim(), Style::default()),
            };
            out.push(Line::from(vec![
                Span::styled(format!("  {glyph} "), glyph_style),
                Span::styled(s.title.clone(), title_style),
            ]));
        }
        if self.plan_pane_expanded && total > max_steps {
            out.push(Line::styled(
                format!("  … +{} more", total - max_steps),
                dim(),
            ));
        }
        out
    }

    fn queue_pane_lines(&self, max: usize) -> Vec<Line<'static>> {
        if self.queue.is_empty() || max == 0 {
            return Vec::new();
        }
        let th = crate::theme::theme();
        let mut out = Vec::new();
        for (i, q) in self.queue.iter().enumerate().take(max) {
            let selected = self.queue_selected == Some(i);
            let first = q
                .lines()
                .map(str::trim)
                .find(|line| !line.is_empty())
                .unwrap_or(q.as_str());
            let extra = q.lines().count().saturating_sub(1);
            let mut text = format!("#{} {first}", i + 1);
            if extra > 0 {
                text.push_str(&format!(" (+{extra} lines)"));
            }
            let prefix = if selected { "▶ " } else { "  " };
            let style = if selected {
                Style::default()
                    .fg(th.accent_running)
                    .add_modifier(Modifier::BOLD)
            } else {
                dim()
            };
            out.push(Line::styled(format!("{prefix}{text}"), style));
        }
        if self.queue.len() > max {
            out.push(Line::styled(
                format!("  … +{} more queued", self.queue.len() - max),
                dim(),
            ));
        }
        out
    }

    fn live_task_lines(&self, max: usize) -> Vec<Line<'static>> {
        if max == 0 {
            return Vec::new();
        }
        let th = crate::theme::theme();
        let mut live: Vec<_> = self.subagents.values().filter(|info| info.live()).collect();
        live.sort_by(|a, b| b.started_at.cmp(&a.started_at));
        if live.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        let item_budget = if max >= 2 {
            out.push(Line::from(vec![
                Span::styled("▾ ", dim()),
                Span::styled(
                    "Subagents",
                    Style::default()
                        .fg(th.gray_bright)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(format!(" {}", live.len()), dim()),
            ]));
            max - 1
        } else {
            max
        };
        for info in live.iter().take(item_budget) {
            let kind = if info.background {
                "task"
            } else {
                info.kind.as_str()
            };
            let status = if info.activity.trim().is_empty() {
                "Responding"
            } else {
                info.activity.as_str()
            };
            out.push(Line::from(vec![
                Span::styled("  ", dim()),
                Span::styled(
                    format!("○ {kind} {}", crate::util::clip_reason(&info.description)),
                    Style::default().fg(th.text_primary),
                ),
                Span::styled(format!(" — {status}"), dim()),
                Span::styled(
                    format!(
                        "  {}",
                        crate::util::fmt_elapsed(info.started_at.elapsed().as_secs())
                    ),
                    dim(),
                ),
            ]));
        }
        if live.len() > item_budget {
            out.push(Line::styled(
                format!("  … +{} more · /tasks", live.len() - item_budget),
                dim(),
            ));
        }
        out
    }

    /// The pinned block for an active long-horizon goal: a `goal · done/total ·
    /// objective` header plus the planner-decomposed sub-goal checklist.
    fn goal_lines(&self, goal: &hi_agent::Goal, max_steps: usize) -> Vec<Line<'static>> {
        const HARD_CAP: usize = 8;
        let max_steps = max_steps.min(HARD_CAP);
        let total = goal.sub_goals.len();
        let done = goal
            .sub_goals
            .iter()
            .filter(|s| s.status == hi_agent::GoalStatus::Done)
            .count();
        let state = match self.last_drive {
            hi_agent::DriveAction::Idle {
                reason: hi_agent::DriveIdleReason::GoalPaused,
            } => " · paused",
            hi_agent::DriveAction::Idle {
                reason: hi_agent::DriveIdleReason::GoalParked,
            } => " · parked",
            _ if goal.is_paused() => " · paused",
            _ => "",
        };
        let mut header = format!("goal · {done}/{total}{state}");
        if !goal.objective.is_empty() {
            header.push_str(" · ");
            header.push_str(&goal.objective);
        }
        let th = crate::theme::theme();
        let mut out = vec![Line::styled(
            header,
            Style::default()
                .fg(th.accent_goal)
                .add_modifier(Modifier::BOLD),
        )];
        for s in goal.sub_goals.iter().take(max_steps) {
            let (glyph, glyph_style, title_style) = match s.status {
                hi_agent::GoalStatus::Done => ('✓', Style::default().fg(th.accent_success), dim()),
                hi_agent::GoalStatus::Active => (
                    '▸',
                    Style::default()
                        .fg(th.accent_goal)
                        .add_modifier(Modifier::BOLD),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                hi_agent::GoalStatus::Failed => ('✗', Style::default().fg(th.accent_error), dim()),
                // Warning, not error: a blocked step is waiting on the user to
                // supply something, and reading as a failure would send them
                // looking for a defect in work that was never judged.
                hi_agent::GoalStatus::Blocked => {
                    ('⛔', Style::default().fg(th.accent_running), dim())
                }
                hi_agent::GoalStatus::Pending => ('○', dim(), Style::default()),
            };
            out.push(Line::from(vec![
                Span::styled(format!("  {glyph} "), glyph_style),
                Span::styled(s.description.clone(), title_style),
            ]));
        }
        if total > max_steps {
            out.push(Line::styled(
                format!("  … +{} more", total - max_steps),
                dim(),
            ));
        }
        out
    }

    /// Render the full-screen diff review overlay (Ctrl-G). A bordered block
    /// filling the screen, showing the entire working-tree diff with
    /// `diff_lines` coloring, scrollable via j/k/arrows/PgUp/PgDn, with n/p
    /// jumping between `@@` hunk headers. The footer shows the keybindings and
    /// the current scroll position.
    fn render_review(&self, frame: &mut ratatui::Frame, area: ratatui::layout::Rect) {
        let text = self.diff_text.as_deref().unwrap_or("").trim();
        let rendered = if text.is_empty() {
            vec![Line::styled("(no changes in the working tree)", dim())]
        } else {
            diff_lines(text)
        };
        let total = rendered.len();
        // The visible height is the area minus 2 border rows minus 1 footer row.
        let visible = area.height.saturating_sub(3) as usize;
        let max_scroll = total.saturating_sub(visible);
        let scroll = self.review_scroll.min(max_scroll);
        let mut body: Vec<Line<'static>> = rendered
            .iter()
            .skip(scroll)
            .take(visible)
            .cloned()
            .collect();
        // Pad with blank lines so the footer stays at the bottom on short diffs.
        while body.len() < visible {
            body.push(Line::raw(""));
        }
        // Footer: keybindings + scroll position.
        let footer = Line::styled(
            format!(
                " j/k scroll · n/p hunks · PgUp/PgDn · G end · q/Esc close   [{}/{}]",
                scroll + 1,
                total
            ),
            dim(),
        );
        body.push(footer);
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(crate::theme::theme().chrome(UiTone::Info).border)
            .title(" Diff review (Ctrl-G) ");
        frame.render_widget(Paragraph::new(body).block(block), area);
    }

    pub(crate) fn render(&mut self, frame: &mut ratatui::Frame) {
        let area = frame.area();
        let _profile = crate::profiling::FrameTimer::begin("session", area);
        let ui_layout = UiLayout::from_width(area.width);
        let metrics = ui_layout.metrics();
        if let Some(tutorial) = &self.tutorial {
            crate::tutorial::render(frame, area, tutorial);
            return;
        }
        if let Some(overlay) = &self.workflow_overlay {
            crate::workflow_tui::render_overlay(frame, area, overlay);
            return;
        }
        if self.inspect_subagent.is_some() {
            crate::subagent_overlay::render_inspect(frame, area, self);
            return;
        }
        if let Some(overlay) = &self.tasks_overlay {
            crate::subagent_overlay::render_tasks(frame, area, overlay);
            return;
        }
        if self.block_viewer.is_some() {
            crate::block_viewer::render(frame, area, self);
            return;
        }
        if let Some(picker) = &self.jump_picker {
            crate::session_pickers::render_jump(frame, area, picker);
            return;
        }
        if let Some(picker) = &self.rewind_picker {
            crate::session_pickers::render_rewind(frame, area, picker);
            return;
        }
        if let Some(browser) = &self.memory_browser {
            crate::memory_browser::render(frame, area, browser);
            return;
        }
        if let Some(overlay) = &self.diff_lab {
            overlay.render(frame, area);
            return;
        }
        if let Some(overlay) = &self.race {
            overlay.render(frame, area);
            return;
        }
        // Full-screen diff review overlay (Ctrl-G): takes over the whole screen
        // with a scrollable, syntax-colored diff and hunk navigation. Rendered
        // before the normal layout and returned early so it's truly modal.
        if self.mode.is_review() {
            self.render_review(frame, area);
            return;
        }
        // Grok-build chrome floats on the canvas: 2-column side inset, a blank
        // row above the status bar and below the shortcuts, and a one-row gap
        // between chrome and the body. `/btw` is an inline overlay above the
        // prompt, not a side column.
        let (th, theme_revision) = crate::theme::snapshot();
        chrome::fill_background(frame, area, &th);
        let (hpad, top_vpad, bottom_vpad) = chrome::outer_pad(area);
        let inner = chrome::inset(area, hpad, top_vpad, bottom_vpad);
        let composer_w = inner.width;
        let overlay_composer = self.confirmation.is_some()
            || self.plan_approval_capturing()
            || self.fetching.is_some()
            || self.picker.is_some()
            || self.local_directory_prompt.is_some()
            || self.local_picker.is_some()
            || self.local_download_confirmation.is_some()
            || ((self.local_startup_blocked || self.local_startup_error.is_some())
                && self.provider_picker.is_none()
                && self.provider_form.is_none()
                && self.picker.is_none())
            || self.provider_picker.is_some()
            || self.provider_form.is_some();
        let turn_visible = if overlay_composer && self.confirmation.is_none() {
            false
        } else {
            crate::turn_status::build(self, inner.width).is_some()
        };
        let turn_h = u16::from(turn_visible && inner.height >= chrome::STATUS_MIN_HEIGHT);
        let gap_h = u16::from(
            turn_h == 0
                && !overlay_composer
                && inner.height >= chrome::SHORTCUTS_MIN_HEIGHT
                && ui_layout.show_secondary_chrome(),
        );
        let status_h = u16::from(
            inner.height >= chrome::STATUS_MIN_HEIGHT && ui_layout.show_secondary_chrome(),
        );
        let shortcuts_h = u16::from(
            inner.height >= chrome::SHORTCUTS_MIN_HEIGHT
                && ui_layout.show_secondary_chrome()
                && self.show_shortcuts_row(),
        );
        let status_gap = u16::from(status_h > 0 && top_vpad > 0);
        let shortcuts_gap = u16::from(shortcuts_h > 0 && bottom_vpad > 0);
        let chrome_rows = status_h + status_gap + shortcuts_h + shortcuts_gap + turn_h + gap_h;
        // The prompt grows to fit multiline input. Plan + queued follow-ups
        // sit in list panes above the box, not inside it.
        let (input_lines, cursor_row, cursor_col) = self.input_view(composer_w.saturating_sub(2));
        let completion_rows = self.completion_items().len();
        // The compact changed-files summary line. The 20-line Ctrl-D dump is
        // gone — Ctrl-D opens the same full-screen review as Ctrl-G.
        let changed_h = usize::from(!self.last_changed_files.is_empty() && !self.working);
        // The Ctrl-? observability panel: header plus present diagnostic lines.
        let debug_h = if self.show_debug {
            let telemetry_h = if let Some(t) = self.last_telemetry.as_ref() {
                1 + usize::from(t.tool_calls > 0)
                    + usize::from(review_repair_summary(t).is_some())
                    + usize::from(!t.review_repair_exhaustion_reason.is_empty())
            } else {
                0
            };
            4 + telemetry_h + usize::from(fmt_rate_limits(self.rate_limits).is_some())
        } else {
            0
        };
        // The `?` keybindings help overlay: title + section rows from keys table.
        // Clamped so a tall cheat sheet can't starve the transcript on small terms.
        let help_h = if self.show_help {
            let full = 1 + crate::keys::help_overlay_height();
            // Prefer the full cheat sheet when the terminal is tall enough; on
            // short terminals leave at least ~6 rows for transcript + input.
            let room = (area.height as usize).saturating_sub(6);
            full.min(room.max(full.min(20)))
        } else {
            0
        };
        // Command palette rows (filter + up to 12 matches).
        let palette_h = if let Some(p) = &self.palette {
            1 + p.items.len().min(12) + 1 // header + rows + hint
        } else {
            0
        };
        // `/btw` overlay sits above the prompt (grok-build). Hide it while a
        // modal composer (picker/confirm) owns that slot so those stay usable.
        // (Live running-command output no longer reserves its own row here —
        // it renders inside the Run row via live_run_tail_lines.)
        let btw_state = if overlay_composer {
            None
        } else {
            self.btw_overlay()
        };
        let desired_btw = crate::btw::btw_panel_height(btw_state.as_ref(), composer_w);
        let overlay_budget = inner
            .height
            .saturating_sub(chrome_rows + metrics.min_transcript_rows + 3);
        let btw_h = crate::btw::clamp_overlay_height(desired_btw, overlay_budget);
        let btw_gap = crate::btw::gap_before_overlay(btw_h, overlay_budget > btw_h);
        // Height of the input box excluding the plan checklist and the 2 border
        // rows. Used to figure out how many plan steps fit on screen.
        let base_h = changed_h
            + debug_h
            + help_h
            + palette_h
            + usize::from(self.startup_notice.is_some())
            + usize::from(self.checkpoint_warning.is_some())
            + usize::from(self.quit_notice.is_some())
            + completion_rows
            + input_lines.len();
        // The live plan checklist, pinned just above the input (input-bar state
        // only). The step count is capped to what fits on screen so a long plan
        // can't make the box taller than the terminal — ratatui's Layout would
        // otherwise clamp the rect and the Paragraph content would spill past
        // the bottom border. Reserve one row for the transcript (Min(1) below).
        let cap = inner
            .height
            .saturating_sub(
                metrics.min_transcript_rows + chrome_rows + btw_h + btw_gap,
            )
            .max(1) as usize;
        let show_lists = !overlay_composer;
        let input_needed = (base_h + 2).max(3).min(cap);
        let list_budget = cap.saturating_sub(input_needed);
        let queue_block = if show_lists && list_budget > 0 {
            self.queue_pane_lines(3.min(list_budget))
        } else {
            Vec::new()
        };
        let queue_pane_h = queue_block.len().min(list_budget);
        let live_budget = list_budget.saturating_sub(queue_pane_h);
        let live_block = if show_lists && live_budget > 0 {
            self.live_task_lines(3.min(live_budget))
        } else {
            Vec::new()
        };
        let live_pane_h = live_block.len().min(live_budget);
        let avail_inner = list_budget.saturating_sub(queue_pane_h + live_pane_h);
        // plan_h = 1 (header) + steps_shown + (1 if total > steps_shown else 0).
        // Pick the largest step count (up to total and HARD_CAP) whose plan_h
        // fits avail_inner. On tiny terminals avail_inner is 0 so the prompt
        // box keeps a closed border.
        let max_steps =
            if !show_lists || avail_inner == 0 || (self.plan.is_empty() && self.goal.is_none()) {
                0
            } else if !self.plan_pane_expanded {
                0
            } else {
                const HARD_CAP: usize = 8;
                let total = if self.goal.as_ref().is_some_and(|g| !g.sub_goals.is_empty()) {
                    self.goal.as_ref().map(|g| g.sub_goals.len()).unwrap_or(0)
                } else {
                    self.plan.len()
                };
                let upper = total.min(HARD_CAP);
                let mut n = upper;
                while n > 0 && 1 + n + usize::from(total > n) > avail_inner {
                    n -= 1;
                }
                if 1 + n + usize::from(total > n) > avail_inner {
                    0
                } else {
                    n
                }
            };
        let plan_block = if show_lists && avail_inner > 0 {
            let mut lines = self.plan_lines(max_steps);
            lines.truncate(avail_inner);
            lines
        } else {
            Vec::new()
        };
        let plan_pane_h = plan_block.len();
        let composer_max = inner
            .height
            .saturating_sub(
                chrome_rows
                    + metrics.min_transcript_rows
                    + btw_h
                    + btw_gap
                    + plan_pane_h as u16
                    + live_pane_h as u16
                    + queue_pane_h as u16,
            )
            .max(1);
        let input_h = if self.confirmation.is_some() || self.plan_approval_capturing() {
            inner
                .height
                .saturating_sub(3 + chrome_rows)
                .clamp(12, 28)
                .min(composer_max)
        } else if self.fetching.is_some() {
            3.min(composer_max)
        } else if let Some(p) = &self.picker {
            // filter line + visible model rows + borders, bounded by the screen.
            let rows = p.matches.len().clamp(1, PICKER_ROWS) as u16;
            (rows + 3).min(composer_max)
        } else if self.local_directory_prompt.is_some() {
            5.min(composer_max)
        } else if let Some(p) = &self.local_picker {
            let rows = (p.matches.len().clamp(1, PICKER_ROWS) + 2) as u16;
            (rows + 2).min(composer_max)
        } else if (self.local_startup_blocked || self.local_startup_error.is_some())
            && self.provider_picker.is_none()
            && self.provider_form.is_none()
            && self.picker.is_none()
        {
            5.min(composer_max)
        } else if let Some(p) = &self.provider_picker {
            // filter line + visible rows + borders, bounded by the screen.
            let rows = p.matches.len().clamp(1, PICKER_ROWS) as u16;
            (rows + 3).min(composer_max)
        } else if let Some(form) = &self.provider_form {
            // Provider row + hint + blank spacer + text fields + borders. The
            // API-key field is hidden for Ollama, so subtract one there.
            let fields = if form.api_key_unneeded() { 3 } else { 4 };
            let form_rows = if ui_layout == UiLayout::Tiny {
                // Tiny terminals keep the provider selector and every field,
                // but drop the explanatory hint and spacer rows.
                fields + 3
            } else {
                fields + 5
            };
            (form_rows as u16).min(cap as u16)
        } else {
            input_needed as u16
        };
        let session = Layout::vertical([
            Constraint::Length(status_h),
            Constraint::Length(status_gap),
            Constraint::Min(1),
            Constraint::Length(shortcuts_gap),
            Constraint::Length(shortcuts_h),
        ])
        .split(inner);
        let status_area = session[0];
        let body_area = session[2];
        let shortcuts_area = session[4];
        let rows = Layout::vertical([
            Constraint::Length(live_pane_h as u16),
            Constraint::Min(1),
            Constraint::Length(gap_h),
            Constraint::Length(turn_h),
            Constraint::Length(btw_gap),
            Constraint::Length(btw_h),
            Constraint::Length(plan_pane_h as u16),
            Constraint::Length(queue_pane_h as u16),
            Constraint::Length(input_h),
        ])
        .split(body_area);
        let live_area = rows[0];
        let transcript_area = rows[1];
        let turn_area = rows[3];
        let btw_area = rows[5];
        let plan_area = rows[6];
        let queue_area = rows[7];
        let composer_area = rows[8];

        let prompt_count = self
            .transcript
            .iter()
            .filter(|e| matches!(e, crate::TranscriptEntry::UserPrompt { .. }))
            .count();
        let show_rail =
            crate::timeline::visible(self.timeline_enabled, transcript_area, prompt_count);
        let (rail_rect, transcript_area) = if show_rail {
            let cols = Layout::horizontal([
                Constraint::Length(crate::timeline::RAIL_WIDTH),
                Constraint::Min(1),
            ])
            .split(transcript_area);
            (Some(cols[0]), cols[1])
        } else {
            (None, transcript_area)
        };

        // --- Transcript ---
        // Status bar: cwd on the left, chips on the right, grok-build's `│`
        // separators. Model identity lives in the prompt's bottom divider.
        let mut info_spans: Vec<Span<'static>> = Vec::new();
        if let Some(goal) = &self.goal {
            let total = goal.sub_goals.len();
            if total > 0 {
                let done = goal
                    .sub_goals
                    .iter()
                    .filter(|s| s.status == hi_agent::GoalStatus::Done)
                    .count();
                let label = if goal.paused {
                    format!("[Goal: {done}/{total} ⏸]")
                } else if done == total {
                    "[Goal: Done]".to_string()
                } else {
                    format!("[Goal: {done}/{total}]")
                };
                chrome::push_chip(
                    &mut info_spans,
                    &th,
                    Span::styled(label, Style::default().fg(th.accent_goal)),
                );
            }
        }
        if ui_layout.show_secondary_chrome() && self.execution.is_durable() {
            chrome::push_chip(
                &mut info_spans,
                &th,
                Span::styled("durable", Style::default().fg(th.accent_success)),
            );
        }
        if ui_layout.show_secondary_chrome() {
            let sandbox = chrome::sandbox_chip();
            let sandbox_fg = if sandbox == "sandbox off" || sandbox == "sandbox?" {
                th.warning
            } else {
                th.gray_dim
            };
            chrome::push_chip(
                &mut info_spans,
                &th,
                Span::styled(sandbox, Style::default().fg(sandbox_fg)),
            );
            chrome::push_chip(
                &mut info_spans,
                &th,
                Span::styled("undo", Style::default().fg(th.gray_dim)),
            );
        }
        if let Some(runtime) = &self.local_runtime {
            let state = if runtime.ready { "ready" } else { "starting" };
            chrome::push_chip(
                &mut info_spans,
                &th,
                Span::styled(
                    format!("local MLX · {state}"),
                    Style::default().fg(if runtime.ready {
                        th.accent_success
                    } else {
                        th.warning
                    }),
                ),
            );
        }
        if ui_layout.show_secondary_chrome()
            && let Some(pct) = self.context_pct()
        {
            let color = if pct >= 80 { th.warning } else { th.gray };
            let hovered = crate::btw::cell_in(self.ctx_chip_rect, self.mouse_col, self.mouse_row);
            let label = if let Some(window) = self.context_window.filter(|&w| w > 0) {
                chrome::context_usage_chip(self.context_used, u64::from(window), hovered)
            } else {
                chrome::context_chip(pct, hovered)
            };
            chrome::push_chip(
                &mut info_spans,
                &th,
                Span::styled(label, Style::default().fg(color)),
            );
        }
        if !self.queue.is_empty() && ui_layout.show_secondary_chrome() {
            let queue_label = if ui_layout == UiLayout::Narrow {
                format!("q{}", self.queue.len())
            } else {
                format!("queue {} · Alt-↑/↓", self.queue.len())
            };
            chrome::push_chip(
                &mut info_spans,
                &th,
                Span::styled(queue_label, Style::default().fg(th.accent_running)),
            );
        }
        let info = Line::from(info_spans);
        let cwd_budget = status_area
            .width
            .saturating_sub(info.width() as u16)
            .saturating_sub(1) as usize;
        let title = Line::from(Span::styled(
            chrome::display_cwd(&self.workspace_root, cwd_budget.max(4)),
            Style::default().fg(th.gray_dim),
        ));
        // Rebuild flatten+wrap cache only when transcript structure / width /
        // fold toggles / density / pending stream / block-nav selection change.
        // Spinner ticks reuse the cache.
        let inner_w_probe = transcript_area.width;
        let selected_block = self.mode.is_block_nav().then(|| self.selected_block_ord());
        self.ensure_view_cache_with_revision(inner_w_probe, selected_block, theme_revision);

        // Borrow the immutable cache and clone only the visible window below.
        // The full line/prefix maps are retained for interaction geometry and
        // are copied only when the cache identity changes.
        let nav_line_target = selected_block.and_then(|sel| {
            self.view_cache
                .block_line_ranges
                .iter()
                .find(|&&(_, _, o)| o == sel)
                .map(|&(s, _, _)| s.saturating_sub(1)) // marker sits just above body
        });

        // Apply selection highlighting on a working copy of the visible window only.
        let inner_w = transcript_area.width;
        let inner_h = transcript_area.height;
        let total = self.view_cache.total_rows();
        let max_scroll = total.saturating_sub(inner_h);
        self.view_max_scroll = max_scroll;
        self.view_total = total;

        // Block-nav follows the cursor.
        if self.mode.is_block_nav()
            && let Some(t) = nav_line_target
        {
            let want = self
                .view_cache
                .prefix
                .get(t)
                .copied()
                .unwrap_or(0)
                .saturating_sub(2);
            self.scroll = want.min(max_scroll as u32) as u16;
            self.following = false;
        }
        if self.page_flip_on_send && self.working {
            if let Some(&idx) = self.view_cache.prompt_line_starts.last() {
                self.scroll = self.view_cache.prefix.get(idx).copied().unwrap_or(0) as u16;
                self.following = false;
            }
        }
        let scroll = if self.following {
            self.page_flip_on_send = false;
            max_scroll
        } else {
            self.scroll.min(max_scroll)
        };

        // Virtualize: only paint the viewport ± overscan lines.
        const OVERSCAN: usize = 8;
        let (line_lo, line_hi, scroll_adj) = crate::view_cache::visible_line_window(
            &self.view_cache.prefix,
            scroll,
            inner_h,
            OVERSCAN,
        );
        let mut lines: Vec<Line<'static>> = self.view_cache.lines[line_lo..line_hi].to_vec();

        // Sunken panels: pad panel-tagged tool-output lines to full width.
        let panel_bg = th.panel;
        if th.paints_backgrounds() {
            for line in &mut lines {
                if line.style.bg == Some(panel_bg) {
                    let used: usize = line
                        .spans
                        .iter()
                        .map(|s| display_width(s.content.as_ref()))
                        .sum();
                    if used < inner_w as usize {
                        let pad = inner_w as usize - used;
                        line.spans
                            .push(Span::styled(" ".repeat(pad), Style::default().bg(panel_bg)));
                    }
                }
            }
        }
        // Mouse text selection highlight (absolute line indices → window-local).
        let sel = th.selection_bg;
        if let Some((line_idx, clo, chi)) = self.char_span() {
            if line_idx >= line_lo && line_idx < line_hi {
                let local = line_idx - line_lo;
                if let Some(line) = lines.get_mut(local) {
                    highlight_char_range(line, clo, chi, sel);
                }
            }
        } else if let Some((lo, hi)) = self.selection_range() {
            let last_abs = self.view_cache.lines.len().saturating_sub(1);
            let lo = lo.min(last_abs);
            let hi = hi.min(last_abs);
            for abs in lo..=hi {
                if abs >= line_lo && abs < line_hi {
                    let local = abs - line_lo;
                    if let Some(line) = lines.get_mut(local) {
                        line.style = line.style.bg(sel);
                        let used: usize = line
                            .spans
                            .iter()
                            .map(|s| display_width(s.content.as_ref()))
                            .sum();
                        for span in &mut line.spans {
                            span.style = span.style.bg(sel);
                        }
                        if used < inner_w as usize {
                            line.spans.push(Span::styled(
                                " ".repeat(inner_w as usize - used),
                                Style::default().bg(sel),
                            ));
                        }
                    }
                }
            }
        }

        if self.timestamps_enabled {
            let stamps: Vec<String> = self
                .transcript
                .iter()
                .filter_map(|entry| match entry {
                    crate::TranscriptEntry::UserPrompt { at, .. } => {
                        Some(crate::util::fmt_clock(*at))
                    }
                    _ => None,
                })
                .collect();
            for (i, &abs) in self.view_cache.prompt_line_starts.iter().enumerate() {
                if abs >= line_lo
                    && abs < line_hi
                    && let (Some(line), Some(stamp)) = (lines.get_mut(abs - line_lo), stamps.get(i))
                {
                    chrome::overlay_right(line, stamp, inner_w, dim());
                }
            }
        }

        // Cache geometry for mouse click / drag outside render.
        self.view_inner = ratatui::layout::Rect {
            x: transcript_area.x,
            y: transcript_area.y,
            width: inner_w,
            height: inner_h,
        };
        self.view_scroll = scroll;
        // Keep full maps for selection copy (absolute indices), but only copy
        // them when flatten/wrap inputs changed. This is the common spinner,
        // scroll, and cursor redraw fast path.
        let cache_key = self.view_cache.key();
        if self.view_geometry_key != Some(cache_key) {
            self.block_row_spans = self
                .view_cache
                .block_line_ranges
                .iter()
                .map(|&(s, e, o)| (self.view_cache.prefix[s], self.view_cache.prefix[e], o))
                .collect();
            self.view_line_texts = self
                .view_cache
                .lines
                .iter()
                .map(crate::render::line_text)
                .collect();
            self.view_prefix = self.view_cache.prefix.clone();
            self.view_geometry_key = Some(cache_key);
        }

        // Sticky header: most recent prompt strictly above the viewport.
        let sticky_prompt: Option<Line<'static>> = if self.following {
            None
        } else {
            self.view_cache
                .prompt_line_starts
                .iter()
                .enumerate()
                .rev()
                .find(|(_, idx)| {
                    (self.view_cache.prefix.get(**idx).copied().unwrap_or(0) as u16) < scroll
                })
                .and_then(|(prompt_i, idx)| {
                    let mut line = self.view_cache.lines.get(*idx).cloned()?;
                    if self.timestamps_enabled
                        && let Some(at) = self
                            .transcript
                            .iter()
                            .filter_map(|e| match e {
                                crate::TranscriptEntry::UserPrompt { at, .. } => Some(*at),
                                _ => None,
                            })
                            .nth(prompt_i)
                    {
                        chrome::overlay_right(
                            &mut line,
                            &crate::util::fmt_clock(at),
                            inner_w,
                            dim(),
                        );
                    }
                    Some(line)
                })
        };

        let mut pad = Block::new();
        if th.paints_backgrounds() {
            pad = pad.style(Style::default().bg(th.bg_base).fg(th.text_primary));
        }
        let para = if total == 0 && !self.working {
            let home = chrome::WelcomeHome {
                location: chrome::welcome_location(
                    &self.workspace_root,
                    self.git_branch.as_deref(),
                ),
                sessions: &self.session_completion_cache,
            };
            Paragraph::new(chrome::welcome_lines(transcript_area, &th, Some(&home)))
                .alignment(ratatui::layout::Alignment::Center)
                .block(pad)
        } else {
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .block(pad)
                .scroll((scroll_adj, 0))
        };
        frame.render_widget(para, transcript_area);

        if let Some(rail_rect) = rail_rect {
            let rail = crate::timeline::compute(
                rail_rect,
                &self.view_cache,
                scroll,
                transcript_area.height,
            );
            crate::timeline::render(frame, &rail);
            self.timeline_hits = rail.hits;
            self.timeline_rect = rail.rect;
        } else {
            self.timeline_hits.clear();
            self.timeline_rect = ratatui::layout::Rect::default();
        }

        let mut status_right = info;
        if !self.following {
            let new = total.saturating_sub(self.total_when_unpinned);
            let label = if new > 0 {
                format!("↓{new} new")
            } else {
                format!("↓{}", max_scroll.saturating_sub(scroll))
            };
            if !status_right.spans.is_empty() {
                status_right.spans.push(chrome::chip_sep(&th));
            }
            status_right.spans.push(Span::styled(
                label,
                Style::default()
                    .fg(th.selection)
                    .add_modifier(Modifier::BOLD)
                    .bg(if th.paints_backgrounds() {
                        th.bg_base
                    } else {
                        Color::Reset
                    }),
            ));
        }
        chrome::render_status_bar(frame, status_area, title, status_right.clone(), &th);
        self.ctx_chip_rect = chrome::span_hit(status_area, &status_right, |text| {
            text.contains(" / ")
                || text.contains("% ctx")
                || (text.contains('#') && text.contains('-'))
        });
        if turn_h > 0
            && let Some(line) = self.turn_status_line(turn_area.width)
        {
            chrome::render_turn_status(frame, turn_area, line, &th);
            self.turn_status_rect = turn_area;
        } else {
            self.turn_status_rect = ratatui::layout::Rect::default();
        }
        if btw_h > 0 {
            if let Some(state) = &btw_state {
                self.last_btw_area = btw_area;
                self.last_btw_close =
                    crate::btw::render_btw_panel(frame, state, btw_area, self.spinner, &th);
            }
        } else {
            self.last_btw_area = ratatui::layout::Rect::default();
            self.last_btw_close = ratatui::layout::Rect::default();
        }

        if plan_pane_h > 0 {
            frame.render_widget(Paragraph::new(plan_block), plan_area);
        }
        if live_pane_h > 0 {
            frame.render_widget(Paragraph::new(live_block), live_area);
        }
        if queue_pane_h > 0 {
            frame.render_widget(Paragraph::new(queue_block), queue_area);
        }

        // Overlay the sticky prompt header on the top inner row, so scrolling
        // through long output always shows which prompt it belongs to. A subtle
        // band (truecolor) marks it as pinned rather than in-flow content.
        if let Some(mut sticky) = sticky_prompt
            && inner_h >= 1
            && inner_w >= 1
        {
            if th.paints_backgrounds() {
                sticky.style = sticky.style.bg(th.band_user);
                let used: usize = sticky
                    .spans
                    .iter()
                    .map(|s| display_width(s.content.as_ref()))
                    .sum();
                if used < inner_w as usize {
                    sticky.spans.push(Span::styled(
                        " ".repeat(inner_w as usize - used),
                        Style::default().bg(th.band_user),
                    ));
                }
            }
            let sticky_area = ratatui::layout::Rect {
                x: transcript_area.x,
                y: transcript_area.y,
                width: inner_w,
                height: 1,
            };
            frame.render_widget(Paragraph::new(vec![sticky]), sticky_area);
        }

        // --- Bottom region: a fetch/plan spinner, the model picker, or the input bar. ---
        if let Some(model) = &self.local_download_confirmation {
            let block = th
                .panel_block(" download local MLX model? ", UiTone::Warning)
                .title_bottom(
                    Line::styled(" Enter/y start · n/Esc cancel ", dim()).right_aligned(),
                );
            let detail = crate::local_picker::option_detail(model);
            let body = vec![
                Line::styled(
                    model.display_name.to_string(),
                    Style::default().fg(th.warning).add_modifier(Modifier::BOLD),
                ),
                Line::styled(detail, th.text_secondary),
                Line::styled(
                    "The model will download in the background and can be resumed later.",
                    dim(),
                ),
            ];
            frame.render_widget(Paragraph::new(body).block(block), composer_area);
        } else if let Some(path) = &self.local_directory_prompt {
            let block = th
                .panel_block(" existing MLX directory ", UiTone::Info)
                .title_bottom(Line::styled(" Enter start · Esc cancel ", dim()).right_aligned());
            let body = vec![
                Line::styled(
                    "Path (supports ~ and workspace-relative paths):",
                    Style::default().fg(th.text_secondary),
                ),
                Line::from(vec![
                    Span::styled("› ", th.chrome(UiTone::Active).selected),
                    Span::raw(path.clone()),
                ]),
            ];
            frame.render_widget(Paragraph::new(body).block(block), composer_area);
            let cx = composer_area.x + 3 + display_width(path) as u16;
            frame.set_cursor_position((
                cx.min(composer_area.right().saturating_sub(2)),
                composer_area.y + 2,
            ));
        } else if let Some(p) = &self.local_picker {
            let block = th
                .panel_block(" local MLX models ", UiTone::Info)
                .title_top(
                    Line::from(format!(" {}/{} ", p.selected + 1, p.matches.len().max(1)))
                        .right_aligned(),
                );
            let mut plines: Vec<Line> = vec![Line::from(vec![
                Span::styled("filter: ", dim()),
                Span::raw(p.filter.clone()),
                Span::styled(
                    "   ↑↓ select · Enter inspect/start · d existing directory · Esc cancel",
                    Style::default().fg(th.gray_dim),
                ),
            ])];
            for (name, model, selected) in p.visible().into_iter().take(PICKER_ROWS) {
                let name = name.unwrap_or("Use existing MLX directory…");
                let detail = model
                    .map(crate::local_picker::option_detail)
                    .unwrap_or_else(|| "validate a local config.json and weight shards".into());
                if selected {
                    plines.push(Line::from(vec![
                        Span::styled(format!("▶ {name}"), th.chrome(UiTone::Active).selected),
                        Span::styled(format!("  {detail}"), th.chrome(UiTone::Warning).body),
                    ]));
                } else {
                    plines.push(Line::from(vec![
                        Span::raw(format!("  {name}")),
                        Span::styled(format!("  {detail}"), th.chrome(UiTone::Muted).hint),
                    ]));
                }
            }
            if p.matches.is_empty() {
                plines.push(Line::styled("  (no matching local models)", dim()));
            }
            frame.render_widget(Paragraph::new(plines).block(block), composer_area);
            let cx = composer_area.x + 1 + 8 + display_width(&p.filter) as u16;
            frame.set_cursor_position((
                cx.min(composer_area.right().saturating_sub(2)),
                composer_area.y + 1,
            ));
        } else if (self.local_startup_blocked || self.local_startup_error.is_some())
            && self.provider_picker.is_none()
            && self.provider_form.is_none()
            && self.picker.is_none()
        {
            let block = th
                .panel_block(" local MLX startup ", UiTone::Warning)
                .title_bottom(
                    Line::styled(" r retry · f fallback · /provider choose · /quit ", dim())
                        .right_aligned(),
                );
            let detail = self
                .local_startup_error
                .as_deref()
                .map(|error| format!("startup failed: {error}"))
                .unwrap_or_else(|| "loading the persisted model; prompts are paused".into());
            let model = self
                .local_runtime
                .as_ref()
                .map(|runtime| runtime.model_id.as_str())
                .unwrap_or("unknown model");
            let body = vec![
                Line::styled(format!("{model} · {detail}"), th.text_secondary),
                Line::styled(
                    "The previous provider remains active until local MLX is ready or you choose another route.",
                    dim(),
                ),
            ];
            frame.render_widget(Paragraph::new(body).block(block), composer_area);
        } else if self.plan_approval_capturing() {
            crate::plan_approval::render(frame, composer_area, self);
        } else if let Some(request) = &self.confirmation {
            let details = request.details();
            let all = confirmation_lines(request, &details);
            let options = crate::confirm_overlay::option_lines(self, request);
            let visible = composer_area
                .height
                .saturating_sub(6 + options.len() as u16) as usize;
            let max_scroll = all.len().saturating_sub(visible.max(1));
            let scroll = self.confirmation_scroll.min(max_scroll);
            let is_ask = matches!(request, hi_agent::ConfirmationRequest::AskUser { .. });
            let mut body = vec![Line::styled(
                if is_ask {
                    "The agent needs a decision before it can continue."
                } else {
                    "This action can change your workspace. Review it before approving."
                },
                Style::default().fg(th.warning).add_modifier(Modifier::BOLD),
            )];
            if is_ask {
                // Options replace the numbered dump from confirmation_lines.
                if let hi_agent::ConfirmationRequest::AskUser { question, .. } = request {
                    body.push(Line::styled(
                        question.clone(),
                        Style::default()
                            .fg(th.text_primary)
                            .add_modifier(Modifier::BOLD),
                    ));
                    body.push(Line::raw(""));
                }
                body.extend(options);
                body.push(Line::raw(""));
                body.push(Line::from(vec![
                    Span::styled("answer: ", dim()),
                    Span::raw(self.ask_user_draft.clone()),
                ]));
            } else {
                body.extend(all.iter().skip(scroll).take(visible.max(1)).cloned());
                body.push(Line::raw(""));
                body.extend(options);
                if self.confirm_focus == crate::confirm_overlay::ConfirmFocus::Followup {
                    body.push(Line::raw(""));
                    body.push(Line::from(vec![
                        Span::styled("follow-up: ", dim()),
                        Span::raw(self.ask_user_draft.clone()),
                    ]));
                }
            }
            let hint = crate::confirm_overlay::hint(
                request,
                self.confirm_focus,
                self.confirmation_waiting,
            );
            let block = th
                .panel_block(request.title(), UiTone::Warning)
                .title_bottom(Line::styled(hint, dim()));
            frame.render_widget(
                Paragraph::new(body).block(block).wrap(Wrap { trim: false }),
                composer_area,
            );
        } else if let Some(started) = self.fetching.or(self.planning) {
            let frame_ch = SPINNER[self.spinner % SPINNER.len()];
            let elapsed = fmt_elapsed(started.elapsed().as_secs());
            let label = if self.planning.is_some() {
                "planning goal with the planner model…".to_string()
            } else {
                format!("fetching models from {}…", self.provider)
            };
            let block = th.panel_block("", UiTone::Info);
            let body = Line::from(vec![
                Span::styled(
                    format!("{frame_ch} {label} {elapsed}"),
                    Style::default()
                        .fg(crate::theme::theme().accent_system)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("   Esc to cancel", dim()),
            ]);
            frame.render_widget(Paragraph::new(body).block(block), composer_area);
        } else if let Some(p) = &self.picker {
            let block = th
                .panel_block(
                    if self.session_picker {
                        " sessions "
                    } else {
                        " select a model "
                    },
                    UiTone::Info,
                )
                .title_top(
                    Line::from(format!(" {}/{} ", p.selected + 1, p.matches.len().max(1)))
                        .right_aligned(),
                );
            let mut plines: Vec<Line> = vec![Line::from(vec![
                Span::raw(format!("filter: {}", p.filter)),
                Span::styled(
                    if self.session_picker {
                        "   ↑↓/wheel move · type to search · Enter switch · r rename · f favorite · a archive · d delete · Esc cancel"
                    } else {
                        "   ↑↓ move · type to filter · Enter select · Esc cancel"
                    },
                    dim(),
                ),
            ])];
            let (_, visible) = p.visible();
            if visible.is_empty() {
                plines.push(Line::styled("  (no matches)".to_string(), dim()));
            }
            for row in visible {
                let mut tag = String::new();
                if self.session_picker
                    && let Some((favorite, archived)) = self.session_catalog_flags.get(row.id)
                {
                    if *favorite {
                        tag.push_str(" ★");
                    }
                    if *archived {
                        tag.push_str(" [archived]");
                    }
                }
                if row.id == p.current {
                    tag.push_str(" (current)");
                }
                let caps = if self.session_picker {
                    String::new()
                } else {
                    display_capabilities(row.meta)
                };
                if !caps.is_empty() {
                    tag.push_str(&format!(" {{{caps}}}"));
                }
                // Price + window columns, right-aligned after the id.
                let price = if self.session_picker {
                    String::new()
                } else {
                    display_price(row.meta)
                };
                let window = if self.session_picker {
                    String::new()
                } else {
                    display_window(row.meta)
                };
                let meta_col = if price.is_empty() && window.is_empty() {
                    String::new()
                } else {
                    format!("  {price:>8}  {window:>5}")
                };
                if row.selected {
                    plines.push(Line::from(vec![
                        Span::styled(format!("▶ {}", row.id), th.chrome(UiTone::Active).selected),
                        Span::styled(meta_col, th.chrome(UiTone::Warning).body),
                        Span::styled(tag, th.chrome(UiTone::Muted).hint),
                    ]));
                } else {
                    plines.push(Line::from(vec![
                        Span::raw(format!("  {}", row.id)),
                        Span::styled(meta_col, th.chrome(UiTone::Muted).hint),
                        Span::styled(tag, th.chrome(UiTone::Muted).hint),
                    ]));
                }
            }
            frame.render_widget(Paragraph::new(plines).block(block), composer_area);
            // Cursor on the filter line, just after "filter: <text>".
            let cx = composer_area.x + 1 + 8 + display_width(&p.filter) as u16;
            frame.set_cursor_position((
                cx.min(composer_area.right().saturating_sub(2)),
                composer_area.y + 1,
            ));
        } else if let Some(p) = &self.provider_picker {
            let block = th.panel_block(" provider ", UiTone::Info);
            let mut plines: Vec<Line> = vec![Line::from(vec![
                Span::styled("filter: ", dim()),
                Span::raw(p.filter.clone()),
                Span::styled(
                    "   ↑↓ select · Enter switch · Esc cancel",
                    Style::default().fg(crate::theme::theme().gray_dim),
                ),
            ])];
            for (name, detail, is_preset, is_local, is_active, is_highlighted) in
                p.visible().into_iter().take(PICKER_ROWS)
            {
                // The active entry keeps its marker even when the highlight is
                // elsewhere, so arrowing around never loses track of what's live.
                let mark = if is_active { "●" } else { " " };
                let kind = if is_local {
                    "local model"
                } else if is_preset {
                    "provider"
                } else {
                    "profile"
                };
                if is_highlighted {
                    plines.push(Line::from(vec![
                        Span::styled(
                            format!("▶{mark} {name}"),
                            th.chrome(UiTone::Active).selected,
                        ),
                        Span::styled(format!("  [{kind}]"), th.chrome(UiTone::Warning).body),
                        Span::styled(format!("  {detail}"), th.chrome(UiTone::Muted).hint),
                    ]));
                } else {
                    plines.push(Line::from(vec![
                        Span::raw(format!(" {mark} {name}")),
                        Span::styled(format!("  [{kind}]"), th.chrome(UiTone::Muted).hint),
                        Span::styled(format!("  {detail}"), th.chrome(UiTone::Muted).hint),
                    ]));
                }
            }
            frame.render_widget(Paragraph::new(plines).block(block), composer_area);
            let cx = composer_area.x + 1 + 8 + display_width(&p.filter) as u16;
            frame.set_cursor_position((
                cx.min(composer_area.right().saturating_sub(2)),
                composer_area.y + 1,
            ));
        } else if let Some(form) = &self.provider_form {
            let title = if form.editing {
                " edit provider "
            } else {
                " add provider "
            };
            let block = th.panel_block(title, UiTone::Info);
            let choices = form.provider_choices();
            let pidx = form.provider_idx();
            let mut lines: Vec<Line> = Vec::new();

            // Provider row: show only the current choice, with ‹ › marking it
            // as cyclable. Listing every option inline crowded the line and put
            // a second "▶" on screen competing with the active-field marker.
            let current_label = choices.get(pidx).map(|(_, label)| *label).unwrap_or("");
            lines.push(Line::from(vec![
                Span::styled("  Provider   ", th.chrome(UiTone::Warning).title),
                Span::styled(
                    format!("‹ {current_label} ›"),
                    th.chrome(UiTone::Active).title,
                ),
                Span::styled(
                    format!("  ({} of {})", pidx + 1, choices.len()),
                    th.chrome(UiTone::Muted).hint,
                ),
            ]));
            if ui_layout != UiLayout::Tiny {
                lines.push(Line::styled(
                    "  ↑↓ change provider · Tab next field · Enter save · Esc cancel".to_string(),
                    th.chrome(UiTone::Muted).hint,
                ));
                lines.push(Line::raw(""));
            }

            // Text fields.
            let unneeded = form.api_key_unneeded();
            for (i, (label, placeholder, value, is_active)) in
                form.field_labels().into_iter().enumerate()
            {
                // Skip rendering the API-key field entirely for Ollama — it
                // would be a confusing, unusable field the user might try to fill.
                if i == 1 && unneeded {
                    continue;
                }
                let display = if value.is_empty() && !placeholder.is_empty() {
                    placeholder.clone()
                } else {
                    value.clone()
                };
                let prefix = if is_active { "▶ " } else { "  " };
                let val_span = if value.is_empty() && !placeholder.is_empty() {
                    Span::styled(display, th.chrome(UiTone::Muted).hint)
                } else if is_active {
                    Span::styled(display, th.chrome(UiTone::Active).body)
                } else {
                    Span::raw(display)
                };
                // Pad labels to a fixed column so the values line up; ragged
                // "Name:" / "Base URL:" starts were most of the visual noise.
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("{prefix}{label:<FORM_LABEL_WIDTH$} "),
                        if is_active {
                            th.chrome(UiTone::Warning).title
                        } else {
                            th.chrome(UiTone::Muted).hint
                        },
                    ),
                    val_span,
                ]));
            }

            frame.render_widget(Paragraph::new(lines).block(block), composer_area);

            // Cursor on the active text field.
            let active_idx = form.active();
            // Account for the hidden API-key field (index 1) when computing the
            // display row: fields after it shift up by one.
            let hidden_before = if form.api_key_unneeded() && active_idx > 1 {
                1
            } else {
                0
            };
            // Border + provider row + hint row + blank spacer.
            let field_offset = if ui_layout == UiLayout::Tiny { 1 } else { 3 };
            let cy = (composer_area.y + 1 + field_offset + (active_idx - hidden_before) as u16)
                .min(composer_area.bottom().saturating_sub(2));
            let prefix_len = 2 + FORM_LABEL_WIDTH + 1; // "▶ " + padded label + " "
            let cx = composer_area.x + 1 + prefix_len as u16 + form.active_cursor_width() as u16;
            frame.set_cursor_position((cx.min(composer_area.right().saturating_sub(2)), cy));
        } else {
            // Grok's prompt: a quiet rounded frame on the terminal background,
            // with the model right-aligned on the bottom divider (`╰── gpt-4o ╯`)
            // and turn state conveyed by the status row rather than a title badge.
            let th = crate::theme::theme();
            let border = if self.plan_mode {
                Style::default().fg(th.accent_plan)
            } else {
                th.input_border(true)
            };
            let flags = self.composer_flags();
            let bg = if th.paints_backgrounds() {
                th.bg_base
            } else {
                Color::Reset
            };
            let model = truncate_display(&self.model, 28);
            let mut bottom: Vec<Span> = vec![Span::styled(" ", Style::default().bg(bg))];
            bottom.push(Span::styled(
                model.clone(),
                Style::default().fg(th.accent_model).bg(bg),
            ));
            let mut used = 2 + display_width(&model);
            for flag in &flags {
                let extra = 3 + display_width(flag);
                if used + extra + 2 > composer_area.width as usize {
                    break;
                }
                bottom.push(Span::styled(" · ", dim().bg(bg)));
                let color = if *flag == "plan" {
                    th.accent_plan
                } else {
                    th.warning
                };
                bottom.push(Span::styled(*flag, Style::default().fg(color).bg(bg)));
                used += extra;
            }
            bottom.push(Span::styled(" ", Style::default().bg(bg)));
            let mut input_block = Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(border)
                .title_bottom(Line::from(bottom).right_aligned());
            if th.paints_backgrounds() {
                input_block = input_block.style(Style::default().bg(th.bg_base));
            }

            let mut ilines: Vec<Line> = Vec::new();
            // Ctrl-R reverse history search overlay: shows the query, the match
            // count, and a few recent matches above the input line.
            if let Some(search) = self.mode.history_search() {
                let count = search.matches.len();
                let preview = search
                    .current()
                    .and_then(|i| self.input.history.get(i))
                    .map(|s| s.replace('\n', " "))
                    .unwrap_or_default();
                // Char-based truncation: history entries are arbitrary input,
                // and a byte slice panics on a multi-byte char at the cut.
                let preview = truncate_display(&preview, 60);
                ilines.push(Line::from(vec![
                    Span::styled(
                        "reverse-i-search: ",
                        Style::default().fg(crate::theme::theme().accent_success),
                    ),
                    Span::styled(
                        search.query.clone(),
                        Style::default().add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!("  ({count} match{})", if count == 1 { "" } else { "es" }),
                        dim(),
                    ),
                ]));
                ilines.push(Line::styled(format!("  → {preview}"), dim()));
            }
            // A compact "changed: …" line so the user always sees what the last
            // turn touched, without opening review or scrolling.
            if !self.last_changed_files.is_empty() && !self.working {
                let summary = self
                    .last_changed_files
                    .iter()
                    .map(|p| p.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                ilines.push(Line::styled(
                    format!("changed: {summary}  (Ctrl-G for review)"),
                    dim(),
                ));
            }
            // The Ctrl-? agent-observability panel: trajectory telemetry, tool
            // calls this turn, and context composition. Read-only diagnostics.
            if self.show_debug {
                ilines.push(Line::styled(
                    "agent (Ctrl-? to close)".to_string(),
                    Style::default()
                        .fg(crate::theme::theme().accent_system)
                        .add_modifier(Modifier::BOLD),
                ));
                let t = self.last_telemetry.as_ref();
                let tel = if let Some(t) = t {
                    format!(
                        "telemetry: {} verify · {} retry · {} repeat · {} continue · {} trunc{}",
                        t.verify_rounds,
                        t.recovery_retries,
                        t.repeat_nudges,
                        t.continue_nudges,
                        t.truncation_retries,
                        if t.stalled_unfinished || t.stalled_repeating {
                            " · stalled"
                        } else {
                            ""
                        }
                    )
                } else {
                    "telemetry: (no turn yet)".to_string()
                };
                ilines.push(Line::styled(tel, dim()));
                if let Some(phase) = self.last_turn_phase {
                    ilines.push(Line::styled(format!("phase: {phase}"), dim()));
                }
                if let Some(t) = self.last_telemetry.as_ref() {
                    ilines.push(Line::styled(
                        format!(
                            "evidence: {} · reads {} · searches {} · listing_only {} · repair {}",
                            t.discovery_depth,
                            t.file_reads,
                            t.targeted_searches,
                            t.listing_only,
                            t.quality_repair_nudges
                        ),
                        dim(),
                    ));
                    if let Some(repair) = review_repair_summary(t) {
                        for chunk in repair.lines() {
                            ilines.push(Line::styled(chunk.to_string(), dim()));
                        }
                    }
                }
                // Scheduler parallelism: max concurrent batch and serial share.
                let sched = if let Some(t) = self.last_telemetry.as_ref() {
                    if t.tool_calls > 0 {
                        format!(
                            "scheduler: {} calls · max batch {} · {} serial",
                            t.tool_calls, t.max_concurrent_batch, t.serial_runs,
                        )
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                };
                if !sched.is_empty() {
                    ilines.push(Line::styled(sched, dim()));
                }
                if let Some(t) = self.last_telemetry.as_ref() {
                    let latency = &t.phase_latencies;
                    ilines.push(Line::styled(
                        format!(
                            "latency: model {}ms · tools {}ms · verify {}ms · review {}ms · finalize {}ms",
                            latency.model_request_ms,
                            latency.tool_batch_ms,
                            latency.verify_ms,
                            latency.review_ms,
                            latency.finalize_ms,
                        ),
                        dim(),
                    ));
                }
                ilines.push(Line::styled(
                    format!("tool calls this turn: {}", self.turn_tool_calls),
                    dim(),
                ));
                // Context composition: occupancy vs. window, plus the current
                // turn's raw prompt estimate and output across all model calls.
                let (input, output) = self.usage;
                let ctx = if let Some(pct) = self.context_pct() {
                    format!(
                        " · ctx {}{pct}%",
                        if self.usage_estimated { "~" } else { "" }
                    )
                } else {
                    String::new()
                };
                ilines.push(Line::styled(
                    format!(
                        "turn: user prompt estimate {} · output across all model calls {}{}{ctx}",
                        fmt_count(input),
                        if self.usage_estimated { "~" } else { "" },
                        fmt_count(output)
                    ),
                    dim(),
                ));
                if let Some(limits) = fmt_rate_limits(self.rate_limits) {
                    ilines.push(Line::styled(limits, dim()));
                }
            }
            // The `?` keybindings help overlay — rows come from `keys::KEY_BINDINGS`.
            if self.show_help {
                let th = crate::theme::theme();
                ilines.push(Line::styled(
                    "keybindings (? to close)".to_string(),
                    Style::default()
                        .fg(th.accent_system)
                        .add_modifier(Modifier::BOLD),
                ));
                let body_cap = help_h.saturating_sub(1);
                for (keys, help) in crate::keys::help_overlay_rows().into_iter().take(body_cap) {
                    if let Some(help) = help {
                        ilines.push(Line::from(vec![
                            Span::styled(
                                format!("  {keys:<22}"),
                                Style::default().fg(th.text_primary),
                            ),
                            Span::styled(help.to_string(), dim()),
                        ]));
                    } else {
                        ilines.push(Line::styled(
                            format!(" {keys}"),
                            Style::default()
                                .fg(th.accent_system)
                                .add_modifier(Modifier::BOLD),
                        ));
                    }
                }
            }
            if let Some(notice) = &self.startup_notice {
                ilines.push(Line::styled(
                    notice.clone(),
                    Style::default().fg(crate::theme::theme().warning),
                ));
            }
            if let Some(warning) = &self.checkpoint_warning {
                ilines.push(Line::styled(
                    warning.clone(),
                    Style::default()
                        .fg(crate::theme::theme().warning)
                        .add_modifier(Modifier::BOLD),
                ));
            }
            if self.quit_notice.is_some() {
                ilines.push(Line::styled(
                    "Press Ctrl-C again to exit",
                    Style::default().fg(crate::theme::theme().warning),
                ));
            }
            // A brief "copied N chars" confirmation after a drag-select copy, so
            // it's clear the selection reached the clipboard. Fades via the idle
            // redraw tick after a couple of seconds.
            const COPY_TOAST_MS: u128 = 2500;
            if let Some((n, at)) = self.copy_toast {
                if at.elapsed().as_millis() < COPY_TOAST_MS {
                    ilines.push(Line::styled(
                        format!("✓ copied {n} chars to the clipboard"),
                        Style::default().fg(crate::theme::theme().accent_success),
                    ));
                } else {
                    self.copy_toast = None;
                }
            }
            // Voice dictation. An open microphone has to be visible at a
            // glance: the transcript line announcing it scrolls away, and
            // leaving a mic recording unnoticed is exactly the failure worth
            // designing against.
            if let Some(line) = self.voice_indicator() {
                ilines.push(line);
            }
            // Ctrl-K command palette.
            if let Some(palette) = &self.palette {
                let th = crate::theme::theme();
                ilines.push(Line::from(vec![
                    Span::styled(
                        "palette ",
                        Style::default()
                            .fg(th.accent_system)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!("> {}_", palette.query),
                        Style::default().fg(th.text_primary),
                    ),
                ]));
                let sel = palette.selected;
                for (i, item) in palette.items.iter().take(12).enumerate() {
                    if i == sel {
                        ilines.push(Line::from(vec![
                            Span::styled(
                                format!("▶ {}", item.label),
                                Style::default()
                                    .fg(th.accent_system)
                                    .add_modifier(Modifier::BOLD),
                            ),
                            Span::styled(format!("  {}", item.help), dim()),
                        ]));
                    } else {
                        ilines.push(Line::from(vec![
                            Span::raw(format!("  {}", item.label)),
                            Span::styled(format!("  {}", item.help), dim()),
                        ]));
                    }
                }
                ilines.push(Line::styled(
                    "  ↑↓ move · Enter run · Esc close · type to filter",
                    dim(),
                ));
            }
            // The `/`-command completion menu sits just above the input line. Rows
            // are command names (`/compact`) or, past the name, argument values
            // (`hybrid`, `full`, `elide`).
            let items = self.completion_items();
            let selected = self.completion.as_ref().map(|c| c.selected).unwrap_or(0);
            let label_w = items.iter().map(|i| i.label.len()).max().unwrap_or(0);
            let prefix = match self.completion.as_ref().map(|c| &c.ctx) {
                Some(crate::completion::CompletionContext::Command(p)) => p.as_str(),
                Some(crate::completion::CompletionContext::Path { prefix }) => prefix.as_str(),
                Some(crate::completion::CompletionContext::Arg { prefix, .. }) => prefix.as_str(),
                None => "",
            };
            for (i, item) in items.iter().enumerate() {
                let label = format!("{:<width$}", item.label, width = label_w);
                let mark = if i == selected { "▶ " } else { "  " };
                let mut row = vec![Span::raw(mark.to_string())];
                row.extend(crate::completion::highlight_label(
                    &label,
                    prefix,
                    i == selected,
                ));
                if !item.help.is_empty() {
                    row.push(Span::styled(format!("  {}", item.help), dim()));
                }
                ilines.push(Line::from(row));
            }
            if self.mode.is_normal() {
                // Vim-style normal mode: show a mode banner instead of the
                // editable input. If a search is in progress, show `/query`.
                if let Some(q) = self.mode.normal_search() {
                    ilines.push(Line::from(vec![
                        Span::styled("-- SEARCH -- ", Style::default().fg(th.warning)),
                        Span::styled(
                            format!("/{q}"),
                            Style::default().fg(crate::theme::theme().text_primary),
                        ),
                        Span::styled("▏", Style::default().fg(crate::theme::theme().gray_dim)),
                    ]));
                } else {
                    ilines.push(Line::from(vec![
                        Span::styled(
                            "-- NORMAL --",
                            Style::default()
                                .fg(crate::theme::theme().warning)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            "  j/k scroll · [] prompts · {} errors · /search · y copy · i insert",
                            dim(),
                        ),
                    ]));
                }
            } else {
                ilines.extend(input_lines);
            }
            frame.render_widget(Paragraph::new(ilines).block(input_block), composer_area);

            // Cursor sits within the editable input — below the optional startup
            // notice, the status line, and the completion menu. Hidden in normal
            // mode (no editable input).
            if !self.mode.is_normal() && self.palette.is_none() && self.tutorial.is_none() {
                let above = changed_h
                    + debug_h
                    + help_h
                    + palette_h
                    + usize::from(self.startup_notice.is_some())
                    + usize::from(self.checkpoint_warning.is_some())
                    + usize::from(self.quit_notice.is_some())
                    + self.completion_items().len();
                let cx = composer_area.x + 1 + cursor_col;
                let cy = composer_area.y + 1 + above as u16 + cursor_row;
                frame.set_cursor_position((
                    cx.min(composer_area.right().saturating_sub(2)),
                    cy.min(composer_area.bottom().saturating_sub(2)),
                ));
            }
        }

        let hints: &[ShortcutHint] =
            if self.confirmation.is_some() || self.plan_approval_capturing() {
                &[
                    ShortcutHint {
                        key: "enter",
                        label: "confirm",
                    },
                    ShortcutHint {
                        key: "esc",
                        label: "cancel",
                    },
                ]
            } else if self.working {
                &[
                    ShortcutHint {
                        key: "ctrl+c",
                        label: "interrupt",
                    },
                    ShortcutHint {
                        key: "Shift+Tab",
                        label: "mode",
                    },
                    ShortcutHint {
                        key: "?",
                        label: "help",
                    },
                ]
            } else {
                &[
                    ShortcutHint {
                        key: "Shift+Tab",
                        label: "mode",
                    },
                    ShortcutHint {
                        key: "?",
                        label: "help",
                    },
                ]
            };
        chrome::render_shortcuts_bar(frame, shortcuts_area, hints, &th);
    }

    /// Rebuild the transcript flatten+wrap cache when its inputs change.
    /// Spinner-only redraws hit the fast path and reuse measured lines.
    pub(crate) fn ensure_view_cache(&mut self, inner_w: u16, selected_block: Option<usize>) {
        let (_, theme_revision) = crate::theme::snapshot();
        self.ensure_view_cache_with_revision(inner_w, selected_block, theme_revision);
    }

    fn ensure_view_cache_with_revision(
        &mut self,
        inner_w: u16,
        selected_block: Option<usize>,
        theme_revision: u64,
    ) {
        let pending_fp = crate::view_cache::pending_fingerprint(&self.pending);
        let trimmed = self.trimmed;
        let subagent_tick = self.live_subagent_tick();
        if self.view_cache.matches(
            self.transcript_gen,
            theme_revision,
            inner_w,
            self.show_reasoning,
            self.show_tool_output,
            self.density,
            selected_block,
            pending_fp,
            trimmed,
            subagent_tick,
        ) {
            return;
        }

        // Incremental path: same structural inputs except generation advanced by
        // appending entries and/or pending text — reuse measured prefix and only
        // flatten the tail. Falls back to full rebuild when flags/width change or
        // the cache is empty.
        if self.try_incremental_view_cache(
            inner_w,
            selected_block,
            pending_fp,
            trimmed,
            theme_revision,
        ) {
            return;
        }

        let th = crate::theme::theme();
        let mut lines: Vec<Line<'static>> = Vec::new();
        let mut prompt_line_starts: Vec<usize> = Vec::new();
        let mut tool_ord = 0usize;
        let mut block_line_ranges: Vec<(usize, usize, usize)> = Vec::new();

        if trimmed > 0 {
            lines.push(Line::styled(
                format!("↑ {} lines compacted (see session log)", self.trimmed),
                Style::default()
                    .fg(th.gray_dim)
                    .add_modifier(Modifier::ITALIC),
            ));
        }

        for entry in &self.transcript {
            if matches!(entry, crate::TranscriptEntry::UserPrompt { .. }) {
                if !lines.is_empty()
                    && !lines
                        .last()
                        .is_some_and(|l| crate::render::line_text(l).trim().is_empty())
                {
                    lines.push(Line::raw(""));
                }
                prompt_line_starts.push(lines.len());
            }
            let ord = if entry.is_foldable() {
                let o = tool_ord;
                tool_ord += 1;
                if selected_block == Some(o) {
                    lines.push(Line::styled(
                        "▶ block selected · Enter fold/unfold · Ctrl-F expand · ↑↓/jk move · Esc exit",
                        Style::default()
                            .fg(th.accent_running)
                            .add_modifier(Modifier::BOLD),
                    ));
                }
                Some(o)
            } else {
                None
            };
            let start = lines.len();
            lines.extend(entry.flatten(self.show_reasoning, self.show_tool_output, self.density));
            if let Some(o) = ord {
                block_line_ranges.push((start, lines.len(), o));
            }
        }
        if let Some((style, markdown, text)) = &self.pending {
            let mut line = if *markdown {
                markdown_line(text, &mut self.code_lang.clone())
            } else {
                Line::styled(text.clone(), *style)
            };
            line.spans
                .push(Span::styled("▍", Style::default().fg(th.gray_dim)));
            lines.push(line);
        }

        let mut prefix: Vec<u32> = Vec::with_capacity(lines.len() + 1);
        let mut cum = 0u32;
        prefix.push(0);
        for line in &lines {
            cum = cum.saturating_add(wrapped_line_height(line, inner_w) as u32);
            prefix.push(cum);
        }

        let committed_entries = self.transcript.len();
        // When pending is present it's the last line; committed flat count excludes it.
        let committed_flat_lines = if self.pending.is_some() {
            lines.len().saturating_sub(1)
        } else {
            lines.len()
        };
        self.view_cache = crate::view_cache::TranscriptViewCache {
            generation: self.transcript_gen,
            theme_revision,
            width: inner_w,
            show_reasoning: self.show_reasoning,
            show_tool_output: self.show_tool_output,
            density: self.density,
            nav_selected: selected_block,
            pending_fp,
            trimmed,
            subagent_tick,
            lines,
            prefix,
            prompt_line_starts,
            block_line_ranges,
            committed_entries,
            committed_flat_lines,
        };
    }

    /// Fast path when only new transcript entries / pending text arrived.
    fn try_incremental_view_cache(
        &mut self,
        inner_w: u16,
        selected_block: Option<usize>,
        pending_fp: u64,
        trimmed: u64,
        theme_revision: u64,
    ) -> bool {
        let c = &self.view_cache;
        if c.prefix.is_empty()
            || c.width != inner_w
            || c.theme_revision != theme_revision
            || c.show_reasoning != self.show_reasoning
            || c.show_tool_output != self.show_tool_output
            || c.density != self.density
            || c.nav_selected != selected_block
            || c.trimmed != trimmed
            || c.subagent_tick != self.live_subagent_tick()
            || selected_block.is_some()
        {
            return false;
        }
        // This path is valid only for append-only transcript growth or a
        // pending streaming-line change. In-place progress updates and other
        // same-length mutations must rebuild the affected existing lines.
        let pending_changed = c.pending_fp != pending_fp;
        let appended_entries = self.transcript.len() > c.committed_entries;
        if !pending_changed && !appended_entries {
            return false;
        }
        // Only append when entries grew (or stayed) and we still have the old flat prefix.
        if self.transcript.len() < c.committed_entries {
            return false;
        }
        let th = crate::theme::theme();
        let mut lines = c.lines.clone();
        let mut prefix = c.prefix.clone();
        let mut prompt_line_starts = c.prompt_line_starts.clone();
        let mut block_line_ranges = c.block_line_ranges.clone();

        // Drop old pending line from the end of the cached flatten.
        if c.pending_fp != 0 && !lines.is_empty() {
            lines.pop();
            prefix.pop();
        }
        // Ensure lines/prefix align to committed_flat_lines.
        if lines.len() > c.committed_flat_lines {
            lines.truncate(c.committed_flat_lines);
            prefix.truncate(c.committed_flat_lines + 1);
        }

        // Append newly committed entries.
        let start_entry = c.committed_entries;
        let mut tool_ord = block_line_ranges.last().map(|(_, _, o)| o + 1).unwrap_or(0);
        for entry in &self.transcript[start_entry..] {
            if matches!(entry, crate::TranscriptEntry::UserPrompt { .. }) {
                if !lines.is_empty()
                    && !lines
                        .last()
                        .is_some_and(|l| crate::render::line_text(l).trim().is_empty())
                {
                    let h = wrapped_line_height(&Line::raw(""), inner_w) as u32;
                    let cum = prefix.last().copied().unwrap_or(0).saturating_add(h);
                    prefix.push(cum);
                    lines.push(Line::raw(""));
                }
                prompt_line_starts.push(lines.len());
            }
            let ord = if entry.is_foldable() {
                let o = tool_ord;
                tool_ord += 1;
                Some(o)
            } else {
                None
            };
            let start = lines.len();
            let flat = entry.flatten(self.show_reasoning, self.show_tool_output, self.density);
            for line in &flat {
                let h = wrapped_line_height(line, inner_w) as u32;
                let cum = prefix.last().copied().unwrap_or(0).saturating_add(h);
                prefix.push(cum);
            }
            lines.extend(flat);
            if let Some(o) = ord {
                block_line_ranges.push((start, lines.len(), o));
            }
        }
        let committed_entries = self.transcript.len();
        let committed_flat_lines = lines.len();

        // Re-add pending.
        if let Some((style, markdown, text)) = &self.pending {
            let mut line = if *markdown {
                markdown_line(text, &mut self.code_lang.clone())
            } else {
                Line::styled(text.clone(), *style)
            };
            line.spans
                .push(Span::styled("▍", Style::default().fg(th.gray_dim)));
            let h = wrapped_line_height(&line, inner_w) as u32;
            let cum = prefix.last().copied().unwrap_or(0).saturating_add(h);
            prefix.push(cum);
            lines.push(line);
        }

        self.view_cache = crate::view_cache::TranscriptViewCache {
            generation: self.transcript_gen,
            theme_revision,
            width: inner_w,
            show_reasoning: self.show_reasoning,
            show_tool_output: self.show_tool_output,
            density: self.density,
            nav_selected: selected_block,
            pending_fp,
            trimmed,
            subagent_tick: self.live_subagent_tick(),
            lines,
            prefix,
            prompt_line_starts,
            block_line_ranges,
            committed_entries,
            committed_flat_lines,
        };
        let _ = th; // silence when paints unused
        true
    }
}

/// Recording-dot colour for a redraw tick: a triangle wave over 20 ticks,
/// breathing between the muted grey and the error accent.
fn recording_dot_color(tick: usize) -> ratatui::style::Color {
    let th = crate::theme::theme();
    let phase = (tick % 20) as f32 / 10.0 - 1.0;
    lerp_color(th.gray_dim, th.accent_error, phase.abs())
}
