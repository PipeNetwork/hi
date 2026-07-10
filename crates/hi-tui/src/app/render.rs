//! `App` methods: render.

use hi_agent::{Agent, PlanStatus};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Paragraph, Wrap};

use crate::model_picker::{display_capabilities, display_price, display_window};
use crate::render::{diff_lines, dim, markdown_line, wrapped_height};
use crate::util::{clip_reason, fmt_count, fmt_elapsed, fmt_rate_limits};
use crate::{PICKER_ROWS, SPINNER, TurnEventKind, TurnState};

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
    if !t.review_repair_exhaustion_reason.is_empty() {
        parts.push(format!(
            "exhausted {}",
            hi_agent::compact_review_repair_label(&t.review_repair_exhaustion_reason)
        ));
    }
    Some(format!("review repair: {}", parts.join(" · ")))
}

impl crate::App {
    /// The live "what's happening now" lead for the working line: the in-flight
    /// tool named with its own elapsed timer, otherwise the model phase —
    /// `thinking…` (reasoning), `responding…` (streaming text), or `Working`
    /// (the round's model call is in flight but nothing's streamed yet). The
    /// `Working` lead is rendered with a rolling gray→white→gray wave animation
    /// (see [`Self::working_spans`]); the others are plain cyan bold. Lets you
    /// tell a slow tool from a slow model at a glance.
    pub(crate) fn activity_line(&self) -> String {
        // A compact progress suffix for multi-step turns: "round 3 · 5 calls".
        // Suppressed on the first round with no tool calls (the common single-shot case).
        let progress = if self.turn_rounds > 1 || (self.turn_rounds > 0 && self.turn_tool_calls > 0)
        {
            format!(
                " · round {} · {} call{}",
                self.turn_rounds,
                self.turn_tool_calls,
                if self.turn_tool_calls == 1 { "" } else { "s" }
            )
        } else {
            String::new()
        };
        if let (Some(tool), Some(started)) = (&self.current_tool, self.current_tool_started) {
            return format!(
                "running {tool} · {}{progress}",
                fmt_elapsed(started.elapsed().as_secs())
            );
        }
        let secs = self.started.map(|t| t.elapsed().as_secs()).unwrap_or(0);
        let verb = match self.last_turn_event {
            Some(TurnEventKind::Reasoning) => "thinking",
            Some(TurnEventKind::Assistant) => "responding",
            _ => "Working",
        };
        format!("{verb}… {}{progress}", fmt_elapsed(secs))
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
        let gray = Style::default().fg(Color::DarkGray);
        let lit_style = Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD);
        chars
            .iter()
            .enumerate()
            .map(|(i, &c)| Span::styled(c.to_string(), if i == lit { lit_style } else { gray }))
            .collect()
    }

    pub(crate) fn report_status(&mut self, agent: &Agent) {
        let (input, output) = self.usage;
        let state = match &self.last_turn_state {
            TurnState::Idle => "idle".to_string(),
            TurnState::Running => "running".to_string(),
            TurnState::Done(s) => format!("done ({s})"),
            TurnState::Warning(s) => format!("warning ({s})"),
            TurnState::Failed(s) => format!("failed ({s})"),
            TurnState::Cancelled => "cancelled".to_string(),
        };
        let ctx = self
            .context_pct()
            .map(|p| format!("{p}%"))
            .unwrap_or_else(|| "unknown".to_string());
        let goal = agent.goal_summary();
        let verify = agent.verify_summary();
        let tel = agent.last_turn_telemetry();
        let error = self.last_error.as_deref().unwrap_or("none");
        for line in [
            format!("status: {state}"),
            format!("provider/model: {} · {}", self.provider, self.model),
            format!("context: {ctx}; turn usage: {input} prompt · {output} generated"),
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
    /// inputs show only their last [`MAX_INPUT_ROWS`] lines with a "… more above"
    /// note so they can't swallow the screen.
    ///
    /// `width` is the inner width of the input box (borders already subtracted).
    /// Each logical line is soft-wrapped to that width so a long single-line
    /// prompt stays visible and the cursor tracks the wrap instead of running off
    /// the right edge.
    pub(crate) fn input_view(&self, width: u16) -> (Vec<Line<'static>>, u16, u16) {
        const MAX_INPUT_ROWS: usize = 10;
        const PREFIX: usize = 2; // "› " or "  "
        let text = self.input.text();
        let before: String = text.chars().take(self.input.cursor()).collect();
        let cursor_col_logical = before.chars().rev().take_while(|&c| c != '\n').count();

        // Inner text width per line (prefix occupies the first 2 columns).
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
            while start < chars.len() {
                let end = (start + wrap_w).min(chars.len());
                let chunk: String = chars[start..end].iter().collect();
                // The cursor is in this display line if its logical column falls
                // within [start, end]. A cursor exactly at `end` (end of a wrapped
                // chunk) stays on this line's last column rather than jumping to
                // the next line's column 0 — matches how terminals render it.
                let cursor_here = cursor_in_this.and_then(|c| {
                    if c >= start && c <= end {
                        Some(c - start)
                    } else {
                        None
                    }
                });
                wrapped.push((chunk, cursor_here));
                start = end;
            }
        }

        let truncated = wrapped.len() > MAX_INPUT_ROWS;
        let start = if truncated {
            wrapped.len() - MAX_INPUT_ROWS
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
        for (i, (chunk, cursor_here)) in wrapped[start..].iter().enumerate() {
            let prefix = if i == 0 && !truncated { "› " } else { "  " };
            lines.push(Line::from(format!("{prefix}{chunk}")));
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
            return self.goal_lines(goal, max_steps);
        }
        if self.plan.is_empty() {
            return Vec::new();
        }
        const HARD_CAP: usize = 8;
        let max_steps = max_steps.min(HARD_CAP);
        let total = self.plan.len();
        let done = self
            .plan
            .iter()
            .filter(|s| s.status == PlanStatus::Done)
            .count();
        let mut out = vec![Line::styled(
            format!("plan · {done}/{total}"),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )];
        for s in self.plan.iter().take(max_steps) {
            let (glyph, glyph_style, title_style) = match s.status {
                PlanStatus::Done => ('✓', Style::default().fg(Color::Green), dim()),
                PlanStatus::Active => (
                    '▸',
                    Style::default()
                        .fg(Color::Cyan)
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
        if total > max_steps {
            out.push(Line::styled(
                format!("  … +{} more", total - max_steps),
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
        let state = if goal.paused { " · paused" } else { "" };
        let mut header = format!("goal · {done}/{total}{state}");
        if !goal.objective.is_empty() {
            header.push_str(" · ");
            header.push_str(&goal.objective);
        }
        let mut out = vec![Line::styled(
            header,
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        )];
        for s in goal.sub_goals.iter().take(max_steps) {
            let (glyph, glyph_style, title_style) = match s.status {
                hi_agent::GoalStatus::Done => ('✓', Style::default().fg(Color::Green), dim()),
                hi_agent::GoalStatus::Active => (
                    '▸',
                    Style::default()
                        .fg(Color::Magenta)
                        .add_modifier(Modifier::BOLD),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                hi_agent::GoalStatus::Failed => ('✗', Style::default().fg(Color::Red), dim()),
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

    pub(crate) fn render(&mut self, frame: &mut ratatui::Frame) {
        let area = frame.area();
        // The input box grows to fit a spinner status line (while working), the
        // (possibly multi-line) input, and up to three queued commands.
        let status_lines = 1usize;
        let queued_shown = self.queue.len().min(3);
        let queue_extra = usize::from(self.queue.len() > 3);
        let (input_lines, cursor_row, cursor_col) = self.input_view(area.width.saturating_sub(2));
        let completion_rows = self.completion_items().len();
        // The optional Ctrl-D diff panel height (header + up to 20 diff lines +
        // optional "more" line) and the compact changed-files summary line.
        let diff_h = if self.show_diff && self.diff_text.is_some() {
            let n = self
                .diff_text
                .as_deref()
                .map(|t| t.trim().lines().count())
                .unwrap_or(0);
            1 + n.min(20) + usize::from(n > 20)
        } else {
            0
        };
        let changed_h = usize::from(!self.last_changed_files.is_empty() && !self.working);
        // The Ctrl-? observability panel: header plus present diagnostic lines.
        let debug_h = if self.show_debug {
            let telemetry_h = if let Some(t) = self.last_telemetry.as_ref() {
                1 + usize::from(t.tool_calls > 0) + usize::from(review_repair_summary(t).is_some())
            } else {
                0
            };
            4 + telemetry_h + usize::from(fmt_rate_limits(self.rate_limits).is_some())
        } else {
            0
        };
        // The `?` keybindings help overlay: header + 10 lines.
        let help_h = if self.show_help { 12 } else { 0 };
        // Live streamed tool output tail (e.g. bash stdout), shown while a tool runs.
        let stream_h = if self.working && !self.tool_stream_tail.is_empty() {
            self.tool_stream_tail.len()
        } else {
            0
        };
        // Height of the input box excluding the plan checklist and the 2 border
        // rows. Used to figure out how many plan steps fit on screen.
        let base_h = diff_h
            + changed_h
            + debug_h
            + help_h
            + stream_h
            + usize::from(self.startup_notice.is_some())
            + usize::from(self.quit_notice.is_some())
            + status_lines
            + completion_rows
            + input_lines.len()
            + queued_shown
            + queue_extra;
        // The live plan checklist, pinned just above the input (input-bar state
        // only). The step count is capped to what fits on screen so a long plan
        // can't make the box taller than the terminal — ratatui's Layout would
        // otherwise clamp the rect and the Paragraph content would spill past
        // the bottom border. Reserve one row for the transcript (Min(1) below).
        let cap = area.height.saturating_sub(1).max(1) as usize;
        let avail_inner = cap.saturating_sub(base_h + 2);
        // plan_h = 1 (header) + steps_shown + (1 if total > steps_shown else 0).
        // Pick the largest step count (up to total and HARD_CAP) whose plan_h
        // fits avail_inner.
        let max_steps = if self.plan.is_empty() {
            0
        } else {
            const HARD_CAP: usize = 8;
            let total = self.plan.len();
            let upper = total.min(HARD_CAP);
            // plan_h for a candidate `n` (n <= upper): 1 + n + (total > n) as usize.
            // Try showing all `upper` first; if it doesn't fit, shrink.
            let mut n = upper;
            while n > 0 && 1 + n + usize::from(total > n) > avail_inner {
                n -= 1;
            }
            // If even n=0 (header only, +maybe more) doesn't fit, show 1 step
            // so the plan is still visible rather than entirely hidden.
            if 1 + n + usize::from(total > n) > avail_inner {
                1
            } else {
                n
            }
        };
        let plan_block = self.plan_lines(max_steps);
        let plan_h = plan_block.len();
        let input_h = if self.fetching.is_some() {
            3
        } else if let Some(p) = &self.picker {
            // filter line + visible model rows + borders, bounded by the screen.
            let rows = p.matches.len().clamp(1, PICKER_ROWS) as u16;
            (rows + 3).min(area.height.saturating_sub(3))
        } else if let Some(form) = &self.provider_form {
            // Provider form: provider picker row + hint row + text fields +
            // borders. The API-key field is hidden for Ollama, so subtract one.
            let fields = if form.api_key_unneeded() { 3 } else { 4 };
            (fields + 4) as u16
        } else {
            (base_h + plan_h + 2).min(cap) as u16
        };
        let rows = Layout::vertical([Constraint::Min(1), Constraint::Length(input_h)]).split(area);

        // --- Transcript ---
        let title = format!(" hi · {} · {} ", self.provider, self.model);
        // Right-aligned: persistent token totals, a context-fill gauge,
        // then the last status — so usage is always visible without a command.
        let mut info_parts: Vec<String> = Vec::new();
        let (input, output) = self.usage;
        if input + output > 0 {
            info_parts.push(format!(
                "prompt↑{} gen↓{}",
                fmt_count(input),
                fmt_count(output)
            ));
        }
        if let Some(goal) = &self.goal {
            let total = goal.sub_goals.len();
            if total > 0 {
                let done = goal
                    .sub_goals
                    .iter()
                    .filter(|s| s.status == hi_agent::GoalStatus::Done)
                    .count();
                if goal.paused {
                    info_parts.push(format!("goal {done}/{total} ⏸"));
                } else {
                    info_parts.push(format!("goal {done}/{total}"));
                }
            }
        }
        if let Some(pct) = self.context_pct() {
            info_parts.push(format!("{pct}% ctx"));
        }
        if !self.status.is_empty() {
            info_parts.push(self.status.clone());
        }
        let info = if info_parts.is_empty() {
            String::new()
        } else {
            format!(" {} ", info_parts.join(" · "))
        };
        let mut lines: Vec<Line<'static>> = Vec::new();
        // If older transcript lines have been trimmed, show a marker at the top
        // so the user knows earlier content scrolled off (it's still in the
        // JSONL session log).
        if self.trimmed > 0 {
            lines.push(Line::styled(
                format!("↑ {} lines compacted (see session log)", self.trimmed),
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC),
            ));
        }
        lines.extend(
            self.transcript
                .iter()
                .flat_map(|e| e.flatten(self.show_reasoning)),
        );
        if let Some((style, markdown, text)) = &self.pending {
            // Style the in-progress line live (headings, bold, code, …) so prose
            // doesn't snap into formatting only when its newline lands. The line
            // isn't committed yet, so apply markdown against a CLONE of the fence
            // state — the real `code_lang` must only advance on a committed line.
            let line = if *markdown {
                markdown_line(text, &mut self.code_lang.clone())
            } else {
                Line::styled(text.clone(), *style)
            };
            lines.push(line);
        }
        let inner_w = rows[0].width.saturating_sub(2);
        let inner_h = rows[0].height.saturating_sub(2);
        let total = wrapped_height(&lines, inner_w);
        let max_scroll = total.saturating_sub(inner_h);
        // Cache the geometry so scroll events (which fire outside render) can clamp
        // and detect the bottom.
        self.view_max_scroll = max_scroll;
        self.view_total = total;
        // Pinned to the bottom while following; otherwise hold the user's absolute
        // offset, re-pinning if the content shrank back to within one screen.
        let scroll = if self.following || self.scroll >= max_scroll {
            self.following = true;
            max_scroll
        } else {
            self.scroll
        };

        let mut block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(dim())
            .title(title)
            .title_top(Line::from(info).right_aligned());
        // While scrolled up, a bottom-right hint shows how much is below — new
        // lines that arrived since you left the bottom, else how far down it is.
        if !self.following {
            let new = total.saturating_sub(self.total_when_unpinned);
            let label = if new > 0 {
                format!(" ↓ {new} new ")
            } else {
                format!(" ↓ {} below ", max_scroll.saturating_sub(scroll))
            };
            block = block.title_bottom(
                Line::from(Span::styled(
                    label,
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ))
                .right_aligned(),
            );
        }
        let para = Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .block(block)
            .scroll((scroll, 0));
        frame.render_widget(para, rows[0]);

        // --- Bottom region: a fetch/plan spinner, the model picker, or the input bar. ---
        if let Some(started) = self.fetching.or(self.planning) {
            let frame_ch = SPINNER[self.spinner % SPINNER.len()];
            let elapsed = fmt_elapsed(started.elapsed().as_secs());
            let label = if self.planning.is_some() {
                "planning goal with the planner model…".to_string()
            } else {
                format!("fetching models from {}…", self.provider)
            };
            let block = Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Cyan));
            let body = Line::from(vec![
                Span::styled(
                    format!("{frame_ch} {label} {elapsed}"),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("   Esc to cancel", dim()),
            ]);
            frame.render_widget(Paragraph::new(body).block(block), rows[1]);
        } else if let Some(p) = &self.picker {
            let block = Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Cyan))
                .title(" select a model ")
                .title_top(
                    Line::from(format!(" {}/{} ", p.selected + 1, p.matches.len().max(1)))
                        .right_aligned(),
                );
            let mut plines: Vec<Line> = vec![Line::from(vec![
                Span::raw(format!("filter: {}", p.filter)),
                Span::styled(
                    "   ↑↓ move · type to filter · Enter select · Esc cancel",
                    dim(),
                ),
            ])];
            let (_, visible) = p.visible();
            if visible.is_empty() {
                plines.push(Line::styled("  (no matches)".to_string(), dim()));
            }
            for row in visible {
                let mut tag = String::new();
                if row.id == p.current {
                    tag.push_str(" (current)");
                }
                let caps = display_capabilities(row.meta);
                if !caps.is_empty() {
                    tag.push_str(&format!(" {{{caps}}}"));
                }
                // Price + window columns, right-aligned after the id.
                let price = display_price(row.meta);
                let window = display_window(row.meta);
                let meta_col = if price.is_empty() && window.is_empty() {
                    String::new()
                } else {
                    format!("  {price:>8}  {window:>5}")
                };
                if row.selected {
                    plines.push(Line::from(vec![
                        Span::styled(
                            format!("▶ {}", row.id),
                            Style::default()
                                .fg(Color::Cyan)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(meta_col, Style::default().fg(Color::Yellow)),
                        Span::styled(tag, dim()),
                    ]));
                } else {
                    plines.push(Line::from(vec![
                        Span::raw(format!("  {}", row.id)),
                        Span::styled(meta_col, Style::default().fg(Color::DarkGray)),
                        Span::styled(tag, dim()),
                    ]));
                }
            }
            frame.render_widget(Paragraph::new(plines).block(block), rows[1]);
            // Cursor on the filter line, just after "filter: <text>".
            let cx = rows[1].x + 1 + 8 + p.filter.chars().count() as u16;
            frame.set_cursor_position((cx.min(rows[1].right().saturating_sub(2)), rows[1].y + 1));
        } else if let Some(form) = &self.provider_form {
            let title = if form.editing {
                " edit provider "
            } else {
                " add provider "
            };
            let block = Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Cyan))
                .title(title);
            let choices = form.provider_choices();
            let pidx = form.provider_idx();
            let mut lines: Vec<Line> = Vec::new();

            // Provider picker row.
            let mut prov_spans = vec![Span::raw("Provider: ")];
            for (i, (_id, label)) in choices.iter().enumerate() {
                if i == pidx {
                    prov_spans.push(Span::styled(
                        format!("▶ {label} "),
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ));
                } else {
                    prov_spans.push(Span::styled(format!("  {label} "), dim()));
                }
            }
            lines.push(Line::from(prov_spans));
            lines.push(Line::styled(
                "  ←→ cycle · Tab next field · Enter save · Esc cancel".to_string(),
                dim(),
            ));

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
                    Span::styled(display, Style::default().fg(Color::DarkGray))
                } else if is_active {
                    Span::styled(display, Style::default().fg(Color::Cyan))
                } else {
                    Span::raw(display)
                };
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("{prefix}{label}: "),
                        Style::default().fg(Color::Yellow),
                    ),
                    val_span,
                ]));
            }

            frame.render_widget(Paragraph::new(lines).block(block), rows[1]);

            // Cursor on the active text field.
            let form_fields = form.field_labels();
            let active_idx = form.active();
            // Account for the hidden API-key field (index 1) when computing the
            // display row: fields after it shift up by one.
            let hidden_before = if form.api_key_unneeded() && active_idx > 1 {
                1
            } else {
                0
            };
            // +3 for border + provider row + hint row.
            let cy = rows[1].y + 1 + 2 + (active_idx - hidden_before) as u16;
            let label = form_fields[active_idx].0;
            let prefix_len = 2 + label.len() + 2; // "▶ " + label + ": "
            let cx = rows[1].x + 1 + prefix_len as u16 + form.active_cursor() as u16;
            frame.set_cursor_position((cx.min(rows[1].right().saturating_sub(2)), cy));
        } else {
            // The border turns cyan and the top inner line becomes a bold
            // spinner + elapsed seconds while a turn runs; the prompt stays
            // editable so you can type the next command (it queues below).
            let input_block = Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(if self.working {
                    Style::default().fg(Color::Cyan)
                } else {
                    dim()
                });

            let mut ilines: Vec<Line> = Vec::new();
            // Pinned plan checklist at the very top of the input box.
            ilines.extend(plan_block);
            // Ctrl-R reverse history search overlay: shows the query, the match
            // count, and a few recent matches above the input line.
            if let Some(search) = &self.history_search {
                let count = search.matches.len();
                let preview = search
                    .current()
                    .and_then(|i| self.input.history.get(i))
                    .map(|s| s.replace('\n', " "))
                    .unwrap_or_default();
                // Char-based truncation: history entries are arbitrary input,
                // and a byte slice panics on a multi-byte char at the cut.
                let preview = if preview.chars().count() > 60 {
                    format!("{}…", preview.chars().take(60).collect::<String>())
                } else {
                    preview
                };
                ilines.push(Line::from(vec![
                    Span::styled("reverse-i-search: ", Style::default().fg(Color::Green)),
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
            // The `Ctrl-D` working-tree diff panel: a compact view of what's
            // changed in the tree, rendered with the same highlighting as
            // tool-output diffs. Sits above the changed-files summary line.
            if self.show_diff
                && let Some(text) = &self.diff_text
            {
                ilines.push(Line::styled(
                    "diff (Ctrl-D to close)".to_string(),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ));
                let trimmed = text.trim();
                if trimmed.is_empty() {
                    ilines.push(Line::styled("(no changes in the working tree)", dim()));
                } else {
                    // `diff_lines` parses the whole body (tracking `@@` line
                    // numbers) into highlighted lines; cap the result so a huge
                    // diff can't swallow the input box. The full diff is one
                    // `git diff` away.
                    let rendered = diff_lines(trimmed);
                    let total = rendered.len();
                    for line in rendered.into_iter().take(20) {
                        ilines.push(line);
                    }
                    if total > 20 {
                        ilines.push(Line::styled(
                            format!("  … +{} more (see `git diff`)", total - 20),
                            dim(),
                        ));
                    }
                }
            }
            // A compact "changed: …" line so the user always sees what the last
            // turn touched, without opening the diff panel or scrolling.
            if !self.last_changed_files.is_empty() && !self.working {
                let summary = self
                    .last_changed_files
                    .iter()
                    .map(|p| p.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                ilines.push(Line::styled(
                    format!("changed: {summary}  (Ctrl-D for diff)"),
                    dim(),
                ));
            }
            // The Ctrl-? agent-observability panel: trajectory telemetry, tool
            // calls this turn, and context composition. Read-only diagnostics.
            if self.show_debug {
                ilines.push(Line::styled(
                    "agent (Ctrl-? to close)".to_string(),
                    Style::default()
                        .fg(Color::Cyan)
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
                        ilines.push(Line::styled(repair, dim()));
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
                ilines.push(Line::styled(
                    format!("tool calls this turn: {}", self.turn_tool_calls),
                    dim(),
                ));
                // Context composition: occupancy vs. window, plus the current
                // turn's raw prompt estimate and generated output.
                let (input, output) = self.usage;
                let ctx = if let Some(pct) = self.context_pct() {
                    format!(" · ctx {pct}%")
                } else {
                    String::new()
                };
                ilines.push(Line::styled(
                    format!(
                        "turn: prompt↑{} gen↓{}{ctx}",
                        fmt_count(input),
                        fmt_count(output)
                    ),
                    dim(),
                ));
                if let Some(limits) = fmt_rate_limits(self.rate_limits) {
                    ilines.push(Line::styled(limits, dim()));
                }
            }
            // The `?` keybindings help overlay: a compact, contextual cheat
            // sheet. Toggled by pressing `?` on an empty input line.
            if self.show_help {
                ilines.push(Line::styled(
                    "keybindings (? to close)".to_string(),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ));
                let bindings = [
                    ("Enter", "send the prompt"),
                    ("Alt-Enter / \\", "insert a newline (multi-line prompt)"),
                    (
                        "Ctrl-C",
                        "interrupt the running turn; double-press idle to quit",
                    ),
                    ("Ctrl-D", "toggle the working-tree diff panel"),
                    ("Ctrl-T", "toggle reasoning (thinking) display"),
                    ("Ctrl-?", "toggle agent observability panel"),
                    ("Ctrl-R", "fuzzy-search input history"),
                    ("PageUp/PageDown", "scroll the transcript"),
                    ("Esc", "clear input or dismiss panels"),
                    ("/quit", "quit"),
                    ("/help", "show all slash commands"),
                ];
                for (key, desc) in bindings {
                    ilines.push(Line::from(vec![
                        Span::styled(format!("  {key:<18}"), dim()),
                        Span::raw(desc),
                    ]));
                }
            }
            if let Some(notice) = &self.startup_notice {
                ilines.push(Line::styled(
                    notice.clone(),
                    Style::default().fg(Color::Yellow),
                ));
            }
            if self.quit_notice.is_some() {
                ilines.push(Line::styled(
                    "Press Ctrl-C again to exit",
                    Style::default().fg(Color::Yellow),
                ));
            }
            if self.working {
                let frame_ch = SPINNER[self.spinner % SPINNER.len()];
                let (input, output) = self.usage;
                // Show the running token total once the first round reports it.
                let mut stats = String::new();
                if input + output > 0 {
                    stats.push_str(&format!(
                        " · prompt↑{} gen↓{}",
                        fmt_count(input),
                        fmt_count(output)
                    ));
                }
                if let Some(pct) = self.context_pct() {
                    stats.push_str(&format!(" · {pct}% ctx"));
                }
                if let Some(limits) = fmt_rate_limits(self.rate_limits) {
                    stats.push_str(&format!(" · {limits}"));
                }
                // The activity lead (named tool + timer, or thinking/responding)
                // replaces the old coarse "working… · last: <event>"; its own timer
                // and the watchdog notices cover the "is it stalled?" signal.
                // The "Working" model phase renders as a rolling gray→white wave
                // (working_spans); other leads stay cyan bold.
                let activity = self.activity_line();
                let is_working_wave = self.current_tool.is_none()
                    && !matches!(
                        self.last_turn_event,
                        Some(TurnEventKind::Reasoning) | Some(TurnEventKind::Assistant)
                    );
                let mut lead: Vec<Span<'static>> =
                    Vec::with_capacity(if is_working_wave { 8 } else { 1 });
                lead.push(Span::styled(
                    format!("{frame_ch} "),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ));
                if is_working_wave {
                    lead.extend(self.working_spans());
                    // activity_line() == "Working… <secs>[ · round N · M calls]";
                    // append everything after "Working" (the "…", timer, progress).
                    if let Some(rest) = activity.strip_prefix("Working") {
                        lead.push(Span::styled(
                            rest.to_string(),
                            Style::default()
                                .fg(Color::Cyan)
                                .add_modifier(Modifier::BOLD),
                        ));
                    }
                } else {
                    lead.push(Span::styled(
                        activity,
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ));
                }
                lead.push(Span::styled(stats.to_string(), Style::default()));
                lead.push(Span::styled("   Ctrl-C to interrupt", dim()));
                ilines.push(Line::from(lead));
                // Show a tail of recent streamed tool output (e.g. bash stdout)
                // so the user sees live progress during long-running commands.
                for line in &self.tool_stream_tail {
                    ilines.push(Line::styled(
                        format!("  │ {}", clip_reason(line)),
                        Style::default().fg(Color::DarkGray),
                    ));
                }
            } else {
                let line = match &self.last_turn_state {
                    TurnState::Idle => "ready".to_string(),
                    TurnState::Running => "working".to_string(),
                    TurnState::Done(s) => format!("ready · last: done ({s})"),
                    TurnState::Warning(s) => format!("ready · last: warning ({s})"),
                    // Show the failure reason inline so you don't have to scroll
                    // the transcript to learn what went wrong.
                    TurnState::Failed(s) => {
                        format!(
                            "ready · last: failed — {} · /retry to rerun",
                            clip_reason(s)
                        )
                    }
                    TurnState::Cancelled => "ready · last: cancelled".to_string(),
                };
                ilines.push(Line::styled(line, dim()));
            }
            // The `/`-command completion menu sits just above the input line. Rows
            // are command names (`/compact`) or, past the name, argument values
            // (`hybrid`, `full`, `elide`).
            let items = self.completion_items();
            let selected = self.completion.as_ref().map(|c| c.selected).unwrap_or(0);
            let label_w = items.iter().map(|i| i.label.len()).max().unwrap_or(0);
            for (i, item) in items.iter().enumerate() {
                let label = format!("{:<width$}", item.label, width = label_w);
                if i == selected {
                    ilines.push(Line::from(vec![
                        Span::styled(
                            format!("▶ {label}"),
                            Style::default()
                                .fg(Color::Cyan)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(format!("  {}", item.help), dim()),
                    ]));
                } else {
                    ilines.push(Line::from(vec![
                        Span::raw(format!("  {label}")),
                        Span::styled(format!("  {}", item.help), dim()),
                    ]));
                }
            }
            ilines.extend(input_lines);
            for q in self.queue.iter().take(3) {
                ilines.push(Line::styled(format!("⏳ {q}"), dim()));
            }
            if self.queue.len() > 3 {
                ilines.push(Line::styled(
                    format!("   … +{} more queued", self.queue.len() - 3),
                    dim(),
                ));
            }
            frame.render_widget(Paragraph::new(ilines).block(input_block), rows[1]);

            // Cursor sits within the editable input — below the optional startup
            // notice, the status line, and the completion menu.
            let above = plan_h
                + diff_h
                + changed_h
                + debug_h
                + help_h
                + stream_h
                + usize::from(self.startup_notice.is_some())
                + usize::from(self.quit_notice.is_some())
                + status_lines
                + self.completion_items().len();
            let cx = rows[1].x + 1 + cursor_col;
            let cy = rows[1].y + 1 + above as u16 + cursor_row;
            frame.set_cursor_position((
                cx.min(rows[1].right().saturating_sub(2)),
                cy.min(rows[1].bottom().saturating_sub(2)),
            ));
        }
    }
}
