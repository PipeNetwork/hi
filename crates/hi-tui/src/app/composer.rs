//! Composer (prompt box) line layout: wrapped input, ghost suggestions, and
//! the overlay rows that sit above the input (history search, debug, toasts).
//!
//! Height, paint, and cursor offset all consume the same line lists so an
//! overlay cannot push ghost text onto the bottom border.

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

use crate::layout::display_width;
use crate::render::dim;
use crate::util::{fmt_count, fmt_rate_limits};

const PREFIX_PROMPT: &str = "❯ ";
const PREFIX_CONT: &str = "  ";
const MAX_PROMPT_ROWS: usize = 10;
const COPY_TOAST_MS: u128 = 2500;
/// Leave one unused inner column when attaching ghost text. `❯` is East-Asian
/// Width Ambiguous; some terminals draw it one cell wider than unicode-width,
/// and a ghost that fills the row wraps that extra cell onto the bottom border.
const GHOST_GUTTER: usize = 1;

/// Clip ghost suffix to `max_width` cells without trimming leading spaces
/// (those spaces are part of the remaining suggestion). Adds an ellipsis when
/// the suggestion does not fit, so a hard cut cannot jam into the right border.
fn clip_ghost(text: &str, max_width: usize) -> String {
    if max_width == 0 {
        return String::new();
    }
    if display_width(text) <= max_width {
        return text.to_string();
    }
    if max_width == 1 {
        return "…".to_string();
    }
    let target = max_width - 1;
    let mut width = 0;
    let mut out = String::new();
    for ch in text.chars() {
        let ch_width = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        if width + ch_width > target {
            break;
        }
        width += ch_width;
        out.push(ch);
    }
    out.push('…');
    out
}

fn clip_composer_line(line: Line<'static>, max_width: usize) -> Line<'static> {
    if max_width == 0 {
        return Line::raw("");
    }
    if line.width() <= max_width {
        return line;
    }
    let mut used = 0;
    let mut spans = Vec::new();
    for span in line.spans {
        if used >= max_width {
            break;
        }
        let w = display_width(span.content.as_ref());
        if used + w <= max_width {
            used += w;
            spans.push(span);
            continue;
        }
        let clipped = clip_ghost(span.content.as_ref(), max_width - used);
        if !clipped.is_empty() {
            spans.push(Span::styled(clipped, span.style));
        }
        break;
    }
    Line::from(spans)
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
    /// Overlay rows painted inside the editable input (history search, debug,
    /// help, notices, toasts, voice, palette, completion).
    ///
    /// Built once per frame so box height, paint, and cursor offset stay aligned.
    pub(crate) fn composer_prefix_lines(
        &mut self,
        help_h: usize,
        inner_w: usize,
    ) -> Vec<Line<'static>> {
        let mut lines: Vec<Line<'static>> = Vec::new();
        if let Some(search) = self.mode.history_search() {
            let count = search.matches.len();
            let preview = search
                .current()
                .and_then(|i| self.input.history.get(i))
                .map(|s| s.replace('\n', " "))
                .unwrap_or_default();
            let preview = crate::layout::truncate_display(&preview, 60);
            lines.push(Line::from(vec![
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
            lines.push(Line::styled(format!("  → {preview}"), dim()));
        }
        if self.show_debug {
            lines.extend(self.debug_panel_lines());
        }
        if self.show_help {
            let th = crate::theme::theme();
            lines.push(Line::styled(
                "keybindings (? to close)".to_string(),
                Style::default()
                    .fg(th.accent_system)
                    .add_modifier(Modifier::BOLD),
            ));
            let body_cap = help_h.saturating_sub(1);
            for (keys, help) in crate::keys::help_overlay_rows().into_iter().take(body_cap) {
                if let Some(help) = help {
                    lines.push(Line::from(vec![
                        Span::styled(
                            format!("  {keys:<22}"),
                            Style::default().fg(th.text_primary),
                        ),
                        Span::styled(help.to_string(), dim()),
                    ]));
                } else {
                    lines.push(Line::styled(
                        format!(" {keys}"),
                        Style::default()
                            .fg(th.accent_system)
                            .add_modifier(Modifier::BOLD),
                    ));
                }
            }
        }
        if let Some(notice) = &self.startup_notice {
            lines.push(Line::styled(
                notice.clone(),
                Style::default().fg(crate::theme::theme().warning),
            ));
        }
        if self.checkpoint_warning.is_some() {
            lines.push(Line::styled(
                "⚠ undo warning — see the top bar for details".to_string(),
                Style::default()
                    .fg(crate::theme::theme().warning)
                    .add_modifier(Modifier::BOLD),
            ));
        }
        if self.quit_notice.is_some() {
            lines.push(Line::styled(
                "Press Ctrl-C again to exit",
                Style::default().fg(crate::theme::theme().warning),
            ));
        }
        if let Some((n, at)) = self.copy_toast {
            if at.elapsed().as_millis() < COPY_TOAST_MS {
                lines.push(Line::styled(
                    format!("✓ copied {n} chars to the clipboard"),
                    Style::default().fg(crate::theme::theme().accent_success),
                ));
            } else {
                self.copy_toast = None;
            }
        }
        if let Some(line) = self.voice_indicator() {
            lines.push(line);
        }
        if let Some(palette) = &self.palette {
            let th = crate::theme::theme();
            lines.push(Line::from(vec![
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
                    lines.push(Line::from(vec![
                        Span::styled(
                            format!("▶ {}", item.label),
                            Style::default()
                                .fg(th.accent_system)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(format!("  {}", item.help), dim()),
                    ]));
                } else {
                    lines.push(Line::from(vec![
                        Span::raw(format!("  {}", item.label)),
                        Span::styled(format!("  {}", item.help), dim()),
                    ]));
                }
            }
            lines.push(Line::styled(
                "  ↑↓ move · Enter run · Esc close · type to filter",
                dim(),
            ));
        }
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
            lines.push(Line::from(row));
        }
        lines
            .into_iter()
            .map(|line| clip_composer_line(line, inner_w))
            .collect()
    }

    fn debug_panel_lines(&self) -> Vec<Line<'static>> {
        let mut lines = Vec::new();
        lines.push(Line::styled(
            "agent (Ctrl-? to close)".to_string(),
            Style::default()
                .fg(crate::theme::theme().accent_system)
                .add_modifier(Modifier::BOLD),
        ));
        let t = self.last_telemetry.as_ref();
        let tel = if let Some(t) = t {
            format!(
                "telemetry: {} verify · {} retry · {} repeat · {} continue · {} trunc{} · cache {}s/{}b{}",
                t.verify_rounds,
                t.recovery_retries,
                t.repeat_nudges,
                t.continue_nudges,
                t.truncation_retries,
                if t.stalled_unfinished || t.stalled_repeating {
                    " · stalled"
                } else {
                    ""
                },
                t.prefix_stable_rounds,
                t.prefix_break_rounds,
                if t.tool_prefix_break_rounds > 0 {
                    format!("/{}t", t.tool_prefix_break_rounds)
                } else {
                    String::new()
                }
            )
        } else {
            "telemetry: (no turn yet)".to_string()
        };
        lines.push(Line::styled(tel, dim()));
        if let Some(phase) = self.last_turn_phase {
            lines.push(Line::styled(format!("phase: {phase}"), dim()));
        }
        if let Some(t) = self.last_telemetry.as_ref() {
            lines.push(Line::styled(
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
                    lines.push(Line::styled(chunk.to_string(), dim()));
                }
            }
        }
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
            lines.push(Line::styled(sched, dim()));
        }
        if let Some(t) = self.last_telemetry.as_ref() {
            let latency = &t.phase_latencies;
            lines.push(Line::styled(
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
        lines.push(Line::styled(
            format!("tool calls this turn: {}", self.turn_tool_calls),
            dim(),
        ));
        let (input, output) = self.usage;
        let ctx = if let Some(pct) = self.context_pct() {
            format!(
                " · ctx {}{pct}%",
                if self.usage_estimated { "~" } else { "" }
            )
        } else {
            String::new()
        };
        lines.push(Line::styled(
            format!(
                "turn: user prompt estimate {} · output across all model calls {}{}{ctx}",
                fmt_count(input),
                if self.usage_estimated { "~" } else { "" },
                fmt_count(output)
            ),
            dim(),
        ));
        if let Some(limits) = fmt_rate_limits(self.rate_limits) {
            lines.push(Line::styled(limits, dim()));
        }
        lines
    }

    /// The last-turn file summary is session chrome, not prompt content. Keep
    /// it on its own row immediately above the editable prompt box.
    pub(crate) fn changed_files_line(&self) -> Option<Line<'static>> {
        if self.last_changed_files.is_empty() || self.working {
            return None;
        }
        let summary = self
            .last_changed_files
            .iter()
            .map(|p| p.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        Some(Line::styled(
            format!("changed: {summary}  (Ctrl-G for review)"),
            dim(),
        ))
    }

    /// The input row(s), or the normal-mode banner that replaces them.
    pub(crate) fn composer_body_lines(
        &self,
        input_lines: Vec<Line<'static>>,
        inner_w: usize,
    ) -> Vec<Line<'static>> {
        if !self.mode.is_normal() {
            return input_lines;
        }
        let th = crate::theme::theme();
        let line = if let Some(q) = self.mode.normal_search() {
            Line::from(vec![
                Span::styled("-- SEARCH -- ", Style::default().fg(th.warning)),
                Span::styled(
                    format!("/{q}"),
                    Style::default().fg(crate::theme::theme().text_primary),
                ),
                Span::styled("▏", Style::default().fg(crate::theme::theme().gray_dim)),
            ])
        } else {
            Line::from(vec![
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
            ])
        };
        vec![clip_composer_line(line, inner_w)]
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
    pub(crate) fn input_view(&self, width: u16) -> (Vec<Line<'static>>, u16, u16) {
        let raw = self.input.text();
        let text = if self.pending_auth.is_some() {
            raw.chars().map(|_| '•').collect()
        } else {
            hi_agent::command::mask_secret_input(&raw)
        };
        let before: String = text.chars().take(self.input.cursor()).collect();
        let cursor_col_logical = display_width(
            before
                .rsplit_once('\n')
                .map(|(_, line)| line)
                .unwrap_or(&before),
        );

        let prefix_w = display_width(PREFIX_PROMPT).max(display_width(PREFIX_CONT));
        let wrap_w = width.saturating_sub(prefix_w as u16).max(1) as usize;

        let all: Vec<&str> = text.split('\n').collect();
        let cursor_logical_row = before.matches('\n').count();

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
                if end == start {
                    end += 1;
                    chunk_width = unicode_width::UnicodeWidthChar::width(chars[start]).unwrap_or(0);
                }
                let chunk: String = chars[start..end].iter().collect();
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
            let first = i == 0 && !truncated;
            let this_prefix = if first { PREFIX_PROMPT } else { PREFIX_CONT };
            let this_prefix_w = display_width(this_prefix);
            let prefix_span = if first {
                Span::styled(
                    PREFIX_PROMPT,
                    Style::default()
                        .fg(crate::theme::theme().accent_user)
                        .add_modifier(Modifier::BOLD),
                )
            } else {
                Span::raw(PREFIX_CONT)
            };
            let suffix = if i == last_visible {
                ghost.as_deref()
            } else {
                None
            };
            let mut spans = vec![prefix_span];
            let inner = width as usize;
            let ghost_budget = inner
                .saturating_sub(this_prefix_w)
                .saturating_sub(display_width(chunk))
                .saturating_sub(GHOST_GUTTER);
            if first && chunk.is_empty() {
                if let Some(suggestion) = suffix {
                    let shown = clip_ghost(suggestion, ghost_budget);
                    if !shown.is_empty() {
                        spans.push(Span::styled(shown, dim()));
                    }
                }
            } else {
                spans.extend(crate::file_mentions::mention_spans(chunk));
                if let Some(suggestion) = suffix
                    && ghost_budget > 0
                {
                    spans.push(Span::styled(clip_ghost(suggestion, ghost_budget), dim()));
                }
            }
            lines.push(Line::from(spans));
            if let Some(col) = cursor_here
                && !found_cursor
            {
                cursor_row = u16::from(truncated) + i as u16;
                cursor_col = (this_prefix_w + col) as u16;
                found_cursor = true;
            }
        }
        if !found_cursor {
            cursor_row = lines.len().saturating_sub(1) as u16;
            cursor_col = prefix_w as u16;
        }
        (lines, cursor_row, cursor_col)
    }
}

#[cfg(test)]
mod tests {
    use super::{clip_composer_line, clip_ghost};
    use ratatui::text::{Line, Span};

    #[test]
    fn clip_ghost_keeps_leading_space_and_ellipsizes() {
        assert_eq!(clip_ghost(" unit tests", 20), " unit tests");
        assert_eq!(clip_ghost("abcdef", 4), "abc…");
        assert_eq!(clip_ghost("  rest", 4), "  r…");
        assert_eq!(clip_ghost("xy", 0), "");
        assert_eq!(clip_ghost("xy", 1), "…");
    }

    #[test]
    fn clip_composer_line_preserves_fits_and_cuts_overflow() {
        let short = Line::from(vec![Span::raw("abc"), Span::raw("def")]);
        assert_eq!(clip_composer_line(short.clone(), 10).to_string(), "abcdef");
        assert_eq!(clip_composer_line(short, 4).to_string(), "abc…");
    }
}
