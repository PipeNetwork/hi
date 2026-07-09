//! `/watch` — a full-screen live dashboard of every active `/loop`.
//!
//! The loop manager (see [`crate::loops`]) is one background task that fires
//! loops on their cadence and keeps a live snapshot of each one — whether it's
//! firing right now, when it fires next, and its recent results. `/watch`
//! renders that snapshot every tick: a table of all loops with live countdowns
//! and a peek panel of the selected loop's firing history. From here you can
//! arm a new loop, fire one immediately, or cancel one — the same controls as
//! `/loop`, on one screen. Closing `/watch` (Esc) returns to the chat; the
//! loops keep firing in the background regardless.
//!
//! This view owns no work of its own: it reads the manager's published snapshot
//! and sends [`LoopCtl`] messages. That's why it can share the chat's terminal,
//! input channel, and ticker and hand them straight back on exit.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Paragraph, Wrap};
use tokio::sync::{mpsc, oneshot};

use crate::input::InputLine;
use crate::loops::{LoopCtl, LoopWatchRow, fmt_tokens, humanize_secs};
use crate::render::dim;
use crate::{App, SPINNER};

/// Which pane has the keyboard: the loop list (single-key controls) or the
/// compose box (typing an interval + prompt to arm a new loop).
#[derive(Clone, Copy, PartialEq)]
enum Focus {
    List,
    Compose,
}

/// Full-screen loop dashboard. Runs over the chat's terminal/input/ticker and
/// returns to the chat on Esc; the loop manager keeps firing throughout.
pub(crate) async fn run_watch(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    input_rx: &mut mpsc::UnboundedReceiver<Event>,
    ticker: &mut tokio::time::Interval,
    app: &mut App,
) -> Result<()> {
    // The manager owns all loop state; we only read its snapshot and send it
    // control messages. Both handles are cheap clones, so grab them once and
    // drop the borrow on `app` (still needed for focus-aware pings).
    let Some((ctl, snap)) = app
        .loops
        .as_ref()
        .map(|h| (h.ctl.clone(), h.snapshot.clone()))
    else {
        return Ok(());
    };

    let mut selected: usize = 0;
    let mut focus = Focus::List;
    let mut compose = InputLine::default();
    let mut flash: Option<String> = None;
    // Peek history scroll: rows back from the newest firing (0 = follow newest).
    let mut hist_offset: usize = 0;
    // Loop id → the `last_fired_ms` we've already reacted to (for ping-on-loud).
    let mut seen_fire: HashMap<u64, u64> = HashMap::new();

    loop {
        let rows = snap.lock().unwrap().clone();
        if selected >= rows.len() {
            selected = rows.len().saturating_sub(1);
        }

        // Ping when a firing newly reports something worth telling you about and
        // the terminal is unfocused — you asked to watch, so a change that lands
        // while you're looking away should still reach you.
        for row in &rows {
            if row.last_fired_ms == 0 {
                continue;
            }
            let prev = seen_fire.insert(row.id, row.last_fired_ms);
            let is_new = prev != Some(row.last_fired_ms);
            if is_new && prev.is_some() && !row.last_quiet && app.focus_known && !app.focused {
                crate::util::notify_done();
            }
        }
        seen_fire.retain(|id, _| rows.iter().any(|r| r.id == *id));

        terminal.draw(|f| {
            render(
                f,
                &rows,
                selected,
                focus,
                &compose,
                hist_offset,
                flash.as_deref(),
            )
        })?;

        tokio::select! {
            _ = ticker.tick() => {
                app.spinner = app.spinner.wrapping_add(1);
            }
            maybe = input_rx.recv() => {
                let Some(event) = maybe else { return Ok(()) };
                match event {
                    Event::FocusGained => app.set_focus(true),
                    Event::FocusLost => app.set_focus(false),
                    Event::Paste(text) if focus == Focus::Compose => compose.insert_str(&text),
                    Event::Key(key) if key.kind == KeyEventKind::Press => {
                        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                        flash = None;

                        // Ctrl+C always closes the dashboard.
                        if ctrl && matches!(key.code, KeyCode::Char('c')) {
                            return Ok(());
                        }

                        if focus == Focus::Compose {
                            match key.code {
                                KeyCode::Esc | KeyCode::Tab => focus = Focus::List,
                                KeyCode::Enter => {
                                    let text = compose.submit();
                                    if let Some(msg) = arm_from_compose(&ctl, text).await {
                                        flash = Some(msg);
                                    }
                                    focus = Focus::List;
                                }
                                KeyCode::Char(c) if !ctrl => compose.insert(c),
                                KeyCode::Backspace => compose.backspace(),
                                KeyCode::Left => compose.left(),
                                KeyCode::Right => compose.right(),
                                KeyCode::Home => compose.home(),
                                KeyCode::End => compose.end(),
                                KeyCode::Char('u') if ctrl => compose.kill_to_start(),
                                _ => {}
                            }
                            continue;
                        }

                        // List focus.
                        match key.code {
                            KeyCode::Esc => return Ok(()),
                            KeyCode::Up | KeyCode::Char('k') => {
                                selected = selected.saturating_sub(1);
                                hist_offset = 0;
                            }
                            KeyCode::Down | KeyCode::Char('j') => {
                                if !rows.is_empty() {
                                    selected = (selected + 1).min(rows.len() - 1);
                                }
                                hist_offset = 0;
                            }
                            KeyCode::PageUp => hist_offset = hist_offset.saturating_add(5),
                            KeyCode::PageDown => hist_offset = hist_offset.saturating_sub(5),
                            KeyCode::Tab | KeyCode::Char('n') => {
                                focus = Focus::Compose;
                                hist_offset = 0;
                            }
                            KeyCode::Char('f') => {
                                if let Some(row) = rows.get(selected) {
                                    let (tx, rx) = oneshot::channel();
                                    let _ = ctl.send(LoopCtl::FireNow { id: row.id, reply: tx });
                                    flash = Some(match rx.await {
                                        Ok(true) => format!("firing loop#{} now…", row.id),
                                        _ => format!("no loop#{}", row.id),
                                    });
                                }
                            }
                            KeyCode::Char('c') => {
                                if let Some(row) = rows.get(selected) {
                                    let (tx, rx) = oneshot::channel();
                                    let _ = ctl.send(LoopCtl::Cancel { id: row.id, reply: tx });
                                    flash = Some(match rx.await {
                                        Ok(true) => format!("cancelled loop#{}", row.id),
                                        _ => format!("no loop#{}", row.id),
                                    });
                                }
                            }
                            KeyCode::Char('p') => {
                                if let Some(row) = rows.get(selected) {
                                    let on = !row.paused;
                                    let (tx, rx) = oneshot::channel();
                                    let _ = ctl.send(LoopCtl::Pause {
                                        id: row.id,
                                        on,
                                        reply: tx,
                                    });
                                    flash = Some(match rx.await {
                                        Ok(true) if on => format!("paused loop#{}", row.id),
                                        Ok(true) => format!("resumed loop#{}", row.id),
                                        _ => format!("no loop#{}", row.id),
                                    });
                                }
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

/// Parse a compose-box entry (`<interval> <prompt>`, or `cancel <id>`) with the
/// same grammar as `/loop`, send the control message, and return a flash line.
async fn arm_from_compose(ctl: &mpsc::UnboundedSender<LoopCtl>, text: String) -> Option<String> {
    if text.trim().is_empty() {
        return None;
    }
    match hi_agent::command::parse_loop_arg(text.trim()) {
        hi_agent::command::LoopArg::Create { secs, prompt } => {
            let (tx, rx) = oneshot::channel();
            let _ = ctl.send(LoopCtl::Create {
                secs,
                prompt,
                reply: tx,
            });
            match rx.await {
                Ok(Ok(spec)) => Some(format!(
                    "armed loop#{} — every {}, firing now",
                    spec.id,
                    humanize_secs(spec.interval_secs)
                )),
                Ok(Err(err)) => Some(err),
                Err(_) => None,
            }
        }
        hi_agent::command::LoopArg::Cancel(id) => {
            let (tx, rx) = oneshot::channel();
            let _ = ctl.send(LoopCtl::Cancel { id, reply: tx });
            Some(match rx.await {
                Ok(true) => format!("cancelled loop#{id}"),
                _ => format!("no loop#{id}"),
            })
        }
        hi_agent::command::LoopArg::Pause(id) => {
            let (tx, rx) = oneshot::channel();
            let _ = ctl.send(LoopCtl::Pause {
                id,
                on: true,
                reply: tx,
            });
            Some(match rx.await {
                Ok(true) => format!("paused loop#{id}"),
                _ => format!("no loop#{id}"),
            })
        }
        hi_agent::command::LoopArg::Resume(id) => {
            let (tx, rx) = oneshot::channel();
            let _ = ctl.send(LoopCtl::Pause {
                id,
                on: false,
                reply: tx,
            });
            Some(match rx.await {
                Ok(true) => format!("resumed loop#{id}"),
                _ => format!("no loop#{id}"),
            })
        }
        hi_agent::command::LoopArg::Budget { id, tokens } => {
            let (tx, rx) = oneshot::channel();
            let _ = ctl.send(LoopCtl::Budget {
                id,
                tokens,
                reply: tx,
            });
            Some(match (rx.await, tokens) {
                (Ok(true), Some(t)) => format!("loop#{id} budget {}", fmt_tokens(t)),
                (Ok(true), None) => format!("loop#{id} budget cleared"),
                _ => format!("no loop#{id}"),
            })
        }
        hi_agent::command::LoopArg::Trigger { id, cmd } => {
            let set = cmd.is_some();
            let (tx, rx) = oneshot::channel();
            let _ = ctl.send(LoopCtl::Trigger { id, cmd, reply: tx });
            Some(match (rx.await, set) {
                (Ok(true), true) => format!("loop#{id} on-change command set"),
                (Ok(true), false) => format!("loop#{id} trigger cleared"),
                _ => format!("no loop#{id}"),
            })
        }
        hi_agent::command::LoopArg::Fix { id, on } => {
            let (tx, rx) = oneshot::channel();
            let _ = ctl.send(LoopCtl::Fix { id, on, reply: tx });
            Some(match (rx.await, on) {
                (Ok(true), true) => format!("loop#{id} auto-fix on"),
                (Ok(true), false) => format!("loop#{id} auto-fix off"),
                _ => format!("no loop#{id}"),
            })
        }
        hi_agent::command::LoopArg::List => None,
        hi_agent::command::LoopArg::Invalid(msg) => Some(msg),
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Compact single-unit duration for countdowns: `45s`, `12m`, `3h`, `2d`.
fn fmt_left(secs: u64) -> String {
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m", secs / 60)
    } else if secs < 86_400 {
        format!("{}h", secs / 3600)
    } else {
        format!("{}d", secs / 86_400)
    }
}

fn render(
    frame: &mut ratatui::Frame,
    rows: &[LoopWatchRow],
    selected: usize,
    focus: Focus,
    compose: &InputLine,
    hist_offset: usize,
    flash: Option<&str>,
) {
    let chunks = Layout::vertical([
        Constraint::Length(1),  // header
        Constraint::Min(3),     // table
        Constraint::Length(11), // peek
        Constraint::Length(1),  // hints
        Constraint::Length(3),  // compose
    ])
    .split(frame.area());

    render_header(frame, rows, flash, chunks[0]);
    render_table(frame, rows, selected, chunks[1]);
    render_peek(frame, rows.get(selected), hist_offset, chunks[2]);
    render_hints(frame, focus, chunks[3]);
    render_compose(frame, focus, compose, chunks[4]);
}

fn render_header(
    frame: &mut ratatui::Frame,
    rows: &[LoopWatchRow],
    flash: Option<&str>,
    area: Rect,
) {
    let firing = rows.iter().filter(|r| r.firing).count();
    let mut spans = vec![
        Span::styled(
            "⟳ watch",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!(
                "  ·  {} active loop{}",
                rows.len(),
                if rows.len() == 1 { "" } else { "s" }
            ),
            dim(),
        ),
    ];
    if firing > 0 {
        spans.push(Span::styled(
            format!("  ·  {firing} firing"),
            Style::default().fg(Color::Yellow),
        ));
    }
    if let Some(msg) = flash {
        spans.push(Span::styled(
            format!("   {msg}"),
            Style::default().fg(Color::Yellow),
        ));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn render_table(frame: &mut ratatui::Frame, rows: &[LoopWatchRow], selected: usize, area: Rect) {
    let now = now_ms();
    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::styled(
        format!(
            "   {:<4} {:<7} {:<11} {:>5} {:>11}  {}",
            "id", "every", "next", "fires", "spent", "last result"
        ),
        dim(),
    ));

    if rows.is_empty() {
        lines.push(Line::styled(
            "  no active loops — press n to arm one:  <interval> <prompt>  (e.g. 30m check CI on main)",
            dim(),
        ));
    }

    for (i, row) in rows.iter().enumerate() {
        let sel = i == selected;
        // Leading glyph: spinner while firing, ⏸ when paused, else a
        // loud/quiet/none result marker.
        let (glyph, glyph_style) = if row.firing {
            (
                SPINNER[frame_tick(row) % SPINNER.len()].to_string(),
                Style::default().fg(Color::Yellow),
            )
        } else if row.paused {
            ("⏸".to_string(), dim())
        } else if row.last_fired_ms == 0 {
            (" ".to_string(), dim())
        } else if row.last_quiet {
            ("·".to_string(), dim())
        } else {
            ("●".to_string(), Style::default().fg(Color::Cyan))
        };

        let next = if row.firing {
            "firing…".to_string()
        } else if row.paused {
            "paused".to_string()
        } else if row.next_ms <= now {
            "due".to_string()
        } else {
            format!("in {}", fmt_left((row.next_ms - now) / 1000))
        };

        // Spend column: `spent/budget` when a budget is set, else spend alone.
        let spent = match row.token_budget {
            Some(b) => format!("{}/{}", fmt_tokens(row.spent_tokens), fmt_tokens(b)),
            None if row.spent_tokens > 0 => fmt_tokens(row.spent_tokens),
            None => "—".to_string(),
        };

        let (last, last_style) = match &row.last_summary {
            _ if row.fixing => ("⚒ fixing…".to_string(), Style::default().fg(Color::Magenta)),
            _ if row.firing => ("checking…".to_string(), Style::default().fg(Color::Yellow)),
            Some(_) if row.last_quiet => ("· nothing new".to_string(), dim()),
            Some(s) => (truncate(s, 60), Style::default().fg(Color::White)),
            None => ("—".to_string(), dim()),
        };

        let body = format!(
            "#{:<3} {:<7} {:<11} {:>5} {:>11}  ",
            row.id,
            humanize_secs(row.interval_secs),
            next,
            row.firings,
            spent,
        );
        let row_style = if sel {
            Style::default().add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };
        // A ⚡ before the result marks a loop that runs an on-change command.
        let mut spans = vec![
            Span::styled(
                if sel { "▸ " } else { "  " },
                Style::default().fg(Color::Cyan),
            ),
            Span::styled(glyph, glyph_style),
            Span::styled(" ", row_style),
            Span::styled(body, row_style),
        ];
        if row.trigger.is_some() {
            spans.push(Span::styled("⚡ ", Style::default().fg(Color::Magenta)));
        }
        if row.autofix {
            spans.push(Span::styled("⚒ ", Style::default().fg(Color::Magenta)));
        }
        spans.push(Span::styled(
            last,
            if sel {
                last_style.add_modifier(Modifier::BOLD)
            } else {
                last_style
            },
        ));
        lines.push(Line::from(spans));
    }

    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(dim())
        .title(" loops ");
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

fn render_peek(
    frame: &mut ratatui::Frame,
    row: Option<&LoopWatchRow>,
    hist_offset: usize,
    area: Rect,
) {
    let now = now_ms();
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(dim());
    let Some(row) = row else {
        frame.render_widget(
            Paragraph::new(Line::styled("no loop selected", dim())).block(block),
            area,
        );
        return;
    };

    let block = block.title(format!(" loop #{} · {} ", row.id, row.name));
    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::styled(
        row.prompt.clone(),
        Style::default().fg(Color::White),
    ));
    let created_ago = fmt_left(now.saturating_sub(row.created_ms) / 1000);
    let expires_in = fmt_left(row.expires_ms.saturating_sub(now) / 1000);
    lines.push(Line::styled(
        format!(
            "every {} · {} firing(s) · started {} ago · expires in {}",
            humanize_secs(row.interval_secs),
            row.firings,
            created_ago,
            expires_in,
        ),
        dim(),
    ));
    // Cost + pause status line.
    let mut status: Vec<Span> = Vec::new();
    if row.paused {
        status.push(Span::styled("⏸ paused", Style::default().fg(Color::Yellow)));
        status.push(Span::styled("  ·  ", dim()));
    }
    match row.token_budget {
        Some(b) => status.push(Span::styled(
            format!(
                "{} / {} tokens",
                fmt_tokens(row.spent_tokens),
                fmt_tokens(b)
            ),
            if row.spent_tokens >= b {
                Style::default().fg(Color::Yellow)
            } else {
                dim()
            },
        )),
        None => status.push(Span::styled(
            format!(
                "{} tokens spent · no budget (p pauses)",
                fmt_tokens(row.spent_tokens)
            ),
            dim(),
        )),
    }
    lines.push(Line::from(status));
    // On-change trigger command + its last outcome.
    if let Some(cmd) = &row.trigger {
        let mut trig = vec![
            Span::styled("⚡ on change: ", Style::default().fg(Color::Magenta)),
            Span::styled(truncate(cmd, 72), dim()),
        ];
        if let Some(out) = &row.last_trigger {
            trig.push(Span::styled(
                format!("  (last: {})", truncate(out, 40)),
                if out.starts_with("ok") {
                    dim()
                } else {
                    Style::default().fg(Color::Yellow)
                },
            ));
        }
        lines.push(Line::from(trig));
    }
    // Auto-fix status + its last outcome.
    if row.autofix {
        let mut fix = vec![Span::styled(
            "⚒ auto-fix: on",
            Style::default().fg(Color::Magenta),
        )];
        if row.fixing {
            fix.push(Span::styled(
                "  · fixing now…",
                Style::default().fg(Color::Yellow),
            ));
        } else if let Some(out) = &row.last_fix {
            fix.push(Span::styled(
                format!("  (last: {})", truncate(out, 44)),
                dim(),
            ));
        }
        lines.push(Line::from(fix));
    }
    lines.push(Line::raw(""));

    if row.firing {
        lines.push(Line::styled(
            format!("{} checking now…", SPINNER[frame_tick(row) % SPINNER.len()]),
            Style::default().fg(Color::Yellow),
        ));
    }

    if row.history.is_empty() && !row.firing {
        lines.push(Line::styled(
            "no firings yet — the first check runs on the next tick",
            dim(),
        ));
    } else if !row.history.is_empty() {
        let total = row.history.len();
        // Newest first; hist_offset scrolls toward older entries.
        let shown = 6usize;
        let start = hist_offset.min(total.saturating_sub(1));
        lines.push(Line::styled(
            if start > 0 {
                format!("recent checks (↑{start} older):")
            } else {
                "recent checks:".to_string()
            },
            dim(),
        ));
        for item in row.history.iter().rev().skip(start).take(shown) {
            let age = fmt_left(now.saturating_sub(item.at_ms) / 1000);
            let (mark, mark_style) = if item.quiet {
                ("·", dim())
            } else {
                ("●", Style::default().fg(Color::Cyan))
            };
            let text = if item.quiet {
                "nothing new".to_string()
            } else {
                truncate(&item.summary, 72)
            };
            lines.push(Line::from(vec![
                Span::styled(format!("{age:>5} "), dim()),
                Span::styled(format!("{mark} "), mark_style),
                Span::styled(text, if item.quiet { dim() } else { Style::default() }),
            ]));
        }
    }

    frame.render_widget(
        Paragraph::new(lines).block(block).wrap(Wrap { trim: true }),
        area,
    );
}

fn render_hints(frame: &mut ratatui::Frame, focus: Focus, area: Rect) {
    let hint = match focus {
        Focus::List => {
            "↑↓ select · f fire · p pause · c cancel · n arm · PgUp/Dn history · Esc close"
        }
        Focus::Compose => {
            "type <interval> <prompt> · pause|resume|budget|on|fix <id> … · Enter · Esc/Tab back"
        }
    };
    frame.render_widget(
        Paragraph::new(Line::styled(format!("  {hint}"), dim())),
        area,
    );
}

fn render_compose(frame: &mut ratatui::Frame, focus: Focus, compose: &InputLine, area: Rect) {
    let accent = if focus == Focus::Compose {
        Color::Cyan
    } else {
        Color::DarkGray
    };
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(accent))
        .title(" arm a loop — <interval> <prompt> ");
    let text = compose.text();
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("› ", dim()),
            Span::raw(text.clone()),
        ]))
        .block(block),
        area,
    );
    if focus == Focus::Compose {
        let cursor_col = compose.cursor().min(text.chars().count()) as u16;
        frame.set_cursor_position((area.x + 3 + cursor_col, area.y + 1));
    }
}

/// A per-row spinner phase so multiple firing rows don't spin in lockstep.
fn frame_tick(row: &LoopWatchRow) -> usize {
    // Advance with wall-clock so the spinner animates even between ticker ticks,
    // offset by id so rows are visually distinct.
    (now_ms() / 90) as usize + row.id as usize
}

fn truncate(s: &str, max: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let cut: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{cut}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmt_left_units() {
        assert_eq!(fmt_left(45), "45s");
        assert_eq!(fmt_left(90), "1m");
        assert_eq!(fmt_left(3600), "1h");
        assert_eq!(fmt_left(2 * 86_400), "2d");
    }

    #[test]
    fn truncate_adds_ellipsis() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("  padded  ", 10), "padded");
        assert_eq!(truncate("abcdefghij", 5), "abcd…");
    }

    use crate::loops::HistItem;
    use ratatui::backend::TestBackend;

    fn dump(term: &Terminal<TestBackend>) -> String {
        let buf = term.backend().buffer();
        let mut out = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                out.push_str(buf[(x, y)].symbol());
            }
            out.push('\n');
        }
        out
    }

    fn sample_rows() -> Vec<LoopWatchRow> {
        let now = now_ms();
        vec![
            LoopWatchRow {
                id: 1,
                name: "watch CI on main".into(),
                prompt: "check whether CI on main is green".into(),
                interval_secs: 1800,
                created_ms: now.saturating_sub(3_600_000),
                next_ms: now + 1_200_000,
                expires_ms: now + 6 * 86_400_000,
                firings: 4,
                firing: false,
                paused: false,
                token_budget: Some(500_000),
                spent_tokens: 123_000,
                trigger: Some("notify-send 'CI red'".into()),
                last_trigger: Some("ok".into()),
                autofix: true,
                fixing: false,
                last_fix: Some("fixed & merged 1 file(s): parser.rs".into()),
                last_summary: Some("CI went red: 3 parser test failures".into()),
                last_quiet: false,
                last_fired_ms: now.saturating_sub(120_000),
                history: vec![
                    HistItem {
                        at_ms: now.saturating_sub(3_600_000),
                        quiet: true,
                        summary: "NOTHING NEW".into(),
                    },
                    HistItem {
                        at_ms: now.saturating_sub(120_000),
                        quiet: false,
                        summary: "CI went red: 3 parser test failures".into(),
                    },
                ],
            },
            LoopWatchRow {
                id: 2,
                name: "prod p99".into(),
                prompt: "watch prod p99 latency".into(),
                interval_secs: 300,
                created_ms: now,
                next_ms: now + 60_000,
                expires_ms: now + 7 * 86_400_000,
                firings: 1,
                firing: true,
                paused: false,
                token_budget: None,
                spent_tokens: 0,
                trigger: None,
                last_trigger: None,
                autofix: false,
                fixing: false,
                last_fix: None,
                last_summary: None,
                last_quiet: false,
                last_fired_ms: 0,
                history: vec![],
            },
        ]
    }

    #[test]
    fn renders_table_countdown_and_peek() {
        let rows = sample_rows();
        let mut term = Terminal::new(TestBackend::new(120, 30)).unwrap();
        let compose = InputLine::default();
        term.draw(|f| render(f, &rows, 0, Focus::List, &compose, 0, None))
            .unwrap();
        let s = dump(&term);

        // Header + counts.
        assert!(s.contains("⟳ watch"), "{s}");
        assert!(s.contains("2 active loops"), "{s}");
        assert!(s.contains("1 firing"), "{s}");
        // Both loops in the table, with a live countdown for the idle one.
        assert!(s.contains("#1"), "{s}");
        assert!(s.contains("#2"), "{s}");
        assert!(s.contains("in "), "table shows a countdown\n{s}");
        assert!(s.contains("firing…"), "firing row shows firing…\n{s}");
        // Cost column: spent/budget for the loop that has one.
        assert!(s.contains("123k/500k"), "spend/budget column\n{s}");
        // Selected row 1 → peek shows its prompt + history (loud + quiet) + budget.
        assert!(s.contains("check whether CI on main is green"), "{s}");
        assert!(s.contains("recent checks"), "{s}");
        assert!(s.contains("CI went red"), "{s}");
        assert!(s.contains("nothing new"), "quiet history rendered\n{s}");
        assert!(s.contains("123k / 500k tokens"), "peek budget line\n{s}");
        // The trigger loop shows ⚡ and its command in the peek.
        assert!(s.contains("⚡"), "trigger marker\n{s}");
        assert!(s.contains("on change:"), "peek trigger line\n{s}");
        assert!(s.contains("notify-send"), "peek trigger command\n{s}");
        // Auto-fix marker + peek line.
        assert!(s.contains("⚒"), "auto-fix marker\n{s}");
        assert!(s.contains("auto-fix: on"), "peek auto-fix line\n{s}");
        // List-focus hints.
        assert!(s.contains("p pause"), "hints show pause\n{s}");
    }

    #[test]
    fn renders_paused_loop() {
        let now = now_ms();
        let rows = vec![LoopWatchRow {
            id: 7,
            name: "nightly build".into(),
            prompt: "watch the nightly build".into(),
            interval_secs: 3600,
            created_ms: now,
            next_ms: now + 3600_000,
            expires_ms: now + 7 * 86_400_000,
            firings: 2,
            firing: false,
            paused: true,
            token_budget: Some(200_000),
            spent_tokens: 200_000,
            trigger: None,
            last_trigger: None,
            autofix: false,
            fixing: false,
            last_fix: None,
            last_summary: Some("build still green".into()),
            last_quiet: true,
            last_fired_ms: now.saturating_sub(60_000),
            history: vec![],
        }];
        let mut term = Terminal::new(TestBackend::new(110, 24)).unwrap();
        let compose = InputLine::default();
        term.draw(|f| render(f, &rows, 0, Focus::List, &compose, 0, None))
            .unwrap();
        let s = dump(&term);
        assert!(s.contains("paused"), "paused shows in the next column\n{s}");
        assert!(s.contains("⏸"), "paused glyph\n{s}");
        assert!(s.contains("200k/200k"), "at-budget spend\n{s}");
    }

    #[test]
    fn renders_empty_state_and_compose_focus() {
        let mut term = Terminal::new(TestBackend::new(100, 24)).unwrap();
        let mut compose = InputLine::default();
        compose.insert_str("30m check the canary");
        term.draw(|f| render(f, &[], 0, Focus::Compose, &compose, 0, None))
            .unwrap();
        let s = dump(&term);
        assert!(s.contains("no active loops"), "{s}");
        assert!(
            s.contains("30m check the canary"),
            "compose text shown\n{s}"
        );
        assert!(s.contains("Esc/Tab back"), "compose hints shown\n{s}");
    }
}
