//! Isolated eye-declare feasibility spike.
//!
//! Run with:
//! `cargo run -p hi-tui --example eye_declare_spike --features eye-declare-spike`
//!
//! This intentionally models only a streaming chat tail. It is not wired into
//! Hi's production full-screen renderer. It includes a small tool/status
//! block so the comparison covers the minimum tool loop surface as well as
//! streaming text. The spike deliberately does not attempt Hi's full-screen
//! selection, overlay, dashboard, or multi-focus behavior; those remain the
//! reasons to keep Ratatui in the production path unless a larger prototype
//! proves otherwise.

use std::time::Duration;

use crossterm::event::KeyCode;
use eye_declare::{
    App, Ctx, Element, ElementExt, Fluent, Focus, FocusHandle, InputEvent, Keymap, RunOptions,
    Task, TextAreaState, col, driver_tokio, key, keymap, panel, spinner, text, text_area,
};
use futures_util::stream;
use ratatui_core::style::{Color, Modifier, Style};

#[derive(Clone)]
enum Msg {
    Input(InputEvent),
    Submit,
    Chunk(String),
    Done,
    Quit,
}

struct Spike {
    input: TextAreaState,
    input_focus: FocusHandle,
    streaming: String,
    tool_status: Option<String>,
    request: Option<Task>,
    turns: usize,
}

impl Spike {
    fn new() -> Self {
        let input_focus = Focus::new().handle();
        input_focus.focus();
        Self {
            input: TextAreaState::new(),
            input_focus,
            streaming: String::new(),
            tool_status: None,
            request: None,
            turns: 0,
        }
    }

    fn busy(&self) -> bool {
        self.request.is_some()
    }
}

impl App for Spike {
    type Msg = Msg;
    type Output = ();

    fn update(&mut self, msg: Msg, ctx: &mut Ctx<'_, Self>) {
        match msg {
            Msg::Input(event) => self.input.handle(&event),
            Msg::Submit if !self.busy() => {
                let prompt = self.input.take_text().trim().to_string();
                if prompt.is_empty() {
                    return;
                }
                self.turns += 1;
                ctx.push(
                    col()
                        .child(text(" User ").style(Style::default().fg(Color::Cyan)))
                        .child(text(prompt).pad_left(2)),
                );
                self.tool_status = Some("◆ tool · inspect workspace · running".into());
                ctx.push(
                    panel(text(self.tool_status.clone().unwrap_or_default()))
                        .border_style(Style::default().fg(Color::DarkGray)),
                );
                self.streaming.clear();
                self.request = Some(ctx.spawn(fake_stream()));
            }
            Msg::Chunk(chunk) if self.busy() => self.streaming.push_str(&chunk),
            Msg::Done => {
                let reply = std::mem::take(&mut self.streaming);
                self.request = None;
                self.tool_status = Some("✓ tool · inspect workspace · complete".into());
                ctx.push(
                    panel(text(self.tool_status.clone().unwrap_or_default()))
                        .border_style(Style::default().fg(Color::DarkGray)),
                );
                ctx.push(
                    col()
                        .child(
                            text(" Assistant ").style(
                                Style::default()
                                    .fg(Color::Yellow)
                                    .add_modifier(Modifier::BOLD),
                            ),
                        )
                        .child(text(reply).pad_left(2)),
                );
            }
            Msg::Quit => ctx.exit(()),
            Msg::Submit | Msg::Chunk(_) => {}
        }
    }

    fn tail(&self) -> impl Element + '_ {
        col()
            .when(!self.streaming.is_empty(), |column| {
                column.child(text(self.streaming.clone()).pad_left(2))
            })
            .when(self.busy() && self.streaming.is_empty(), |column| {
                column.child(
                    spinner("streaming response…")
                        .label_style(Style::default().fg(Color::DarkGray)),
                )
            })
            .child(
                panel(text_area(&self.input).track_focus(&self.input_focus))
                    .title("eye-declare spike")
                    .title_right(format!("turns {}", self.turns))
                    .footer(if self.busy() {
                        "[Esc] cancel"
                    } else {
                        "[Enter] send [Esc] quit"
                    })
                    .border_style(Style::default().fg(Color::DarkGray)),
            )
    }

    fn keymap(&self) -> Keymap<Msg> {
        let mut map = keymap()
            .on_override(key(KeyCode::Char('c')).ctrl(), Msg::Quit)
            .on_override(key(KeyCode::Esc), Msg::Quit);
        if !self.busy() && !self.input.is_blank() {
            map = map.on(key(KeyCode::Enter), Msg::Submit);
        }
        map.fallthrough(&self.input_focus, Msg::Input)
    }
}

fn fake_stream() -> impl futures_util::Stream<Item = Msg> + Send + use<> {
    let chunks = [
        "This response is rendered as a live tail. ",
        "Committed blocks and the input stay separate. ",
        "That makes the architecture easy to compare with Hi.\n",
    ];
    stream::unfold(0usize, move |index| async move {
        if index < chunks.len() {
            tokio::time::sleep(Duration::from_millis(40)).await;
            Some((Msg::Chunk(chunks[index].to_string()), index + 1))
        } else if index == chunks.len() {
            Some((Msg::Done, index + 1))
        } else {
            None
        }
    })
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> std::io::Result<()> {
    driver_tokio::run_with(Spike::new(), RunOptions::default()).await
}
