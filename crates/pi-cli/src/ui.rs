//! Plain stdout frontend — the default and the fallback when not on a TTY.

use std::io::Write;

use pi_agent::Ui;
use pi_agent::preview_args;

pub struct PlainUi;

impl Ui for PlainUi {
    fn assistant_text(&mut self, text: &str) {
        print!("{text}");
        flush();
    }

    fn assistant_reasoning(&mut self, text: &str) {
        print!("\x1b[2m{text}\x1b[0m");
        flush();
    }

    fn assistant_end(&mut self) {
        println!();
    }

    fn tool_call(&mut self, name: &str, arguments: &str) {
        println!("\x1b[36m⏺ {name}({})\x1b[0m", preview_args(arguments));
    }

    fn tool_result(&mut self, result: &str) {
        const MAX_LINES: usize = 12;
        let lines: Vec<&str> = result.lines().collect();
        for line in lines.iter().take(MAX_LINES) {
            println!("\x1b[2m  {}\x1b[0m", pi_agent::ui::clip(line, 200));
        }
        if lines.len() > MAX_LINES {
            println!("\x1b[2m  … {} more lines\x1b[0m", lines.len() - MAX_LINES);
        }
    }

    fn status(&mut self, text: &str) {
        println!("\x1b[34m{text}\x1b[0m");
    }

    fn turn_end(&mut self, summary: &str) {
        println!("\x1b[2m{summary}\x1b[0m");
    }
}

fn flush() {
    let _ = std::io::stdout().flush();
}
