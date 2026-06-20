//! Plain stdout frontend — the fallback when not on a TTY or with `--plain`.

use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use hi_agent::Ui;
use hi_agent::preview_args;

pub struct PlainUi {
    /// Set true on the first real output; the REPL's spinner watches it and
    /// stops drawing once the model starts responding.
    progress: Option<Arc<AtomicBool>>,
}

impl PlainUi {
    pub fn new() -> Self {
        Self { progress: None }
    }

    /// A PlainUi that clears a REPL "working…" spinner on its first output.
    pub fn with_progress(progress: Arc<AtomicBool>) -> Self {
        Self {
            progress: Some(progress),
        }
    }

    fn begin_output(&mut self) {
        if let Some(flag) = &self.progress
            && !flag.swap(true, Ordering::Relaxed)
        {
            print!("\r\x1b[K"); // erase the spinner line before the first output
        }
    }
}

impl Ui for PlainUi {
    fn assistant_text(&mut self, text: &str) {
        self.begin_output();
        print!("{text}");
        flush();
    }

    fn assistant_reasoning(&mut self, text: &str) {
        self.begin_output();
        print!("\x1b[2m{text}\x1b[0m");
        flush();
    }

    fn assistant_end(&mut self) {
        println!();
    }

    fn tool_call(&mut self, name: &str, arguments: &str) {
        self.begin_output();
        println!("\x1b[36m⏺ {name}({})\x1b[0m", preview_args(arguments));
    }

    fn tool_result(&mut self, result: &str) {
        const MAX_LINES: usize = 12;
        let lines: Vec<&str> = result.lines().collect();
        for line in lines.iter().take(MAX_LINES) {
            println!("\x1b[2m  {}\x1b[0m", hi_agent::ui::clip(line, 200));
        }
        if lines.len() > MAX_LINES {
            println!("\x1b[2m  … {} more lines\x1b[0m", lines.len() - MAX_LINES);
        }
    }

    fn status(&mut self, text: &str) {
        self.begin_output();
        println!("\x1b[34m{text}\x1b[0m");
    }

    fn turn_end(&mut self, summary: &str) {
        self.begin_output();
        println!("\x1b[2m{summary}\x1b[0m");
    }
}

fn flush() {
    let _ = std::io::stdout().flush();
}

/// Prints only the assistant's text — no tool chatter, reasoning, or usage
/// line. For scripting/piping: `cat data | hi -q "extract emails" | sort`.
pub struct QuietUi;

impl Ui for QuietUi {
    fn assistant_text(&mut self, text: &str) {
        print!("{text}");
        flush();
    }

    fn assistant_reasoning(&mut self, _text: &str) {}
    fn assistant_end(&mut self) {
        println!();
    }
    fn tool_call(&mut self, _name: &str, _arguments: &str) {}
    fn tool_result(&mut self, _result: &str) {}
    fn status(&mut self, _text: &str) {}
    fn turn_end(&mut self, _summary: &str) {}
}
