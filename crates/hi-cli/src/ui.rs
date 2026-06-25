//! Plain stdout frontend — the fallback when not on a TTY or with `--plain`.

use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use hi_agent::Ui;
use hi_agent::tool_label;

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
        println!("\x1b[36m⏺ {}\x1b[0m", tool_label(name, arguments));
    }

    fn tool_result(&mut self, result: &str) {
        // Enough to show a small edit's diff with its context inline; larger
        // results truncate with a footer (use `/diff` for the full diff).
        const MAX_LINES: usize = 16;
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

    fn plan(&mut self, steps: &[hi_agent::PlanStep]) {
        use hi_agent::PlanStatus;
        self.begin_output();
        let done = steps
            .iter()
            .filter(|s| s.status == PlanStatus::Done)
            .count();
        println!("\x1b[1m⏺ plan · {done}/{}\x1b[0m", steps.len());
        for s in steps {
            let (glyph, color) = match s.status {
                PlanStatus::Done => ('✓', "\x1b[32m"),
                PlanStatus::Active => ('▸', "\x1b[36m"),
                PlanStatus::Pending => ('☐', "\x1b[2m"),
            };
            println!("{color}  {glyph} {}\x1b[0m", s.title);
        }
    }

    fn turn_end(&mut self, summary: &str) {
        self.begin_output();
        println!("\x1b[2m{summary}\x1b[0m");
    }

    fn usage(
        &mut self,
        _input_tokens: u64,
        _output_tokens: u64,
        context_used: u64,
        context_window: Option<u32>,
    ) {
        // Show a context-fill percentage when the window is known,
        // so the user can see when auto-compaction is approaching.
        if let Some(window) = context_window
            && window > 0
        {
            let pct = (context_used * 100 / window as u64).min(100);
            if pct >= 60 {
                self.begin_output();
                let bar = "█".repeat((pct / 5) as usize);
                let pad = "░".repeat(20 - (pct / 5) as usize);
                println!("\x1b[2m  ctx [{bar}{pad}] {pct}%\x1b[0m");
            }
        }
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
