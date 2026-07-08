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
    /// For read/list/grep we defer the `⏺` header until the result lands, so the
    /// file name and line count collapse into one line instead of two.
    pending_explore_label: Option<String>,
}

impl PlainUi {
    pub fn new() -> Self {
        Self {
            progress: None,
            pending_explore_label: None,
        }
    }

    /// A PlainUi that clears a REPL "working…" spinner on its first output.
    pub fn with_progress(progress: Arc<AtomicBool>) -> Self {
        Self {
            progress: Some(progress),
            pending_explore_label: None,
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
        // Exploration tools defer their header until the result lands, so the
        // file name and line count share one line instead of two.
        if matches!(name, "read" | "list" | "grep") {
            self.pending_explore_label = Some(tool_label(name, arguments));
            return;
        }
        println!("\x1b[36m⏺ {}\x1b[0m", tool_label(name, arguments));
    }

    fn tool_stream(&mut self, _name: &str, line: &str) {
        // Live bash output: print dimmed so it's distinguishable from the
        // final result. No begin_output — the spinner was already cleared
        // by the tool_call header.
        println!("\x1b[2m  │ {line}\x1b[0m");
    }

    fn confirm_edit(&mut self, path: &str, diff: &str) -> bool {
        use std::io::Write;
        self.begin_output();
        println!("\x1b[33m⏺ edit {} — apply? [y/N]\x1b[0m", path);
        if !diff.is_empty() {
            for line in diff.lines().take(20) {
                println!("\x1b[2m  {line}\x1b[0m");
            }
        }
        print!("\x1b[33m  › \x1b[0m");
        let _ = std::io::stdout().flush();
        let mut input = String::new();
        let _ = std::io::stdin().read_line(&mut input);
        input.trim().eq_ignore_ascii_case("y")
    }

    fn tool_result(&mut self, name: &str, result: &str) {
        // Read-only exploration tools (read/list/grep) collapse the header and
        // the line count into one line: `⏺ read path/to/file · 113 lines`.
        if matches!(name, "read" | "list" | "grep") {
            let n = result.lines().count();
            let header = self
                .pending_explore_label
                .take()
                .unwrap_or_else(|| name.to_string());
            let suffix = if n == 0 {
                "(no output)".to_string()
            } else {
                format!("{n} line{}", if n == 1 { "" } else { "s" })
            };
            self.begin_output();
            println!("\x1b[36m⏺ {header} · {suffix}\x1b[0m");
            return;
        }
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

    fn subagent_note(&mut self, text: &str) {
        self.begin_output();
        // Bold magenta so a subagent delegation stands out from the cyan tool
        // lines and blue status notes around it.
        println!("\x1b[1;35m{text}\x1b[0m");
    }

    fn plan(&mut self, steps: &[hi_agent::PlanStep]) {
        use hi_agent::PlanStatus;
        self.begin_output();
        let done = steps
            .iter()
            .filter(|s| s.status == PlanStatus::Done)
            .count();
        let active = steps.iter().find(|s| s.status == PlanStatus::Active);
        // Compact one-line summary with the active step, so a reader scanning
        // the REPL output sees progress at a glance without reading every line.
        if let Some(a) = active {
            println!(
                "\x1b[1m⏺ plan · {done}/{} · ▸ {}\x1b[0m",
                steps.len(),
                a.title
            );
        } else {
            println!("\x1b[1m⏺ plan · {done}/{}\x1b[0m", steps.len());
        }
        // Show up to 8 steps; clip long plans so they don't flood the REPL.
        const MAX_SHOWN: usize = 8;
        for s in steps.iter().take(MAX_SHOWN) {
            let (glyph, color) = match s.status {
                PlanStatus::Done => ('✓', "\x1b[32m"),
                PlanStatus::Active => ('▸', "\x1b[36m"),
                PlanStatus::Pending => ('☐', "\x1b[2m"),
            };
            println!("{color}  {glyph} {}\x1b[0m", s.title);
        }
        if steps.len() > MAX_SHOWN {
            println!("\x1b[2m  … {} more steps\x1b[0m", steps.len() - MAX_SHOWN);
        }
    }

    fn turn_end(&mut self, summary: &str) {
        self.begin_output();
        println!("\x1b[2m{summary}\x1b[0m");
    }

    fn turn_error(&mut self, kind: &str, message: &str, guidance: &str) {
        self.begin_output();
        let suffix = if guidance.is_empty() {
            String::new()
        } else {
            format!(" — {guidance}")
        };
        eprintln!("\x1b[31m{kind}: {message}{suffix}\x1b[0m");
    }

    fn changed_files(&mut self, files: &[String]) {
        self.begin_output();
        let label = if files.len() == 1 { "file" } else { "files" };
        println!(
            "\x1b[32m  ✎ {} {} changed: {}\x1b[0m",
            files.len(),
            label,
            files.join(", ")
        );
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
    fn tool_result(&mut self, _name: &str, _result: &str) {}
    fn status(&mut self, _text: &str) {}
    fn turn_end(&mut self, _summary: &str) {}
}
