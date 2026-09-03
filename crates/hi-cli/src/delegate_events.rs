//! Live progress pipe for a `delegate` child: JSONL writer + parent file tailer.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use hi_agent::{
    AskUserFuture, ConfirmationFuture, ConfirmationRequest, DelegateChildEvent, DelegateProgress,
    Ui, dispatch_delegate_child_event, parse_delegate_child_event,
};

const MAX_EVENT_FIELD_CHARS: usize = 2_000;
const TAIL_POLL_MS: u64 = 40;

/// Wraps a child `Ui` and appends compact JSONL events the parent can tail.
pub(crate) struct JsonlProgressUi {
    inner: Box<dyn Ui>,
    file: Mutex<BufWriter<File>>,
}

impl JsonlProgressUi {
    #[cfg(test)]
    pub(crate) fn open(inner: Box<dyn Ui>, path: &Path) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            inner,
            file: Mutex::new(BufWriter::new(file)),
        })
    }

    fn emit(&self, event: &DelegateChildEvent) {
        let Ok(mut file) = self.file.lock() else {
            return;
        };
        let Ok(line) = serde_json::to_string(event) else {
            return;
        };
        let _ = writeln!(file, "{line}");
        let _ = file.flush();
    }
}

fn clip_field(text: &str) -> String {
    text.chars().take(MAX_EVENT_FIELD_CHARS).collect()
}

/// Wrap `inner` so a parent-supplied `--events-jsonl` path receives live events.
/// File-open failure keeps the inner UI (the row stays on `"running"`).
pub(crate) fn wrap_event_ui(inner: Box<dyn Ui>, events_jsonl: Option<&Path>) -> Box<dyn Ui> {
    let Some(path) = events_jsonl else {
        return inner;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let file = match OpenOptions::new().create(true).append(true).open(path) {
        Ok(file) => file,
        Err(_) => return inner,
    };
    Box::new(JsonlProgressUi {
        inner,
        file: Mutex::new(BufWriter::new(file)),
    })
}

impl Ui for JsonlProgressUi {
    fn assistant_text(&mut self, text: &str) {
        self.inner.assistant_text(text);
    }
    fn assistant_reasoning(&mut self, text: &str) {
        self.emit(&DelegateChildEvent::Reasoning);
        self.inner.assistant_reasoning(text);
    }
    fn assistant_end(&mut self) {
        self.inner.assistant_end();
    }
    fn tool_started(&mut self, name: &str, arguments: &str) {
        self.emit(&DelegateChildEvent::ToolStarted {
            name: name.to_string(),
            arguments: clip_field(arguments),
        });
        self.inner.tool_started(name, arguments);
    }
    fn tool_stream(&mut self, name: &str, line: &str) {
        let clipped = clip_field(line);
        if !clipped.trim().is_empty() {
            self.emit(&DelegateChildEvent::ToolStream { line: clipped });
        }
        self.inner.tool_stream(name, line);
    }
    fn tool_call(&mut self, name: &str, arguments: &str) {
        self.inner.tool_call(name, arguments);
    }
    fn tool_result(&mut self, name: &str, result: &str) {
        self.emit(&DelegateChildEvent::ToolResult {
            name: name.to_string(),
            result: clip_field(result),
        });
        self.inner.tool_result(name, result);
    }
    fn plan_result_id(
        &mut self,
        id: &str,
        name: &str,
        result: &str,
        status: hi_tools::ToolStatus,
        steps: &[hi_agent::PlanStep],
    ) {
        self.inner.plan_result_id(id, name, result, status, steps);
    }
    fn confirm(&mut self, request: ConfirmationRequest) -> ConfirmationFuture<'_> {
        self.inner.confirm(request)
    }
    fn ask_user(&mut self, question: &str, options: &[String]) -> AskUserFuture<'_> {
        self.inner.ask_user(question, options)
    }
    fn status(&mut self, text: &str) {
        self.inner.status(text);
    }
    fn subagent_note(&mut self, text: &str) {
        self.inner.subagent_note(text);
    }
    fn plan(&mut self, steps: &[hi_agent::PlanStep]) {
        self.inner.plan(steps);
    }
    fn usage(
        &mut self,
        prompt_tokens: u64,
        generated_tokens: u64,
        context_used: u64,
        context_window: Option<u32>,
        usage_estimated: bool,
    ) {
        self.inner.usage(
            prompt_tokens,
            generated_tokens,
            context_used,
            context_window,
            usage_estimated,
        );
    }
    fn session_usage(&mut self, usage: &hi_ai::Usage) {
        self.inner.session_usage(usage);
    }
    fn turn_end(&mut self, summary: &str) {
        self.inner.turn_end(summary);
    }
    fn changed_files(&mut self, files: &[String]) {
        self.inner.changed_files(files);
    }
    fn suggested_prompt(&mut self, text: &str) {
        self.inner.suggested_prompt(text);
    }
    fn turn_error(&mut self, kind: &str, message: &str, guidance: &str) {
        self.inner.turn_error(kind, message, guidance);
    }
    fn nudge(&mut self, text: &str) {
        self.inner.nudge(text);
    }
}

/// Stops the tailer thread and drains remaining complete lines.
pub(crate) struct EventTailer {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl EventTailer {
    pub(crate) fn finish(self) {
        drop(self);
    }
}

impl Drop for EventTailer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

pub(crate) fn start_event_tailer(
    path: PathBuf,
    progress: Arc<dyn DelegateProgress>,
) -> Option<EventTailer> {
    let stop = Arc::new(AtomicBool::new(false));
    let stop_thread = stop.clone();
    let handle = std::thread::Builder::new()
        .name("hi-delegate-events".into())
        .spawn(move || tail_delegate_events(&path, progress, stop_thread))
        .ok()?;
    Some(EventTailer {
        stop,
        handle: Some(handle),
    })
}

pub(crate) fn tail_delegate_events(
    path: &Path,
    progress: Arc<dyn DelegateProgress>,
    stop: Arc<AtomicBool>,
) {
    let mut leftover = String::new();
    let mut file: Option<File> = None;
    loop {
        if file.is_none() {
            file = File::open(path).ok();
        }
        if let Some(open) = file.as_mut() {
            drain_delegate_event_file(open, &mut leftover, progress.as_ref());
        }
        if stop.load(Ordering::Acquire) {
            if let Some(open) = file.as_mut() {
                drain_delegate_event_file(open, &mut leftover, progress.as_ref());
            }
            break;
        }
        std::thread::sleep(Duration::from_millis(TAIL_POLL_MS));
    }
}

fn drain_delegate_event_file(
    file: &mut File,
    leftover: &mut String,
    progress: &dyn DelegateProgress,
) {
    let mut buf = Vec::new();
    if file.read_to_end(&mut buf).is_err() || buf.is_empty() {
        return;
    }
    leftover.push_str(&String::from_utf8_lossy(&buf));
    while let Some(idx) = leftover.find('\n') {
        let line = leftover[..idx].to_string();
        leftover.replace_range(..=idx, "");
        if let Some(event) = parse_delegate_child_event(&line) {
            dispatch_delegate_child_event(progress, &event);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    use crate::ui::QuietUi;

    struct RecProgress(Mutex<Vec<(String, Option<String>)>>);

    impl DelegateProgress for RecProgress {
        fn progress(&self, activity: &str, line: Option<&str>) {
            self.0
                .lock()
                .unwrap()
                .push((activity.to_string(), line.map(str::to_string)));
        }
    }

    fn temp_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "hi-delegate-events-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ))
    }

    fn wait_until(pred: impl Fn() -> bool) {
        let started = Instant::now();
        while !pred() {
            assert!(
                started.elapsed() < Duration::from_secs(2),
                "timed out waiting for delegate event tailer"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn jsonl_progress_ui_writes_tool_started() {
        let path = temp_path("write");
        let mut ui = JsonlProgressUi::open(Box::new(QuietUi), &path).unwrap();
        ui.tool_started("read", r#"{"path":"lib.rs"}"#);
        ui.assistant_reasoning("hmm");
        drop(ui);
        let text = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let lines: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 2, "got {text}");
        let started = parse_delegate_child_event(lines[0]).unwrap();
        assert!(matches!(
            started,
            DelegateChildEvent::ToolStarted { ref name, .. } if name == "read"
        ));
        assert_eq!(
            parse_delegate_child_event(lines[1]).unwrap(),
            DelegateChildEvent::Reasoning
        );
    }

    #[test]
    fn tailer_reads_incremental_jsonl() {
        let path = temp_path("tail");
        std::fs::write(&path, []).unwrap();
        let rec = Arc::new(RecProgress(Mutex::new(Vec::new())));
        let rec_progress: Arc<dyn DelegateProgress> = rec.clone();
        let tailer = start_event_tailer(path.clone(), rec_progress).expect("spawn tailer");
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        writeln!(
            file,
            r#"{{"kind":"tool_started","name":"read","arguments":"{{\"path\":\"lib.rs\"}}"}}"#
        )
        .unwrap();
        file.flush().unwrap();
        wait_until(|| {
            rec.0
                .lock()
                .unwrap()
                .iter()
                .any(|(a, _)| a == "Reading lib.rs")
        });
        writeln!(file, r#"{{"kind":"tool_stream","line":"fn foo() {{}}"}}"#).unwrap();
        file.flush().unwrap();
        wait_until(|| {
            rec.0
                .lock()
                .unwrap()
                .iter()
                .any(|(a, line)| a.is_empty() && line.as_deref() == Some("fn foo() {}"))
        });
        tailer.finish();
        let _ = std::fs::remove_file(&path);
        let got = rec.0.lock().unwrap().clone();
        assert_eq!(got[0].0, "Reading lib.rs");
        assert_eq!(got.last().unwrap().1.as_deref(), Some("fn foo() {}"));
    }
}
