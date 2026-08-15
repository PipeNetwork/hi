//! Child-agent UI that maps tools/reasoning to live activity labels.
//!
//! Used by explore/delegate/task so parent transcripts get a typed subagent
//! row instead of a dump of `explore:read` / `explore:grep` calls.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::ui::{SubagentSink, Ui, subagent_activity_label};

/// One JSONL record a `delegate` child writes for the parent tailer.
/// Tagged `kind` (same constraint as other event enums: the tag occupies
/// `kind`, so variants must not also have a `kind` field).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DelegateChildEvent {
    ToolStarted { name: String, arguments: String },
    ToolStream { line: String },
    ToolResult { name: String, result: String },
    Reasoning,
}

/// Parse one child JSONL line. Malformed or empty lines are ignored.
pub fn parse_delegate_child_event(line: &str) -> Option<DelegateChildEvent> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    serde_json::from_str(line).ok()
}

/// Map a child event onto [`crate::DelegateProgress`]. Empty `activity` with a
/// line updates the inspect overlay without changing the live row suffix.
pub fn dispatch_delegate_child_event(
    progress: &dyn crate::DelegateProgress,
    event: &DelegateChildEvent,
) {
    match event {
        DelegateChildEvent::ToolStarted { name, arguments } => {
            let activity = subagent_activity_label(name, arguments);
            progress.progress(&activity, Some(&activity));
        }
        DelegateChildEvent::ToolStream { line } => {
            let line = line.trim();
            if !line.is_empty() {
                progress.progress("", Some(line));
            }
        }
        DelegateChildEvent::ToolResult { result, .. } => {
            let clip: String = result.chars().take(240).collect();
            if !clip.trim().is_empty() {
                progress.progress("", Some(&clip));
            }
        }
        DelegateChildEvent::Reasoning => progress.progress("Thinking", None),
    }
}

/// Forwards [`crate::DelegateProgress`] into a parent [`SubagentSink`] under a
/// fixed child id (`delegate-{slot}`).
pub(crate) struct SinkDelegateProgress {
    pub(crate) id: String,
    pub(crate) sink: Arc<dyn SubagentSink>,
}

impl crate::DelegateProgress for SinkDelegateProgress {
    fn progress(&self, activity: &str, line: Option<&str>) {
        self.sink.progress(&self.id, activity, line);
    }
}

pub(crate) fn bound_sink_progress(
    ui: &dyn Ui,
    id: &str,
) -> Option<Arc<dyn crate::DelegateProgress>> {
    ui.subagent_sink().map(|sink| {
        Arc::new(SinkDelegateProgress {
            id: id.to_string(),
            sink,
        }) as Arc<dyn crate::DelegateProgress>
    })
}

/// Maps a child's `Ui` events onto [`SubagentSink`]. `Send` so parallel explore
/// jobs and background `task` workers can hold it.
pub(crate) struct SubagentProgressUi {
    pub(crate) id: String,
    pub(crate) sink: Option<Arc<dyn SubagentSink>>,
}

impl SubagentProgressUi {
    fn progress(&self, activity: &str) {
        if let Some(sink) = &self.sink {
            sink.progress(&self.id, activity, None);
        }
    }

    fn line(&self, activity: &str, line: &str) {
        let line = line.trim();
        if line.is_empty() {
            return;
        }
        if let Some(sink) = &self.sink {
            sink.progress(&self.id, activity, Some(line));
        }
    }
}

impl Ui for SubagentProgressUi {
    fn assistant_text(&mut self, text: &str) {
        self.line("", text);
    }

    fn assistant_reasoning(&mut self, _text: &str) {
        self.progress("Thinking");
    }

    fn assistant_end(&mut self) {}

    fn tool_started(&mut self, name: &str, arguments: &str) {
        let activity = subagent_activity_label(name, arguments);
        self.progress(&activity);
        self.line(&activity, &activity);
    }

    fn tool_call(&mut self, name: &str, arguments: &str) {
        self.tool_started(name, arguments);
    }

    fn tool_result(&mut self, _name: &str, result: &str) {
        let clip: String = result.chars().take(240).collect();
        self.line("", &clip);
    }

    fn tool_stream(&mut self, _name: &str, line: &str) {
        self.line("", line);
    }

    fn status(&mut self, _text: &str) {}

    fn turn_end(&mut self, _summary: &str) {}
}

/// Serial-path wrapper: live labels plus parent `tool_result` for RecUi tests.
pub(crate) struct SubagentParentUi<'a> {
    pub(crate) inner: SubagentProgressUi,
    pub(crate) parent: &'a mut dyn Ui,
}

impl Ui for SubagentParentUi<'_> {
    fn assistant_text(&mut self, text: &str) {
        self.inner.assistant_text(text);
        self.parent.assistant_text(text);
    }

    fn assistant_reasoning(&mut self, text: &str) {
        self.inner.assistant_reasoning(text);
        self.parent.subagent_progress(&self.inner.id, "Thinking");
    }

    fn assistant_end(&mut self) {}

    fn tool_started(&mut self, name: &str, arguments: &str) {
        self.inner.tool_started(name, arguments);
        let activity = subagent_activity_label(name, arguments);
        self.parent.subagent_progress(&self.inner.id, &activity);
    }

    fn tool_call(&mut self, name: &str, arguments: &str) {
        self.tool_started(name, arguments);
    }

    fn tool_result(&mut self, name: &str, result: &str) {
        self.inner.tool_result(name, result);
        self.parent.tool_result(name, result);
    }

    fn status(&mut self, _text: &str) {}

    fn turn_end(&mut self, _summary: &str) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct RecProgress(Mutex<Vec<(String, Option<String>)>>);

    impl crate::DelegateProgress for RecProgress {
        fn progress(&self, activity: &str, line: Option<&str>) {
            self.0
                .lock()
                .unwrap()
                .push((activity.to_string(), line.map(str::to_string)));
        }
    }

    struct RecSink(Mutex<Vec<(String, String, Option<String>)>>);

    impl SubagentSink for RecSink {
        fn spawned(&self, _id: &str, _kind: &str, _description: &str, _background: bool) {}
        fn progress(&self, id: &str, activity: &str, line: Option<&str>) {
            self.0.lock().unwrap().push((
                id.to_string(),
                activity.to_string(),
                line.map(str::to_string),
            ));
        }
        fn finished(&self, _id: &str, _status: &str, _elapsed_ms: u64, _summary: &str) {}
    }

    #[test]
    fn parse_and_dispatch_maps_tool_started_to_activity_label() {
        let progress = RecProgress(Mutex::new(Vec::new()));
        let event = parse_delegate_child_event(
            r#"{"kind":"tool_started","name":"read","arguments":"{\"path\":\"lib.rs\"}"}"#,
        )
        .expect("parse tool_started");
        dispatch_delegate_child_event(&progress, &event);
        let got = progress.0.lock().unwrap().clone();
        assert_eq!(
            got,
            vec![("Reading lib.rs".into(), Some("Reading lib.rs".into()))]
        );
    }

    #[test]
    fn parse_and_dispatch_reasoning_and_inspect_line() {
        let progress = RecProgress(Mutex::new(Vec::new()));
        dispatch_delegate_child_event(
            &progress,
            &parse_delegate_child_event(r#"{"kind":"reasoning"}"#).unwrap(),
        );
        dispatch_delegate_child_event(
            &progress,
            &parse_delegate_child_event(r#"{"kind":"tool_stream","line":"compiling hi-agent"}"#)
                .unwrap(),
        );
        let got = progress.0.lock().unwrap().clone();
        assert_eq!(got[0], ("Thinking".into(), None));
        assert_eq!(
            got[1],
            (String::new(), Some("compiling hi-agent".into())),
            "stream lines must not replace the live suffix"
        );
    }

    #[test]
    fn bound_sink_forwards_under_delegate_id() {
        let sink = Arc::new(RecSink(Mutex::new(Vec::new())));
        let progress = SinkDelegateProgress {
            id: "delegate-3".into(),
            sink: sink.clone(),
        };
        let event = DelegateChildEvent::ToolStarted {
            name: "bash".into(),
            arguments: r#"{"command":"cargo test"}"#.into(),
        };
        dispatch_delegate_child_event(&progress, &event);
        let got = sink.0.lock().unwrap().clone();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, "delegate-3");
        assert!(got[0].1.starts_with("Run "), "got {}", got[0].1);
    }

    #[test]
    fn malformed_jsonl_is_ignored() {
        assert!(parse_delegate_child_event("").is_none());
        assert!(parse_delegate_child_event("{not json}").is_none());
        assert!(parse_delegate_child_event(r#"{"kind":"nope"}"#).is_none());
    }
}
