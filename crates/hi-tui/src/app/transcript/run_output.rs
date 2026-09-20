//! Live shell-result aggregation and background-poll placeholders.

use crate::TranscriptEntry;
use crate::activity_feed::ActivityKind;

impl crate::App {
    pub(super) fn apply_run_result(&mut self, command: &str, display: &str, live: bool) {
        self.freeze_verb_group();
        let chunk = strip_bg_status_lines(display);
        if live {
            if !self.run_streamed_this_call && !chunk.trim().is_empty() {
                self.append_run_output_for(command, &chunk);
            }
            self.touch_idle_run(command);
            return;
        }
        let reuse = matches!(
            self.transcript.last(),
            Some(TranscriptEntry::Activity(block))
                if matches!(
                    &block.kind,
                    ActivityKind::Run {
                        command: existing,
                        idle,
                        ..
                    } if (*idle || existing == command) && !(!*idle && existing == "grep" && command == "grep")
                )
        );
        if reuse
            && let Some(TranscriptEntry::Activity(block)) = self.transcript.last_mut()
            && let Some((_, dest, idle, _)) = block.as_run_mut()
        {
            if !self.run_streamed_this_call {
                dest.clear();
                dest.push_str(&chunk);
            }
            *idle = false;
            self.bump_transcript();
            return;
        }
        self.push_activity(ActivityKind::Run {
            command: command.to_string(),
            body: chunk,
            idle: false,
            poll_count: 0,
        });
    }

    pub(super) fn append_live_run_output(&mut self, line: &str) -> bool {
        for entry in self.transcript.iter_mut().rev() {
            if let TranscriptEntry::Activity(block) = entry
                && let Some((_, body, idle, _)) = block.as_run_mut()
                && *idle
            {
                append_capped_run_body(body, line);
                self.bump_transcript();
                return true;
            }
        }
        false
    }

    pub(super) fn append_run_output_for(&mut self, command: &str, chunk: &str) {
        for entry in self.transcript.iter_mut().rev() {
            if let TranscriptEntry::Activity(block) = entry
                && let Some((existing, body, idle, _)) = block.as_run_mut()
                && (existing == command || *idle)
            {
                append_capped_run_body(body, chunk);
                *idle = true;
                self.bump_transcript();
                return;
            }
        }
        let mut body = String::new();
        append_capped_run_body(&mut body, chunk);
        self.push_activity(ActivityKind::Run {
            command: command.to_string(),
            body,
            idle: true,
            poll_count: 0,
        });
    }

    pub(super) fn idle_run_count_and_match(&self, command: &str) -> (usize, bool) {
        let mut count = 0;
        let mut matching = false;
        for entry in &self.transcript {
            if let TranscriptEntry::Activity(block) = entry
                && let ActivityKind::Run {
                    command: existing,
                    idle: true,
                    ..
                } = &block.kind
            {
                count += 1;
                if existing == command {
                    matching = true;
                }
            }
        }
        (count, matching)
    }

    pub(super) fn touch_idle_run(&mut self, command: &str) {
        let (count, matching) = self.idle_run_count_and_match(command);
        if matching || count == 1 {
            for entry in self.transcript.iter_mut().rev() {
                if let TranscriptEntry::Activity(block) = entry
                    && let Some((existing, _, idle, poll_count)) = block.as_run_mut()
                    && *idle
                    && (existing == command || count == 1)
                {
                    *poll_count = poll_count.saturating_add(1);
                    self.bump_transcript();
                    return;
                }
            }
        }
        self.push_activity(ActivityKind::Run {
            command: command.to_string(),
            body: String::new(),
            idle: true,
            poll_count: 1,
        });
    }

    pub(super) fn note_idle_bash_poll(&mut self, id: &str) {
        self.freeze_verb_group();
        self.touch_idle_run(id);
    }

    pub(super) fn pop_idle_run(&mut self, command: &str) {
        let drop_last = matches!(
            self.transcript.last(),
            Some(TranscriptEntry::Activity(block))
                if matches!(
                    &block.kind,
                    ActivityKind::Run {
                        command: existing,
                        idle: true,
                        ..
                    } if existing == command
                )
        );
        if drop_last {
            self.transcript.pop();
            self.bump_transcript();
        }
    }

    pub(super) fn ensure_run_placeholder(&mut self, command: &str) {
        let (count, matching) = self.idle_run_count_and_match(command);
        if matching || count == 1 {
            return;
        }
        self.push_activity(ActivityKind::Run {
            command: command.to_string(),
            body: String::new(),
            idle: true,
            poll_count: 0,
        });
    }
}

pub(super) fn bash_process_live(result: &str) -> bool {
    result.lines().next().is_some_and(|status| {
        status.contains("still running")
            || status.contains("continued as")
            || (status.starts_with("Started ") && status.contains('('))
    })
}

fn strip_bg_status_lines(text: &str) -> String {
    text.lines()
        .filter(|line| {
            let trimmed = line.trim();
            !(trimmed.starts_with('[')
                && trimmed.ends_with(']')
                && (trimmed.contains("still running")
                    || trimmed.contains("exited")
                    || trimmed.contains("stopped")
                    || trimmed.contains(": failed")))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

const LIVE_RUN_BODY_MAX: usize = 64 * 1024;

fn append_capped_run_body(body: &mut String, chunk: &str) {
    if chunk.is_empty() {
        return;
    }
    if !body.is_empty() && !body.ends_with('\n') && !chunk.starts_with('\n') {
        body.push('\n');
    }
    body.push_str(chunk);
    if !body.ends_with('\n') {
        body.push('\n');
    }
    if body.len() <= LIVE_RUN_BODY_MAX {
        return;
    }
    // `overflow` is a byte index into a live bash transcript. Box-drawing and
    // other multibyte UTF-8 (a 64k cap lands inside `─` easily) must not be
    // sliced at a non-boundary — that panic aborted a supervised wrap with
    // `turn_unclosed` and no auto-repair while Sentinel was job-stopped.
    let overflow = floor_char_boundary(body, body.len() - LIVE_RUN_BODY_MAX);
    let cut = body
        .get(overflow..)
        .and_then(|tail| tail.find('\n'))
        .map(|i| overflow + i + 1)
        .unwrap_or(overflow)
        .min(body.len());
    let cut = floor_char_boundary(body, cut);
    body.replace_range(..cut, "");
}

fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

pub(super) fn bash_output_is_idle(result: &str) -> bool {
    result.lines().next().is_some_and(|status| {
        status.contains("still running — no new output")
            || status.contains("running — no new output")
    })
}

pub(super) fn is_missing_background_process_result(result: &str) -> bool {
    result
        .trim_start()
        .starts_with("Error: no background process")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capping_live_run_body_does_not_panic_on_multibyte_at_cut() {
        // One ASCII byte plus 3-byte `─` so a 64KiB cap lands inside a char.
        let mut body = String::from("x");
        body.push_str(&"─".repeat(22_000));
        assert!(body.len() > LIVE_RUN_BODY_MAX);
        append_capped_run_body(&mut body, "more ─ output");
        assert!(body.len() <= LIVE_RUN_BODY_MAX + 32);
        assert!(std::str::from_utf8(body.as_bytes()).is_ok());
        let _ = body.chars().count();
    }
}
