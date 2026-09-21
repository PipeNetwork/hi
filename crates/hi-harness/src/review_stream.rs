//! Hides the `<review>…</review>` verdict block from streamed assistant text
//! in review-drive audit turns.
//!
//! The drive parses the block from the completion and renders it once as a
//! report. Without this the user saw the raw `coverage: … | … | …` rows in
//! the transcript and then the same rows again as the report (a live audit
//! replied with nothing but the block). The filter only touches what the
//! `Ui` sees; the model-facing message keeps the block.

use crate::ui::Ui;

const OPEN: &str = "<review>";
const CLOSE: &str = "</review>";

/// Per-round text filter: pass prose through, hold back a verdict block.
pub(crate) struct VerdictBlockFilter {
    enabled: bool,
    /// Not yet forwarded: a possible partial `<review>` tag while scanning,
    /// the whole block once one has opened.
    pending: String,
    withholding: bool,
}

impl VerdictBlockFilter {
    pub(crate) fn new(enabled: bool) -> Self {
        Self {
            enabled,
            pending: String::new(),
            withholding: false,
        }
    }

    /// Forward `text`, minus a verdict block once one opens. Text around
    /// the block is passed verbatim; frontends already trim a message.
    pub(crate) fn text(&mut self, text: &str, ui: &mut dyn Ui) {
        if !self.enabled {
            ui.assistant_text(text);
            return;
        }
        self.pending.push_str(text);
        if self.withholding {
            return;
        }
        if let Some(open) = self.pending.find(OPEN) {
            let before: String = self.pending.drain(..open).collect();
            if !before.is_empty() {
                ui.assistant_text(&before);
            }
            self.withholding = true;
            return;
        }
        let keep = partial_tag_suffix_len(&self.pending);
        let forward_to = self.pending.len() - keep;
        if forward_to > 0 {
            let out: String = self.pending.drain(..forward_to).collect();
            ui.assistant_text(&out);
        }
    }

    /// End of the model's reply. A closed block is dropped and any text
    /// after it forwarded; an unclosed one (truncated output) is shown as
    /// is, so nothing the model wrote silently disappears.
    pub(crate) fn finish(&mut self, ui: &mut dyn Ui) {
        if !self.enabled {
            return;
        }
        let pending = std::mem::take(&mut self.pending);
        let out = if self.withholding {
            match pending.find(CLOSE) {
                Some(close) => pending[close + CLOSE.len()..].to_string(),
                None => pending,
            }
        } else {
            pending
        };
        self.withholding = false;
        if !out.is_empty() {
            ui.assistant_text(&out);
        }
    }
}

/// Length of the longest suffix of `text` that is a proper prefix of
/// `<review>`: a tag split across chunks (`"…<rev"`, `"iew>…"`) must not
/// leak its first half.
fn partial_tag_suffix_len(text: &str) -> usize {
    (1..OPEN.len())
        .rev()
        .find(|&len| text.ends_with(&OPEN[..len]))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::TestUi;

    fn stream(enabled: bool, chunks: &[&str]) -> String {
        let mut ui = TestUi::default();
        let mut filter = VerdictBlockFilter::new(enabled);
        for chunk in chunks {
            filter.text(chunk, &mut ui);
        }
        filter.finish(&mut ui);
        ui.texts.join("")
    }

    const BLOCK: &str = "<review>\nverdict: COMPLETE\nfinding: none\n</review>";

    #[test]
    fn prose_passes_and_the_block_is_dropped() {
        let shown = stream(true, &["Read three files.\n\n", BLOCK]);
        assert_eq!(shown, "Read three files.\n\n");
        assert_eq!(stream(true, &[BLOCK]), "", "block-only reply shows nothing");
        assert_eq!(
            stream(true, &["Before.\n", BLOCK, "\nAfter."]),
            "Before.\n\nAfter.",
            "text after the block still shows"
        );
    }

    #[test]
    fn a_tag_split_across_chunks_does_not_leak() {
        let shown = stream(
            true,
            &["Summary.\n<rev", "iew>\nverdict: COMPLETE\n</rev", "iew>"],
        );
        assert_eq!(shown, "Summary.\n");
        assert_eq!(
            stream(true, &["a < b and <r", "eally> not a tag"]),
            "a < b and <really> not a tag",
            "a held-back `<r` is released once it is not the tag"
        );
    }

    #[test]
    fn an_unclosed_block_is_shown_and_disabled_passes_everything() {
        let truncated = "<review>\nverdict: INCOMP";
        assert_eq!(
            stream(true, &["Partial.\n", truncated]),
            format!("Partial.\n{truncated}")
        );
        assert_eq!(stream(false, &["x ", BLOCK]), format!("x {BLOCK}"));
    }

    #[test]
    fn partial_suffix_lengths() {
        assert_eq!(partial_tag_suffix_len("abc<"), 1);
        assert_eq!(partial_tag_suffix_len("abc<revie"), 6);
        assert_eq!(
            partial_tag_suffix_len("abc<review>"),
            0,
            "a full tag is not partial"
        );
        assert_eq!(partial_tag_suffix_len("abc"), 0);
    }
}
