//! Fold superseded observations while retaining distinct source and process evidence.

use super::{Content, ReadCallKey, Transcript, background_poll_handle, read_call_key};

impl Transcript {
    /// Shorten old idle status lines. Background polls return incremental
    /// output: a later poll cannot replace a previous compiler error or test
    /// failure. Every poll with actual output remains intact.
    pub(crate) fn fold_superseded_background_polls(&mut self, handle: &str) {
        self.fold_superseded_tool_results(
            "[no new output]",
            |name, arguments| {
                name == "bash_output"
                    && background_poll_handle(arguments).as_deref() == Some(handle)
            },
            |output| {
                let output = output.trim();
                !output.contains('\n')
                    && output.starts_with(&format!("[{handle} · "))
                    && output.ends_with(": still running — no new output]")
            },
        );
    }

    /// Shrink every superseded `read` result with the same call shape
    /// (path + offset + limit) to a one-line digest, keeping only the newest
    /// verbatim. Edit-heavy sessions re-read the same file after each change
    /// (one real session read `src/model.rs` 21×), and only the newest copy
    /// reflects the file — the older ones are pure context bloat.
    pub(crate) fn fold_superseded_file_reads(&mut self, key: &ReadCallKey) {
        let target = if key.paths.len() == 1 {
            key.paths[0].clone()
        } else {
            format!("{} files ({})", key.paths.len(), key.paths.join(", "))
        };
        let digest = format!(
            "[superseded read of {} — see the latest read result]",
            target
        );
        self.fold_superseded_tool_results(
            &digest,
            |name, arguments| name == "read" && read_call_key(arguments).as_ref() == Some(key),
            |_| true,
        );
    }

    /// Shared folding walk: find every ToolCall matching `matches`, then
    /// rewrite the ToolResults of all but the last one to `digest`.
    fn fold_superseded_tool_results(
        &mut self,
        digest: &str,
        matches: impl Fn(&str, &str) -> bool,
        may_fold: impl Fn(&str) -> bool,
    ) {
        let mut matching_call_ids: Vec<String> = Vec::new();
        for message in self.messages.iter() {
            for block in &message.content {
                if let Content::ToolCall {
                    id,
                    name,
                    arguments,
                } = block
                    && matches(name, arguments)
                {
                    matching_call_ids.push(id.clone());
                }
            }
        }
        if matching_call_ids.len() < 2 {
            return;
        }
        let superseded: std::collections::HashSet<&str> = matching_call_ids
            [..matching_call_ids.len() - 1]
            .iter()
            .map(String::as_str)
            .collect();
        // Skip the rewrite (and the copy-on-write clone) when every superseded
        // result is already folded — the common case after the first fold.
        let already_folded = self
            .messages
            .iter()
            .flat_map(|message| &message.content)
            .all(|block| match block {
                Content::ToolResult { call_id, output }
                    if superseded.contains(call_id.as_str()) && may_fold(output) =>
                {
                    output == digest
                }
                _ => true,
            });
        if already_folded {
            return;
        }
        for message in self.make_mut().iter_mut() {
            for block in &mut message.content {
                if let Content::ToolResult { call_id, output } = block
                    && superseded.contains(call_id.as_str())
                    && may_fold(output)
                {
                    *output = digest.to_string();
                }
            }
        }
    }
}
