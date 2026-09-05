//! Fold superseded observations while retaining distinct source and process evidence.

use super::{Content, ReadCallKey, Role, Transcript, background_poll_handle, read_call_key};

#[cfg(test)]
mod tests;

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

    /// Pair each result with its assistant round before deciding whether it
    /// was superseded. Providers may reuse call IDs in subsequent rounds.
    fn fold_superseded_tool_results(
        &mut self,
        digest: &str,
        matches: impl Fn(&str, &str) -> bool,
        may_fold: impl Fn(&str) -> bool,
    ) {
        let mut matching_results = Vec::new();
        for (message_index, message) in self.messages.iter().enumerate() {
            if message.role != Role::Assistant {
                continue;
            }
            let mut results_by_id =
                std::collections::HashMap::<_, std::collections::VecDeque<_>>::new();
            for (result_index, result_message) in self
                .messages
                .iter()
                .enumerate()
                .skip(message_index + 1)
                .take_while(|(_, m)| m.role == Role::Tool)
            {
                for (block_index, block) in result_message.content.iter().enumerate() {
                    if let Content::ToolResult { call_id, .. } = block {
                        results_by_id
                            .entry(call_id.as_str())
                            .or_default()
                            .push_back((result_index, block_index));
                    }
                }
            }
            // Consume in call order, including nonmatching calls. Legacy
            // batches with duplicate IDs pair by occurrence, just as append
            // and provider validation do; IDs alone are never global identity.
            for block in &message.content {
                if let Content::ToolCall {
                    id,
                    name,
                    arguments,
                } = block
                    && let Some(position) = results_by_id
                        .get_mut(id.as_str())
                        .and_then(|positions| positions.pop_front())
                    && matches(name, arguments)
                {
                    matching_results.push(position);
                }
            }
        }
        if matching_results.len() < 2 {
            return;
        }
        matching_results.pop();
        let replacements = matching_results.into_iter().filter(|&(message_index, block_index)| {
            matches!(&self.messages[message_index].content[block_index], Content::ToolResult { output, .. }
                if output.len() > digest.len() && may_fold(output))
        }).collect::<Vec<_>>();
        // No copy-on-write clone or revision bump for already-folded results
        // or observations shorter than their replacement notice.
        if replacements.is_empty() {
            return;
        }
        let messages = self.make_mut();
        for (message_index, block_index) in replacements {
            if let Content::ToolResult { output, .. } =
                &mut messages[message_index].content[block_index]
            {
                *output = digest.to_string();
            }
        }
    }
}
