//! Retain the newest usable observations without discarding failed-read evidence.

use std::collections::BTreeSet;

pub(super) fn read_result_is_complete(output: &hi_tools::ToolOutcome) -> bool {
    output.status == hi_tools::ToolStatus::Succeeded
        && output.truncation == hi_tools::TruncationState::Complete
}

pub(super) fn fold_completed_observations(
    transcript: &mut crate::Transcript,
    calls: &[(String, String, String)],
    complete_reads: &[bool],
) {
    let mut handles = BTreeSet::new();
    let mut read_keys = Vec::new();
    // Only the newest read of a given shape can supersede its predecessors.
    // An earlier successful read in the same batch cannot authorize folding
    // through a later failed, denied, cancelled, or clipped result.
    for (index, (_, name, arguments)) in calls.iter().enumerate().rev() {
        if name == "bash_output"
            && let Some(handle) = crate::transcript::background_poll_handle(arguments)
            && handles.insert(handle.clone())
        {
            transcript.fold_superseded_background_polls(&handle);
        }
        if name == "read"
            && let Some(key) = crate::transcript::read_call_key(arguments)
            && !read_keys.contains(&key)
        {
            if complete_reads[index] {
                transcript.fold_superseded_file_reads(&key);
            }
            read_keys.push(key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_ai::{Content, Message};

    #[test]
    fn incomplete_results_cannot_supersede_source() {
        for status in [
            hi_tools::ToolStatus::Succeeded,
            hi_tools::ToolStatus::Failed,
            hi_tools::ToolStatus::Denied,
            hi_tools::ToolStatus::Cancelled,
            hi_tools::ToolStatus::TimedOut,
        ] {
            for truncated in [false, true] {
                let output = hi_tools::ToolOutcome {
                    status,
                    truncation: if truncated {
                        hi_tools::TruncationState::Truncated {
                            original_bytes: 99,
                            retained_bytes: 9,
                        }
                    } else {
                        hi_tools::TruncationState::Complete
                    },
                    content: "result".into(),
                    display: None,
                    plan: None,
                    process: None,
                    background: None,
                    effects: hi_tools::ToolEffects::default(),
                    images: Vec::new(),
                };
                assert_eq!(
                    read_result_is_complete(&output),
                    status == hi_tools::ToolStatus::Succeeded && !truncated
                );
            }
        }
    }

    #[test]
    fn successful_earlier_read_does_not_mask_a_failed_latest_read_in_the_batch() {
        let mut transcript = crate::Transcript::new(vec![Message::user("inspect it")]);
        let calls: Vec<(String, String, String)> = ["first", "last"]
            .into_iter()
            .map(|id| {
                (
                    id.into(),
                    "read".into(),
                    r#"{"path":"src/parser.rs"}"#.into(),
                )
            })
            .collect();
        transcript.push_assistant_with_results(
            calls
                .iter()
                .map(|(id, name, arguments)| Content::ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: arguments.clone(),
                })
                .collect(),
            vec![
                ("first".into(), "usable source".into()),
                ("last".into(), "Error: file disappeared".into()),
            ],
        );
        let before = serde_json::to_value(transcript.as_slice()).unwrap();
        fold_completed_observations(&mut transcript, &calls, &[true, false]);
        assert_eq!(serde_json::to_value(transcript.as_slice()).unwrap(), before);
    }
}
