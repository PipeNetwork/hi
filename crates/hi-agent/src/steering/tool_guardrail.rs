use std::collections::HashSet;
use std::hash::{Hash, Hasher};

#[derive(Default)]
pub(crate) struct ToolLoopGuardrail {
    seen_idempotent_result_hashes: HashSet<String>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ToolResultProgress {
    pub(crate) hashable_idempotent: bool,
    pub(crate) repeated_idempotent_result: bool,
}

impl ToolLoopGuardrail {
    pub(crate) fn record_tool_result(
        &mut self,
        name: &str,
        arguments: &str,
        output: &str,
    ) -> ToolResultProgress {
        // Wait-polls ("sleep 300 && du -sh …") are exempt from the
        // signature-based repeat guards, so their loop bound lives here: the
        // same poll returning byte-identical output means the awaited state
        // stopped changing.
        let hashable = is_hashable_idempotent_tool(name)
            || (name == "bash" && super::implementation::bash_call_waits(arguments));
        if !hashable || output.starts_with("Error:") {
            return ToolResultProgress::default();
        }
        let key = format!("{name}:{}", stable_result_hash(output));
        let repeated = !self.seen_idempotent_result_hashes.insert(key);
        ToolResultProgress {
            hashable_idempotent: true,
            repeated_idempotent_result: repeated,
        }
    }
}

fn is_hashable_idempotent_tool(name: &str) -> bool {
    matches!(name, "read" | "list" | "grep" | "glob")
}

fn stable_result_hash(output: &str) -> u64 {
    let normalized = serde_json::from_str::<serde_json::Value>(output)
        .map(|value| value.to_string())
        .unwrap_or_else(|_| output.replace("\r\n", "\n"));
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    normalized.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_read_result_is_no_progress_even_with_different_args() {
        let mut guard = ToolLoopGuardrail::default();

        let first = guard.record_tool_result("read", r#"{"path":"a.rs"}"#, "same output");
        let second = guard.record_tool_result("read", r#"{"path":"b.rs"}"#, "same output");

        assert!(first.hashable_idempotent);
        assert!(!first.repeated_idempotent_result);
        assert!(second.hashable_idempotent);
        assert!(second.repeated_idempotent_result);
    }

    #[test]
    fn wait_poll_bash_is_hash_guarded_but_plain_bash_is_not() {
        let mut guard = ToolLoopGuardrail::default();
        let wait_args = r#"{"command":"sleep 300 && du -sh models/"}"#;

        let first = guard.record_tool_result("bash", wait_args, "90G\t18 shards");
        assert!(first.hashable_idempotent);
        assert!(!first.repeated_idempotent_result);

        let progressed = guard.record_tool_result("bash", wait_args, "124G\t27 shards");
        assert!(progressed.hashable_idempotent);
        assert!(
            !progressed.repeated_idempotent_result,
            "changing output is progress"
        );

        let static_poll = guard.record_tool_result("bash", wait_args, "124G\t27 shards");
        assert!(
            static_poll.repeated_idempotent_result,
            "identical output means the awaited state stopped changing"
        );

        let plain = guard.record_tool_result("bash", r#"{"command":"cargo test"}"#, "ok");
        assert!(!plain.hashable_idempotent, "plain bash is not hash guarded");
    }

    #[test]
    fn mutating_tools_are_not_hash_guarded() {
        let mut guard = ToolLoopGuardrail::default();

        let first = guard.record_tool_result("write", r#"{"path":"a.rs"}"#, "Wrote a.rs");
        let second = guard.record_tool_result("write", r#"{"path":"b.rs"}"#, "Wrote a.rs");

        assert!(!first.hashable_idempotent);
        assert!(!second.repeated_idempotent_result);
    }
}
