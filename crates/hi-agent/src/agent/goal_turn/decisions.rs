//! Durable decision tool publication.
use super::*;

impl crate::Agent {
    /// Handle a `record_decision` tool call: parse the args, append to the
    /// durable decision log (which feeds the system prompt), and return a
    /// terse confirmation for the model. Malformed args yield an error string
    /// (the model sees it and can retry), not a panic.
    pub(crate) async fn handle_record_decision(
        &mut self,
        arguments: &str,
    ) -> hi_tools::ToolOutcome {
        #[derive(serde::Deserialize)]
        struct DecisionArgs {
            summary: String,
            rationale: String,
            #[serde(default)]
            files: Vec<String>,
        }
        match serde_json::from_str::<DecisionArgs>(arguments) {
            Ok(args) => {
                let summary = args.summary.trim().to_string();
                if summary.is_empty() {
                    return decision_tool_outcome(
                        "Error: record_decision needs a non-empty summary".to_string(),
                        hi_tools::ToolStatus::Failed,
                    );
                }
                let mut next = self.decisions.clone();
                next.record(Decision {
                    summary,
                    rationale: args.rationale.trim().to_string(),
                    files: args.files,
                });
                let record = next.clone();
                if let Err(err) = self
                    .write_session(move |sink| sink.record_decisions(&record))
                    .await
                {
                    return decision_tool_outcome(
                        format!("Error: couldn't persist decision: {err}"),
                        hi_tools::ToolStatus::Failed,
                    );
                }
                self.decisions = next;
                // Refresh the system prompt so the decision is injected on the
                // next turn (and visible to the model immediately in history).
                self.refresh_system_message();
                decision_tool_outcome(
                    "Decision recorded — it will persist across compaction.".to_string(),
                    hi_tools::ToolStatus::Succeeded,
                )
            }
            Err(err) => decision_tool_outcome(
                format!("Error: bad record_decision arguments: {err}"),
                hi_tools::ToolStatus::Failed,
            ),
        }
    }
}
