//! Auto-record durable coding facts after a green verified turn.

use crate::Ui;
use crate::coding_memory::{
    CodingFactInput, extract_coding_facts, merge_facts_into_workspace_memory,
};
use crate::memory::memory_file_at;

impl crate::Agent {
    /// After a turn that passed verification and changed files, extract durable
    /// coding facts (verify command, package ownership, stack, test gate) into
    /// the session decision log and project memory. Best-effort, no model call.
    pub(crate) async fn record_coding_facts_turn_end(&mut self, ui: &mut dyn Ui) {
        if !self.report.verify.passed() || self.workspace.last_changed_files.is_empty() {
            return;
        }

        let wants_tests = self
            .task
            .last_task_contract
            .as_ref()
            .is_some_and(|c| c.wants_tests);
        let facts = extract_coding_facts(&CodingFactInput {
            changed_files: &self.workspace.last_changed_files,
            verify_executions: &self.report.last_turn_telemetry.verification_executions,
            wants_tests,
            workspace_root: self.runtime.root(),
        });
        if facts.is_empty() {
            return;
        }

        let mut next = self.decisions.clone();
        for fact in &facts {
            next.record(fact.clone());
        }
        if let Some(session) = self.session.as_mut()
            && let Err(err) = session.record_decisions(&next)
        {
            ui.status(&format!("(couldn't persist coding facts: {err})"));
            return;
        }
        self.decisions = next;
        self.subagents.coding_facts_written = self
            .subagents
            .coding_facts_written
            .saturating_add(u32::try_from(facts.len()).unwrap_or(u32::MAX));
        self.refresh_system_message();

        // Project memory merge is best-effort and independent of the decision log.
        let mem_path = memory_file_at(self.runtime.root());
        let declared_paths = crate::memory::memory_write_paths(&mem_path);
        let fact_count = facts.len();
        let workspace_root = self.runtime.root().to_path_buf();
        let state_root = self.runtime.state_root().to_path_buf();
        let write = self
            .run_internal_file_mutation(
                "record_coding_memory",
                "record verifier-backed coding memory",
                &declared_paths,
                serde_json::json!({ "fact_count": fact_count }),
                || {
                    let added = merge_facts_into_workspace_memory(
                        &workspace_root,
                        &state_root,
                        &mem_path,
                        &facts,
                    )
                    .map_err(anyhow::Error::msg)?;
                    Ok((
                        added,
                        format!(
                            "recorded {added} new coding-memory bullets from {fact_count} facts"
                        ),
                    ))
                },
            )
            .await;
        match write {
            Ok(0) => {
                ui.status(&format!(
                    "coding memory · {} decision(s) (memory already current)",
                    facts.len()
                ));
            }
            Ok(n) => {
                ui.status(&format!(
                    "coding memory · {} decision(s), {n} new memory bullet(s)",
                    facts.len()
                ));
            }
            Err(err) => {
                ui.status(&format!(
                    "coding memory · {} decision(s); memory write skipped: {err}",
                    facts.len()
                ));
            }
        }
        // Phase P: re-rank live memory so the next model call sees new bullets
        // without waiting for process restart / next session load.
        let task = self.task.last_task_prompt.clone().unwrap_or_default();
        self.refresh_memory_context(&task);
        self.refresh_system_message();
    }
}
