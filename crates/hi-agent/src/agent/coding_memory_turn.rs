//! Auto-record durable coding facts after a green verified turn.

use crate::Ui;
use crate::coding_memory::{
    CodingFactInput, MAX_CODING_FACTS_PER_SESSION, extract_coding_facts, merge_facts_into_memory,
};
use crate::memory::memory_file_at;

impl crate::Agent {
    /// After a turn that passed verification and changed files, extract durable
    /// coding facts (verify command, package ownership, stack, test gate) into
    /// the session decision log and project memory. Best-effort, no model call.
    pub(crate) fn record_coding_facts_turn_end(&mut self, ui: &mut dyn Ui) {
        if self.subagents.coding_facts_written >= MAX_CODING_FACTS_PER_SESSION {
            return;
        }
        if self.report.last_verify != Some(true) || self.workspace.last_changed_files.is_empty() {
            return;
        }

        let wants_tests = self
            .task.last_task_contract
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

        // Room under the session cap.
        let budget = MAX_CODING_FACTS_PER_SESSION.saturating_sub(self.subagents.coding_facts_written) as usize;
        let facts: Vec<_> = facts.into_iter().take(budget).collect();
        if facts.is_empty() {
            return;
        }

        let mut next = self.decisions.clone();
        let before = next.entries().len();
        for fact in &facts {
            next.record(fact.clone());
        }
        let recorded = next.entries().len().saturating_sub(before).max(
            // record() may replace duplicates — still count attempted facts toward the cap
            // so a sticky summary can't spam every turn.
            facts.len().min(1),
        );
        if let Some(session) = self.session.as_mut() {
            if let Err(err) = session.record_decisions(&next) {
                ui.status(&format!("(couldn't persist coding facts: {err})"));
                return;
            }
        }
        self.decisions = next;
        self.subagents.coding_facts_written = self
            .subagents.coding_facts_written
            .saturating_add(facts.len() as u32);
        self.refresh_system_message();

        // Project memory merge is best-effort and independent of the decision log.
        let mem_path = memory_file_at(self.runtime.root());
        match merge_facts_into_memory(&mem_path, &facts) {
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
        let _ = recorded;
    }
}
