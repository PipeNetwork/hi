//! Compact model-facing session snapshot used by `/btw` side questions.

impl crate::Agent {
    /// A compact, model-facing snapshot of the current session, attached to
    /// `/btw` side questions so the model can answer "what's the status / what
    /// are you doing / what changed / how old is the repo" without a tool round
    /// when the fact is already known. Kept short — it is injected into a
    /// throwaway side completion (which may still run a few read-only tools).
    pub(crate) fn btw_session_snapshot(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("- model: {}", self.model()));
        if let Some(route) = self.provider_route() {
            lines.push(format!("- provider route: {route}"));
        }
        lines.push(format!("- workspace: {}", self.workspace_root().display()));
        lines.push(crate::today::snapshot_line());
        // Cheap git facts (branch, HEAD, first/latest commit). The snapshot is
        // rebuilt at every model boundary, but the facts only change when the
        // reconciled workspace revision changes. Cache them so `/btw` does not
        // synchronously launch several Git processes on every round.
        let revision = self.runtime.ledger().revision();
        let cached_facts = self.btw_git_facts_cache.lock().ok().and_then(|cache| {
            cache
                .as_ref()
                .filter(|(cached_revision, _)| *cached_revision == revision)
                .map(|(_, facts)| facts.clone())
        });
        let git_facts = cached_facts.unwrap_or_else(|| {
            let facts = crate::git_identity::btw_lines(self.workspace_root());
            if let Ok(mut cache) = self.btw_git_facts_cache.lock() {
                *cache = Some((revision, facts.clone()));
            }
            facts
        });
        for line in git_facts {
            lines.push(line);
        }
        let goal = self.goal_summary();
        if goal != "off" {
            lines.push(format!("- goal: {goal}"));
        }
        let plan = self.current_plan();
        if !plan.is_empty() {
            let done = plan
                .iter()
                .filter(|s| s.status == hi_tools::PlanStatus::Done)
                .count();
            lines.push(format!("- plan: {done}/{} steps done", plan.len()));
            for step in plan {
                let mark = match step.status {
                    hi_tools::PlanStatus::Done => "✓",
                    hi_tools::PlanStatus::Active => "→",
                    hi_tools::PlanStatus::Pending => "·",
                };
                lines.push(format!("    {mark} {}", step.title));
            }
        }
        let checkpoints = self.checkpoint_count();
        if checkpoints > 0 {
            lines.push(format!("- checkpoints: {checkpoints}"));
        }
        let changed = self.last_changed_files();
        if !changed.is_empty() {
            let preview: Vec<&str> = changed.iter().take(8).map(String::as_str).collect();
            let mut line = format!("- files changed this turn: {}", preview.join(", "));
            if changed.len() > preview.len() {
                line.push_str(&format!(" (+{} more)", changed.len() - preview.len()));
            }
            lines.push(line);
        }
        // Live background jobs (loops, dev servers, training runs the agent
        // spawned). Lets the model answer "is my job still running / did it
        // finish" without polling. Command is truncated to keep the snapshot small.
        let jobs = self.background_snapshot();
        if !jobs.is_empty() {
            lines.push(format!("- background jobs: {}", jobs.len()));
            for (id, command, status) in &jobs {
                let cmd = if command.chars().count() > 60 {
                    let truncated: String = command.chars().take(57).collect();
                    format!("{truncated}…")
                } else {
                    command.clone()
                };
                lines.push(format!("    {id}: {cmd} ({status})"));
            }
        }
        lines.join("\n")
    }
}
