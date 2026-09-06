//! Sealed-envelope validation and launch accounting for workflow speculation.

use super::*;

#[derive(Clone)]
pub(crate) struct ProgramSpeculator {
    pub(super) runner: ProgramToolRunner,
    pub(super) tool_envelope: std::sync::Arc<hi_tools::envelope::ToolEnvelope>,
    pub(super) program_specs: std::sync::Arc<[hi_ai::ToolSpec]>,
    pub(super) turn_id: String,
    pub(super) enabled: bool,
    pub(super) external_allowed: bool,
    pub(super) max_calls: usize,
    pub(super) context_generation: u64,
    pub(super) ledger_revision: u64,
    pub(super) external_freshness_epoch: u64,
}

impl ProgramSpeculator {
    /// Returns true once an idempotent external shadow request was actually
    /// launched. Cancellation may win before the real Rhai host request, so
    /// callers must retain this bit in settlement evidence.
    pub(crate) fn launch(
        &self,
        speculation_registry: &SpeculationRegistry,
        program_id: &str,
        source: &str,
    ) -> bool {
        if !self.enabled
            || !self.tool_envelope.digest_is_valid()
            || !self.tool_envelope.admits("run_program")
            || !self
                .tool_envelope
                .matches_program_specs(&self.program_specs)
        {
            return false;
        }
        let mut external_effect_may_have_started = false;
        for call in extract_safe_literal_calls(source)
            .into_iter()
            .take(self.max_calls)
        {
            if !self.tool_envelope.admits_program(&call.name)
                || matches!(call.name.as_str(), "bash_output" | "bash_kill")
            {
                continue;
            }
            let external = matches!(
                hi_tools::speculation_class(&call.name),
                hi_tools::SpeculationClass::IdempotentExternal
            );
            if external && !self.external_allowed {
                continue;
            }
            if !matches!(
                hi_tools::speculation_class(&call.name),
                hi_tools::SpeculationClass::PureLocal
                    | hi_tools::SpeculationClass::IdempotentExternal
            ) {
                continue;
            }
            let args = serde_json::to_string(&call.arguments).unwrap_or_default();
            if hi_ai::validate_client_tool_call_with_limit(
                &format!("speculation_{}", call.occurrence),
                &call.name,
                &args,
                &self.program_specs,
                self.tool_envelope.payload.limits.max_tool_argument_bytes as usize,
            )
            .is_err()
            {
                continue;
            }
            let key = SpeculationKey::new(
                &self.turn_id,
                program_id,
                call.occurrence,
                &call.name,
                &args,
                self.context_generation,
                self.ledger_revision,
                if external {
                    self.external_freshness_epoch
                } else {
                    0
                },
            );
            let registry = speculation_registry.clone();
            let runner = self.runner.clone();
            let launched =
                registry.launch(key, external, async move { runner.execute(&call).await.0 });
            external_effect_may_have_started |= external && launched;
        }
        external_effect_may_have_started
    }
}
