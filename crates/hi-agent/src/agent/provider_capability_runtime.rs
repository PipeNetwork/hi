//! Provider capability negotiation at the model-request boundary.

impl crate::Agent {
    /// Replace the default no-I/O registry. Frontends may install an explicitly
    /// bounded provider probe; ordinary construction never contacts a backend.
    pub fn set_provider_capability_registry(
        &mut self,
        registry: hi_ai::ProviderCapabilityRegistry,
    ) {
        self.provider_capability_registry = registry;
    }

    /// Bounded in-memory audit history for diagnostics and eval manifests.
    pub fn provider_capability_audit(&self) -> Vec<hi_ai::CapabilityProbeAuditRecord> {
        self.provider_capability_registry.audit_records()
    }

    pub(crate) async fn effective_provider_capabilities(
        &mut self,
    ) -> hi_ai::EffectiveProviderCapabilities {
        let model = self.config.routing.model.clone();
        self.effective_provider_capabilities_for_model(&model).await
    }

    pub(crate) async fn effective_provider_capabilities_for_model(
        &mut self,
        model: &str,
    ) -> hi_ai::EffectiveProviderCapabilities {
        let target = hi_ai::CapabilityRoute::new(
            self.config
                .routing
                .provider_route
                .as_deref()
                .unwrap_or("unknown"),
            model,
        );
        let candidates = self
            .provider
            .capability_candidates(&target.route, &target.model);
        self.provider_capability_registry
            .resolve_candidates(target, &candidates)
            .await
    }
}
