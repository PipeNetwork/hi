//! Provider capability negotiation at the model-request boundary.

impl crate::Agent {
    /// Update a provider while retaining the route identities supplied by an
    /// older frontend. The capability registry is still rotated, so a switch
    /// can never inherit observations from the prior provider instance.
    pub fn set_provider(
        &mut self,
        provider: std::sync::Arc<dyn hi_ai::Provider>,
        model: String,
        context_window: Option<u32>,
        requested_max_tokens: u32,
        max_tokens_explicit: bool,
        max_output_tokens: Option<u32>,
    ) {
        let label = self
            .config
            .routing
            .provider_route
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        let capability_identity = self
            .config
            .routing
            .capability_route
            .clone()
            .unwrap_or_else(|| label.clone());
        self.set_provider_with_route(
            provider,
            crate::AgentProviderRoute::new(label, capability_identity),
            model,
            context_window,
            requested_max_tokens,
            max_tokens_explicit,
            max_output_tokens,
        );
    }

    /// Replace the provider and both route identities atomically between turns.
    #[allow(clippy::too_many_arguments)]
    pub fn set_provider_with_route(
        &mut self,
        provider: std::sync::Arc<dyn hi_ai::Provider>,
        route: crate::AgentProviderRoute,
        model: String,
        context_window: Option<u32>,
        requested_max_tokens: u32,
        max_tokens_explicit: bool,
        max_output_tokens: Option<u32>,
    ) {
        self.provider = provider;
        self.config.routing.provider_route = Some(route.label);
        self.config.routing.capability_route = Some(route.capability_identity);
        self.provider_capability_registry = self.provider_capability_registry.rotated();
        self.config.routing.requested_max_tokens = requested_max_tokens;
        self.config.routing.max_tokens_explicit = max_tokens_explicit;
        self.set_model(model, context_window, max_output_tokens);
    }

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

    #[cfg(test)]
    pub(crate) async fn effective_provider_capabilities(
        &mut self,
    ) -> hi_ai::EffectiveProviderCapabilities {
        let model = self.config.routing.model.clone();
        self.effective_provider_capabilities_for_request(
            &model,
            hi_ai::ProviderRequestContext::auxiliary(),
        )
        .await
    }

    pub(crate) async fn user_turn_capabilities(
        &mut self,
        canonical_objective: &str,
    ) -> hi_ai::EffectiveProviderCapabilities {
        let model = self.config.routing.model.clone();
        self.effective_provider_capabilities_for_request(
            &model,
            hi_ai::ProviderRequestContext::user_turn(canonical_objective),
        )
        .await
    }

    pub(crate) async fn effective_provider_capabilities_for_model(
        &mut self,
        model: &str,
    ) -> hi_ai::EffectiveProviderCapabilities {
        self.effective_provider_capabilities_for_request(
            model,
            hi_ai::ProviderRequestContext::auxiliary(),
        )
        .await
    }

    async fn effective_provider_capabilities_for_request(
        &mut self,
        model: &str,
        context: hi_ai::ProviderRequestContext<'_>,
    ) -> hi_ai::EffectiveProviderCapabilities {
        let target = hi_ai::CapabilityRoute::new(
            self.config
                .routing
                .capability_route
                .as_deref()
                .or(self.config.routing.provider_route.as_deref())
                .unwrap_or("unknown"),
            model,
        );
        let candidates =
            self.provider
                .capability_candidates_for_request(&target.route, &target.model, context);
        self.provider_capability_registry
            .resolve_candidates(target, &candidates)
            .await
    }
}
