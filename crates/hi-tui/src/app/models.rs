//! `App` methods: models.

use std::collections::HashMap;

use hi_agent::Agent;
use ratatui::style::{Color, Style};
use ratatui::text::Line;

use crate::render::dim;

impl crate::App {
    /// Health tags (id → label) for the models we have live metadata on, for the
    /// `/model` picker. Healthy models are omitted.
    /// Build a per-model-id capabilities map from the registry, for the model
    /// picker's capability-tag column.
    pub(crate) fn capabilities_map(
        registry: &hi_ai::Registry,
        ids: &[String],
    ) -> HashMap<String, Vec<&'static str>> {
        ids.iter()
            .map(|id| (id.clone(), registry.capabilities(id)))
            .collect()
    }

    pub(crate) fn served_tags(&self) -> HashMap<String, String> {
        self.served
            .iter()
            .map(|(id, m)| {
                let tag = match (
                    m.health().map(str::to_string),
                    self.model_issues.get(id).copied().unwrap_or(0),
                ) {
                    (Some(endpoint), issues) if issues > 0 => {
                        format!("{endpoint}; degraded in-session")
                    }
                    (Some(endpoint), _) => endpoint,
                    (None, issues) if issues > 0 => "degraded in-session".to_string(),
                    (None, _) => String::new(),
                };
                (id.clone(), tag)
            })
            .filter_map(|(id, tag)| (!tag.is_empty()).then_some((id, tag)))
            .collect()
    }

    /// Apply `id` as the model: prefer live endpoint metadata (window/price) when
    /// we have it, else the catalog. Updates the agent and the gauge. Returns the
    /// model's health label if the endpoint flags it as not fully available.
    pub(crate) fn apply_model(
        &mut self,
        agent: &mut Agent,
        registry: &hi_ai::Registry,
        id: &str,
    ) -> Option<String> {
        let (_cat_price, cat_window) = registry.metadata(id);
        let served = self.served.get(id);
        let window = served.and_then(|m| m.context_window).or(cat_window);
        agent.set_model(id.to_string(), window);
        self.model = id.to_string();
        self.context_window = window;
        served.and_then(|m| m.health()).map(str::to_string)
    }

    /// Push a yellow line warning that `id` is in a non-healthy state.
    pub(crate) fn warn_degraded(&mut self, id: &str, health: &str) {
        self.push(Line::styled(
            format!(
                "⚠ {id} is reported {health} on this endpoint — responses may be slow or flaky; \
                 /model to pick another"
            ),
            Style::default().fg(Color::Yellow),
        ));
    }

    /// Percent of the context window currently occupied, when the window is known.
    pub(crate) fn context_pct(&self) -> Option<u64> {
        let window = u64::from(self.context_window?);
        (window > 0).then(|| (self.context_used * 100 / window).min(100))
    }

    /// Apply the picker's current selection as the model, then close it.
    pub(crate) fn pick_model(&mut self, agent: &mut Agent, registry: &hi_ai::Registry) {
        let id = self
            .picker
            .as_ref()
            .and_then(|p| p.current())
            .map(str::to_string);
        if let Some(id) = id {
            let health = self.apply_model(agent, registry, &id);
            self.push(Line::styled(format!("model set to {id}"), dim()));
            if let Some(h) = health {
                self.warn_degraded(&id, &h);
            }
        }
        self.picker = None;
        self.follow();
    }
}
