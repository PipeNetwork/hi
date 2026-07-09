//! `App` methods: models.

use std::collections::HashMap;

use anyhow::Result;
use hi_agent::Agent;
use ratatui::style::{Color, Style};
use ratatui::text::Line;

use crate::render::dim;

impl crate::App {
    pub(crate) fn served_tags(&self) -> HashMap<String, String> {
        HashMap::new()
    }

    /// Apply `id` as the model: prefer live endpoint metadata (window/price) when
    /// we have it. Updates the agent and the gauge.
    pub(crate) fn apply_model(&mut self, agent: &mut Agent, id: &str) {
        let served = self.served.get(id);
        let window = served.and_then(|m| m.context_window);
        agent.set_model(
            id.to_string(),
            window,
            served.and_then(|m| m.max_output_tokens),
        );
        self.model = id.to_string();
        self.context_window = window;
    }

    /// Persist a user-selected model back to the active profile, when there is
    /// one. Startup metadata refreshes call `apply_model` directly and skip this.
    pub(crate) fn persist_active_profile_model(&mut self, id: &str) -> Result<Option<String>> {
        let Some(name) = self.active_profile.clone() else {
            return Ok(None);
        };
        let mut data = (self.loader)(&name)?;
        if data.model != id {
            data.model = id.to_string();
            self.profiles = (self.saver)(&data)?;
        }
        Ok(Some(name))
    }

    /// Apply an explicit user model selection and save it to the active profile.
    pub(crate) fn select_model(&mut self, agent: &mut Agent, id: &str) {
        self.apply_model(agent, id);
        match self.persist_active_profile_model(id) {
            Ok(Some(name)) => self.push(Line::styled(
                format!("model set to {id} (saved to profile {name})"),
                dim(),
            )),
            Ok(None) => self.push(Line::styled(format!("model set to {id}"), dim())),
            Err(err) => {
                self.push(Line::styled(format!("model set to {id}"), dim()));
                self.push(Line::styled(
                    format!("couldn't save model to active profile: {err:#}"),
                    Style::default().fg(Color::Yellow),
                ));
            }
        }
    }

    /// Percent of the context window currently occupied, when the window is known.
    pub(crate) fn context_pct(&self) -> Option<u64> {
        let window = u64::from(self.context_window?);
        (window > 0).then(|| (self.context_used * 100 / window).min(100))
    }

    /// Apply the picker's current selection as the model, then close it.
    pub(crate) fn pick_model(&mut self, agent: &mut Agent) {
        let id = self
            .picker
            .as_ref()
            .and_then(|p| p.current())
            .map(str::to_string);
        if let Some(id) = id {
            self.select_model(agent, &id);
        }
        self.picker = None;
        self.follow();
    }
}
