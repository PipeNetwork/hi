//! `App` methods: models.

use std::collections::HashMap;

use anyhow::Result;
use hi_agent::Agent;
use ratatui::style::Style;
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
        // `/provider xai` switches to a provider preset without creating a
        // profile, so the active name may not name one. There is nothing to
        // persist into — report "not saved" rather than failing the selection.
        if !self.profiles.iter().any(|profile| profile.name == name) {
            return Ok(None);
        }
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
                    Style::default().fg(crate::theme::theme().warning),
                ));
            }
        }
        // Always snapshot the live selection for the next workspace launch.
        self.remember_session_routing();
    }

    /// Percent of the context window currently occupied, when the window is known.
    pub(crate) fn context_pct(&self) -> Option<u64> {
        let window = u64::from(self.context_window?);
        (window > 0).then(|| (self.context_used * 100 / window).min(100))
    }

    /// Apply the picker's current selection, then close it. A picker opened
    /// by `/team <role>` assigns the chosen supported local model to that
    /// Close the picker and clear team-menu routing state (Esc/cancel path).
    pub(crate) fn close_picker(&mut self) {
        self.picker = None;
        self.team_picker_role = None;
        self.team_role_menu = false;
    }

    /// role; otherwise the selection switches the driver model as always.
    pub(crate) fn pick_model(&mut self, agent: &mut Agent) {
        let id = self
            .picker
            .as_ref()
            .and_then(|p| p.current())
            .map(str::to_string);
        if self.team_role_menu {
            self.team_role_menu = false;
            self.picker = None;
            let Some(id) = id else {
                self.follow();
                return;
            };
            match id.split_whitespace().next().unwrap_or_default() {
                "auto-setup" => self.run_team_auto_setup(agent),
                role @ ("delegate" | "editor" | "explore") => {
                    self.open_team_model_picker(role);
                }
                "skeptic" => self.toggle_team_skeptic(agent),
                _ => self.push(Line::styled(
                    "planner: set with /team planner <model|off>",
                    dim(),
                )),
            }
            self.follow();
            return;
        }
        if let Some(role) = self.team_picker_role.take() {
            self.picker = None;
            if let Some(id) = id {
                // Rows are rendered as "name — label · fit"; the leading
                // token is the catalog name.
                let name = id.split_whitespace().next().unwrap_or_default().to_string();
                if let Some(resolved) = hi_agent::local_skeptic::resolve_team_local_model(
                    &name,
                    hi_agent::local_skeptic::system_ram_gb(),
                    hi_agent::local_skeptic::detect_backend(),
                ) {
                    self.assign_supported_local_model(agent, &role, resolved);
                }
            }
            self.follow();
            return;
        }
        if let Some(id) = id {
            self.select_model(agent, &id);
        }
        self.picker = None;
        self.follow();
    }
}
