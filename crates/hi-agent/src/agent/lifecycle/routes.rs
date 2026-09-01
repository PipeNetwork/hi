//! Team-role routing and managed local model-server ownership.

impl crate::Agent {
    /// The team-role table for `/team`: each role with the model and route it
    /// currently runs on. Roles that inherit the driver say so explicitly.
    pub fn team_roles(&self) -> Vec<crate::TeamRole> {
        let driver_model = self.config.routing.model.clone();
        let driver_route = self
            .config
            .routing
            .provider_route
            .clone()
            .unwrap_or_else(|| "driver provider".to_string());
        let sub = &self.config.subagents;
        let role = |role: &'static str,
                    model: &Option<String>,
                    endpoint: &Option<String>|
         -> crate::TeamRole {
            let stale = self.team_route_is_dead(model.as_deref(), endpoint.as_deref());
            let model = if stale { None } else { model.clone() };
            let endpoint = if stale { None } else { endpoint.clone() };
            let inherited = model.is_none() && endpoint.is_none();
            crate::TeamRole {
                role,
                model: model.clone().unwrap_or_else(|| driver_model.clone()),
                route: endpoint.clone().unwrap_or_else(|| driver_route.clone()),
                inherited,
            }
        };
        let delegate_dead = self.team_route_is_dead(
            sub.delegate_model.as_deref(),
            sub.delegate_endpoint.as_deref(),
        );
        let editor_dead =
            self.team_route_is_dead(sub.editor_model.as_deref(), sub.editor_endpoint.as_deref());
        let editor_inherited =
            (sub.editor_model.is_none() && sub.editor_endpoint.is_none()) || editor_dead;
        let editor = if editor_inherited
            && !delegate_dead
            && (sub.delegate_model.is_some() || sub.delegate_endpoint.is_some())
        {
            // Mechanical edits fall back to the delegate lane when no editor
            // override exists. Show that effective route in `/team`; reporting
            // the driver here made the table disagree with actual execution.
            crate::TeamRole {
                role: "editor",
                model: sub
                    .delegate_model
                    .clone()
                    .unwrap_or_else(|| driver_model.clone()),
                route: sub
                    .delegate_endpoint
                    .clone()
                    .unwrap_or_else(|| driver_route.clone()),
                // It is inherited from the delegate role, not the driver;
                // mark it non-inherited so frontends do not label it as a
                // driver route.
                inherited: false,
            }
        } else {
            role("editor", &sub.editor_model, &sub.editor_endpoint)
        };
        let skeptic_stale = self.skeptic_route_is_dead();
        let skeptic_model = if skeptic_stale {
            None
        } else {
            sub.skeptic_model.clone()
        };
        let skeptic_endpoint = if skeptic_stale {
            None
        } else {
            sub.skeptic_endpoint.clone()
        };
        let skeptic = crate::TeamRole {
            role: "skeptic",
            model: skeptic_model
                .clone()
                .unwrap_or_else(|| driver_model.clone()),
            route: skeptic_endpoint
                .clone()
                .unwrap_or_else(|| driver_route.clone()),
            inherited: skeptic_model.is_none() && skeptic_endpoint.is_none(),
        };
        vec![
            crate::TeamRole {
                role: "driver",
                model: driver_model.clone(),
                route: driver_route.clone(),
                inherited: false,
            },
            role("explore", &sub.explore_model, &sub.explore_endpoint),
            role("delegate", &sub.delegate_model, &sub.delegate_endpoint),
            editor,
            skeptic,
            role("planner", &sub.planner_model, &None),
        ]
    }

    /// Point the write-capable `delegate` executors at a different model
    /// and/or OpenAI-compatible endpoint (`None`s inherit the driver).
    /// Applies to delegates started after the call.
    pub fn set_delegate_route(
        &mut self,
        model: Option<String>,
        endpoint: Option<String>,
        api_key: Option<String>,
    ) {
        self.config.subagents.delegate_model = normalized(model);
        self.config.subagents.delegate_endpoint = normalized(endpoint);
        self.config.subagents.delegate_endpoint_key = normalized(api_key);
        self.release_unreferenced_team_servers();
    }

    /// Point read-only `explore` recon children at a different model and/or
    /// endpoint (`None`s inherit the driver). Applies to explores started
    /// after the call.
    pub fn set_explore_route(
        &mut self,
        model: Option<String>,
        endpoint: Option<String>,
        api_key: Option<String>,
    ) {
        self.config.subagents.explore_model = normalized(model);
        self.config.subagents.explore_endpoint = normalized(endpoint);
        self.config.subagents.explore_endpoint_key = normalized(api_key);
        self.release_unreferenced_team_servers();
    }

    /// Point `delegate` calls tagged `kind: "edit"` (mechanical changes) at a
    /// different model and/or endpoint (`None`s fall back to the delegate
    /// route). Applies to delegates started after the call.
    pub fn set_editor_route(
        &mut self,
        model: Option<String>,
        endpoint: Option<String>,
        api_key: Option<String>,
    ) {
        self.config.subagents.editor_model = normalized(model);
        self.config.subagents.editor_endpoint = normalized(endpoint);
        self.config.subagents.editor_endpoint_key = normalized(api_key);
        self.release_unreferenced_team_servers();
    }

    /// Route a `/team` role by name (`delegate`, `explore`, `editor`).
    /// Returns `false` for roles without a model route so frontends can fall
    /// through to their own help text.
    pub fn set_team_route(
        &mut self,
        role: &str,
        model: Option<String>,
        endpoint: Option<String>,
        api_key: Option<String>,
    ) -> bool {
        match role {
            "delegate" => self.set_delegate_route(model, endpoint, api_key),
            "explore" => self.set_explore_route(model, endpoint, api_key),
            "editor" => self.set_editor_route(model, endpoint, api_key),
            _ => return false,
        }
        true
    }

    /// Set or clear the goal-decomposition planner model (`/team planner`).
    pub fn set_planner_model(&mut self, model: Option<String>) {
        self.config.subagents.planner_model = normalized(model);
    }

    /// The auto-managed local model server, when one is running (started by
    /// `/config skeptic-local on`): `(base_url, model_id)`. `/team <role>
    /// local` reuses it so a role can move on-device with one command.
    pub fn managed_local_route(&self) -> Option<(String, String)> {
        self.local_skeptic
            .as_ref()
            .filter(|state| self.local_skeptic_server_is_running(state))
            .map(|state| (state.endpoint.clone(), state.model_id.clone()))
    }

    /// A running managed server (skeptic or team) already serving `model_id`,
    /// if any — `/team` reuses it instead of spawning a duplicate.
    pub fn running_local_model_server(&self, model_id: &str) -> Option<(String, String)> {
        if let Some(server) = &self.driver_local_server
            && server.model_id == model_id
            && hi_tools::local_server_is_running(&server.process_id)
        {
            return Some((server.endpoint.clone(), server.model_id.clone()));
        }
        if let Some(state) = &self.local_skeptic
            && state.model_id == model_id
            && self.local_skeptic_server_is_running(state)
        {
            return Some((state.endpoint.clone(), state.model_id.clone()));
        }
        self.team_local_servers
            .iter()
            .find(|server| {
                server.model_id == model_id && hi_tools::local_server_is_running(&server.process_id)
            })
            .map(|server| (server.endpoint.clone(), server.model_id.clone()))
    }

    /// Return the process id for a reusable managed local runtime.
    pub fn running_local_model_process(&self, model_id: &str) -> Option<String> {
        if let Some(server) = &self.driver_local_server
            && server.model_id == model_id
            && hi_tools::local_server_is_running(&server.process_id)
        {
            return Some(server.process_id.clone());
        }
        if let Some(state) = &self.local_skeptic
            && state.model_id == model_id
            && !state.process_id.is_empty()
            && hi_tools::local_server_is_running(&state.process_id)
        {
            return Some(state.process_id.clone());
        }
        self.team_local_servers
            .iter()
            .find(|server| {
                server.model_id == model_id && hi_tools::local_server_is_running(&server.process_id)
            })
            .map(|server| server.process_id.clone())
    }

    /// Register a managed local server as the driver runtime.
    pub fn register_driver_local_server(
        &mut self,
        endpoint: String,
        model_id: String,
        process_id: String,
    ) {
        if let Some(previous) = self.driver_local_server.take()
            && previous.process_id != process_id
            && !self
                .team_local_servers
                .iter()
                .any(|server| server.process_id == previous.process_id)
        {
            hi_tools::stop_local_server(&previous.process_id);
        }
        self.driver_local_server = Some(crate::TeamLocalServer {
            process_id,
            endpoint,
            model_id,
        });
    }

    /// Clear the driver runtime when switching back to a non-local provider.
    pub fn clear_driver_local_server(&mut self) {
        let Some(previous) = self.driver_local_server.take() else {
            return;
        };
        if !self
            .team_local_servers
            .iter()
            .any(|server| server.process_id == previous.process_id)
        {
            hi_tools::stop_local_server(&previous.process_id);
        }
    }

    /// Any running team-role local server: `(endpoint, model_id)`. The
    /// skeptic reuses it — a provisioned executor (e.g. laguna) reviews for
    /// free instead of downloading and serving a second, smaller model.
    pub fn any_team_local_server(&self) -> Option<(String, String)> {
        self.team_local_servers
            .iter()
            .find(|server| hi_tools::local_server_is_running(&server.process_id))
            .map(|server| (server.endpoint.clone(), server.model_id.clone()))
    }

    pub(crate) fn local_skeptic_server_is_running(
        &self,
        state: &crate::local_skeptic::LocalSkepticState,
    ) -> bool {
        if !state.process_id.is_empty() {
            return hi_tools::local_server_is_running(&state.process_id);
        }
        // An empty process id means the skeptic is riding a team server. Find
        // that owner and verify its child before treating the route as reusable.
        self.team_local_servers.iter().any(|server| {
            server.endpoint == state.endpoint
                && server.model_id == state.model_id
                && hi_tools::local_server_is_running(&server.process_id)
        })
    }

    /// Whether a configured team route points at a managed local server that
    /// has exited. Explicit external endpoints are not considered stale: hi
    /// does not own their process and cannot infer their health here.
    pub(crate) fn team_route_is_dead(&self, model: Option<&str>, endpoint: Option<&str>) -> bool {
        let (Some(model), Some(endpoint)) = (model, endpoint) else {
            return false;
        };
        let matching = self
            .team_local_servers
            .iter()
            .filter(|server| server.model_id == model && server.endpoint == endpoint);
        let mut found = false;
        let mut running = false;
        for server in matching {
            found = true;
            running |= hi_tools::local_server_is_running(&server.process_id);
        }
        found && !running
    }

    /// Whether the configured skeptic endpoint belongs to one of hi's managed
    /// local servers and that server has exited. Explicit external endpoints
    /// are intentionally left alone: hi cannot probe or own their lifecycle,
    /// and an HTTP endpoint may be healthy even when it is not in our process
    /// registry.
    pub(crate) fn skeptic_route_is_dead(&self) -> bool {
        let (Some(model), Some(endpoint)) = (
            self.config.subagents.skeptic_model.as_deref(),
            self.config.subagents.skeptic_endpoint.as_deref(),
        ) else {
            return false;
        };
        if let Some(state) = &self.local_skeptic
            && state.model_id == model
            && state.endpoint == endpoint
        {
            return !self.local_skeptic_server_is_running(state);
        }
        self.team_route_is_dead(Some(model), Some(endpoint))
    }

    /// Record a provisioned team-role server so later `/team` picks of the
    /// same model reuse it and session teardown can stop it.
    pub fn register_team_local_server(
        &mut self,
        endpoint: String,
        model_id: String,
        process_id: String,
    ) {
        if self
            .team_local_servers
            .iter()
            .any(|server| server.process_id == process_id)
        {
            return;
        }
        self.team_local_servers.push(crate::TeamLocalServer {
            process_id,
            endpoint,
            model_id,
        });
    }

    /// Stop team-local servers that are no longer referenced by any executor
    /// route or by a skeptic riding a team server. Without this reconciliation,
    /// switching `/team delegate` back to the driver leaves a model server
    /// consuming its memory until the whole session exits.
    pub(crate) fn release_unreferenced_team_servers(&mut self) {
        let sub = &self.config.subagents;
        let driver_route = self
            .driver_local_server
            .as_ref()
            .map(|server| (server.model_id.as_str(), server.endpoint.as_str()));
        let skeptic_route = sub
            .skeptic_model
            .as_deref()
            .zip(sub.skeptic_endpoint.as_deref());
        let mut stopped = Vec::new();
        self.team_local_servers.retain(|server| {
            let referenced = [
                driver_route,
                sub.delegate_model
                    .as_deref()
                    .zip(sub.delegate_endpoint.as_deref()),
                sub.explore_model
                    .as_deref()
                    .zip(sub.explore_endpoint.as_deref()),
                sub.editor_model
                    .as_deref()
                    .zip(sub.editor_endpoint.as_deref()),
                skeptic_route,
            ]
            .into_iter()
            .flatten()
            .any(|(model, endpoint)| server.model_id == model && server.endpoint == endpoint);
            if !referenced {
                stopped.push(server.process_id.clone());
            }
            referenced
        });
        for process_id in stopped {
            hi_tools::stop_local_server(&process_id);
        }
    }
}

/// Trim a role-route input; empty strings mean "inherit" (`None`).
fn normalized(value: Option<String>) -> Option<String> {
    value
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}
