//! Remote self-improvement status and live control settings.

use anyhow::Result;

impl crate::Agent {
    pub fn rsi_status(&self) -> (&'static str, &'static str, Option<bool>) {
        let requested = if self.config.rsi.enabled { "on" } else { "off" };
        let mode = if self.config.rsi.managed {
            "managed"
        } else if self.config.rsi.enabled {
            "remote"
        } else {
            "off"
        };
        (requested, mode, self.rsi_observe.last_fully_observed)
    }

    pub fn rsi_maximum_cost_microusd(&self) -> Option<u64> {
        self.config
            .rsi
            .control
            .as_ref()
            .map(|control| control.maximum_cost_microusd())
    }

    pub fn rsi_channel(&self) -> &'static str {
        self.config
            .rsi
            .control
            .as_ref()
            .map_or("stable", |control| control.channel())
    }

    pub fn set_rsi_channel(&mut self, channel: crate::command::RsiChannel) -> Result<()> {
        let control = self
            .config
            .rsi
            .control
            .clone()
            .ok_or_else(|| anyhow::anyhow!("remote RSI is not configured"))?;
        control.set_channel(channel.as_str())
    }

    pub async fn rsi_public_status(&self) -> Result<String> {
        let control = self
            .config
            .rsi
            .control
            .clone()
            .ok_or_else(|| anyhow::anyhow!("remote RSI is not configured"))?;
        control.status().await
    }

    pub fn set_rsi_maximum_cost_microusd(&mut self, value: u64) -> Result<()> {
        anyhow::ensure!(
            (1..=15_000_000).contains(&value),
            "RSI spend limit must be greater than $0 and no more than $15"
        );
        let control = self
            .config
            .rsi
            .control
            .clone()
            .ok_or_else(|| anyhow::anyhow!("remote RSI is not configured"))?;
        control.set_maximum_cost_microusd(value)
    }

    pub fn set_rsi_enabled(&mut self, enabled: bool) -> Result<()> {
        anyhow::ensure!(
            !enabled || !self.workspace_durability_enabled(),
            "remote RSI is unavailable while PipeFS is active because its patch runner is bound to the launch workspace"
        );
        anyhow::ensure!(
            !self.config.rsi.managed || enabled,
            "managed RSI cannot be disabled"
        );
        if enabled && !self.config.rsi.managed {
            anyhow::ensure!(
                self.config.rsi.remote_switch.is_some(),
                "remote RSI requires PIPENETWORK_API_KEY or an active Pipe provider key"
            );
        }
        self.config.rsi.enabled = enabled;
        if let Some(switch) = &self.config.rsi.remote_switch {
            switch.store(enabled, std::sync::atomic::Ordering::SeqCst);
        }
        Ok(())
    }

    pub async fn set_rsi_enabled_validated(&mut self, enabled: bool) -> Result<()> {
        anyhow::ensure!(
            !enabled || !self.workspace_durability_enabled(),
            "remote RSI is unavailable while PipeFS is active because its patch runner is bound to the launch workspace"
        );
        let control = self.config.rsi.control.clone();
        if enabled && !self.config.rsi.managed {
            let control = control
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("remote RSI is not configured"))?;
            control.validate().await?;
        }
        if !self.config.rsi.managed
            && let Some(control) = &control
        {
            control.persist_enabled(enabled)?;
        }
        self.set_rsi_enabled(enabled)
    }

    pub async fn rsi_command(&self, argument: &str) -> Result<String> {
        anyhow::ensure!(
            !self.workspace_durability_enabled(),
            "RSI commands are unavailable while PipeFS is active because RSI artifacts are bound to the launch workspace"
        );
        let control = self
            .config
            .rsi
            .control
            .clone()
            .ok_or_else(|| anyhow::anyhow!("remote RSI is not configured"))?;
        control.command(argument).await
    }

    pub fn set_last_rsi_fully_observed(&mut self, observed: Option<bool>) {
        self.rsi_observe.set_last_fully_observed(observed);
    }
}
