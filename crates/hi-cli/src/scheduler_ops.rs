//! Scheduler presets, feature rollback switches, and stale-state recovery.

use std::path::Path;
use std::time::{Duration, SystemTime};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SchedulerPreset {
    Conservative,
    Balanced,
    Throughput,
}

impl SchedulerPreset {
    pub(crate) fn from_env() -> Self {
        match std::env::var("HI_SCHEDULER_PRESET").as_deref() {
            Ok("conservative") => Self::Conservative,
            Ok("throughput") => Self::Throughput,
            _ => Self::Balanced,
        }
    }

    pub(crate) fn process_capacity(self) -> usize {
        match self {
            Self::Conservative => 1,
            Self::Balanced => 4,
            Self::Throughput => 8,
        }
    }
}

pub(crate) fn feature_enabled(name: &str) -> bool {
    if SchedulerPreset::from_env() == SchedulerPreset::Conservative {
        return false;
    }
    std::env::var(name).map_or(true, |value| value != "0" && value != "false")
}

pub(crate) fn effective_summary() -> String {
    let preset = SchedulerPreset::from_env();
    format!(
        "preset={preset:?} process_capacity={} adaptive={} warm_workers={}",
        preset.process_capacity(),
        feature_enabled("HI_ADAPTIVE_SCHEDULER"),
        feature_enabled("HI_WARM_WORKERS"),
    )
}

pub(crate) fn recover_stale_state(state_root: &Path) -> usize {
    let mut recovered = 0;
    for directory in ["resource-leases", "verification-flights"] {
        let root = state_root.join(directory);
        let Ok(entries) = std::fs::read_dir(root) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let stale = entry
                .metadata()
                .ok()
                .and_then(|metadata| metadata.modified().ok())
                .and_then(|modified| SystemTime::now().duration_since(modified).ok())
                .is_some_and(|age| age >= Duration::from_secs(60 * 60));
            if stale && std::fs::remove_file(path).is_ok() {
                recovered += 1;
            }
        }
    }
    recovered
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_have_ordered_capacity() {
        assert!(
            SchedulerPreset::Conservative.process_capacity()
                < SchedulerPreset::Balanced.process_capacity()
        );
        assert!(
            SchedulerPreset::Balanced.process_capacity()
                < SchedulerPreset::Throughput.process_capacity()
        );
    }
}
