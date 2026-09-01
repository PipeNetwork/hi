//! Append-only local orchestration metrics and a compact per-kind dashboard.

use std::collections::BTreeMap;
use std::path::Path;

pub(crate) fn record(state_root: &Path, kind: &str, duration_ms: u128, success: bool) {
    record_detailed(state_root, kind, duration_ms, 0, duration_ms, success, "");
}

pub(crate) fn record_detailed(
    state_root: &Path,
    kind: &str,
    duration_ms: u128,
    queue_ms: u128,
    execution_ms: u128,
    success: bool,
    resource: &str,
) {
    let path = state_root.join("orchestration-metrics.csv");
    let line = format!(
        "{kind},{duration_ms},{},{queue_ms},{execution_ms},{resource}\n",
        u8::from(success)
    );
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        use std::io::Write;
        let _ = file.write_all(line.as_bytes());
    }
}

#[derive(Default)]
struct Samples {
    durations: Vec<u128>,
    successes: usize,
}

fn percentile(samples: &mut [u128], pct: usize) -> u128 {
    if samples.is_empty() {
        return 0;
    }
    samples.sort_unstable();
    samples[(samples.len() * pct)
        .div_ceil(100)
        .saturating_sub(1)
        .min(samples.len() - 1)]
}

pub(crate) fn print_dashboard(state_root: &Path) {
    let text =
        std::fs::read_to_string(state_root.join("orchestration-metrics.csv")).unwrap_or_default();
    let mut groups: BTreeMap<String, Samples> = BTreeMap::new();
    for line in text.lines() {
        let mut fields = line.split(',');
        let (Some(kind), Some(duration), Some(success)) =
            (fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        let Ok(duration) = duration.parse::<u128>() else {
            continue;
        };
        let entry = groups.entry(kind.to_string()).or_default();
        entry.durations.push(duration);
        entry.successes += usize::from(success == "1" || success == "true");
    }
    if groups.is_empty() {
        println!("runs: 0\np50_ms: 0\np95_ms: 0");
        return;
    }
    for (kind, mut samples) in groups {
        let runs = samples.durations.len();
        let p50 = percentile(&mut samples.durations, 50);
        let p95 = percentile(&mut samples.durations, 95);
        println!(
            "{kind}: runs={runs} success={}/{} p50_ms={p50} p95_ms={p95}",
            samples.successes, runs
        );
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn empty_dashboard_is_supported() {
        let root = std::env::temp_dir().join(format!("hi-metrics-empty-{}", std::process::id()));
        super::print_dashboard(&root);
    }

    #[test]
    fn percentile_uses_nearest_rank() {
        let mut values = [1, 5, 2, 4, 3];
        assert_eq!(super::percentile(&mut values, 50), 3);
        assert_eq!(super::percentile(&mut values, 95), 5);
    }
}
