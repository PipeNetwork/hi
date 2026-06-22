//! Measure the Tier-0 test-output condenser against plain head+tail clipping on
//! a real log. Generate one and run it:
//!
//! ```sh
//! cargo test --workspace > /tmp/hi_test.log 2>&1 || true
//! cargo run -p hi-tools --example condense -- /tmp/hi_test.log 24000
//! cargo run -p hi-tools --example condense -- /tmp/hi_test.log 8000
//! ```
//!
//! "signal hits" counts surviving failure markers — the number that must not
//! drop. Head+tail can lose middle failures at a tight budget; condense keeps
//! them and is far smaller.

use std::{env, fs};

/// Mirror of `hi_tools`' private `truncate_to`, the baseline we're comparing to.
fn head_tail(s: &str, max: usize) -> String {
    let total = s.chars().count();
    if total <= max {
        return s.to_string();
    }
    let head_budget = max * 6 / 10;
    let tail_budget = max - head_budget;
    let head: String = s.chars().take(head_budget).collect();
    let tail: String = s.chars().skip(total - tail_budget).collect();
    format!("{head}\n… …\n{tail}")
}

fn signal_hits(s: &str) -> usize {
    ["FAILED", "panicked", "error[", "could not compile", "test result: FAILED"]
        .iter()
        .map(|m| s.matches(m).count())
        .sum()
}

fn row(label: &str, s: &str) {
    println!(
        "{label:<11} {:>8} chars   {:>3} signal hits",
        s.chars().count(),
        signal_hits(s)
    );
}

fn main() {
    let mut args = env::args().skip(1);
    let path = args.next().expect("usage: condense <logfile> [budget]");
    let budget: usize = args.next().map_or(24_000, |b| b.parse().expect("budget"));
    let log = fs::read_to_string(&path).expect("read log");

    println!("budget: {budget} chars\n");
    row("original", &log);
    row("head+tail", &head_tail(&log, budget));
    row("condensed", &hi_tools::condense_test_output(&log, budget));
}
