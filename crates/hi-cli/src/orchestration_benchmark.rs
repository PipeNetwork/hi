//! Deterministic orchestration microbenchmark.

use std::time::{Duration, Instant};

pub(crate) fn run() {
    let max_regression_percent = std::env::var("HI_BENCH_MAX_REGRESSION_PERCENT")
        .ok()
        .and_then(|value| value.parse::<u128>().ok())
        .unwrap_or(15);
    let candidates = [1usize, 2, 4, 8];
    println!("candidates,wall_ms,throughput_per_sec");
    for count in candidates {
        let started = Instant::now();
        std::thread::scope(|scope| {
            let gate = std::sync::Arc::new(std::sync::Mutex::new(()));
            for index in 0..count {
                let gate = gate.clone();
                scope.spawn(move || {
                    // Mix parallel work with a short shared critical section to
                    // model process-lease and destination-merge contention.
                    std::thread::sleep(Duration::from_millis(8));
                    if index % 2 == 0 {
                        let _guard = gate.lock().unwrap();
                        std::thread::sleep(Duration::from_millis(2));
                    }
                });
            }
        });
        let elapsed = started.elapsed();
        let throughput = count as f64 / elapsed.as_secs_f64().max(0.001);
        let wall_ms = elapsed.as_millis();
        println!("{count},{wall_ms},{throughput:.2}");
        if let Ok(baseline) = std::env::var(format!("HI_BENCH_BASELINE_{count}_MS"))
            && let Ok(baseline) = baseline.parse::<u128>()
        {
            let limit = baseline.saturating_mul(100 + max_regression_percent) / 100;
            assert!(
                wall_ms <= limit,
                "orchestration benchmark regressed: {wall_ms}ms > {limit}ms"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn benchmark_completes() {
        super::run();
    }
}
