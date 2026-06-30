#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuSpec {
    pub name: &'static str,
    pub tflops: f64,
}

pub const GPU_CATALOG: &[GpuSpec] = &[
    GpuSpec {
        name: "H100 80GB",
        tflops: 989.0,
    },
    GpuSpec {
        name: "A100 80GB",
        tflops: 312.0,
    },
    GpuSpec {
        name: "L40S",
        tflops: 362.0,
    },
    GpuSpec {
        name: "RTX 4090",
        tflops: 330.0,
    },
    GpuSpec {
        name: "RTX 3090",
        tflops: 142.0,
    },
    GpuSpec {
        name: "MI300X",
        tflops: 1300.0,
    },
];

#[derive(Clone, Debug, PartialEq)]
pub enum EstimateError {
    InvalidWorkload,
    InvalidUtilization,
    NoGpuThroughput,
}

impl std::fmt::Display for EstimateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidWorkload => write!(f, "parameters and tokens must be positive numbers"),
            Self::InvalidUtilization => {
                write!(f, "utilization percent must be greater than 0 and at most 100")
            }
            Self::NoGpuThroughput => write!(f, "at least one GPU must be selected"),
        }
    }
}

impl std::error::Error for EstimateError {}

pub fn training_flops(params: f64, tokens: f64) -> Result<f64, EstimateError> {
    if !params.is_finite() || !tokens.is_finite() || params <= 0.0 || tokens <= 0.0 {
        return Err(EstimateError::InvalidWorkload);
    }
    Ok(6.0 * params * tokens)
}

pub fn total_tflops(counts: &[u32]) -> f64 {
    GPU_CATALOG
        .iter()
        .zip(counts.iter().copied())
        .map(|(gpu, count)| gpu.tflops * f64::from(count))
        .sum()
}

pub fn estimate_seconds(
    params: f64,
    tokens: f64,
    utilization_percent: f64,
    counts: &[u32],
) -> Result<f64, EstimateError> {
    if !utilization_percent.is_finite()
        || utilization_percent <= 0.0
        || utilization_percent > 100.0
    {
        return Err(EstimateError::InvalidUtilization);
    }
    let tflops = total_tflops(counts);
    if tflops <= 0.0 {
        return Err(EstimateError::NoGpuThroughput);
    }
    let denominator = tflops * 1_000_000_000_000.0 * (utilization_percent / 100.0);
    Ok(training_flops(params, tokens)? / denominator)
}

pub fn format_duration(seconds: f64) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        return "unavailable".to_string();
    }
    let minute = 60.0;
    let hour = 60.0 * minute;
    let day = 24.0 * hour;
    let year = 365.0 * day;
    if seconds < minute {
        format!("{seconds:.1} seconds")
    } else if seconds < hour {
        format!("{:.1} minutes", seconds / minute)
    } else if seconds < day {
        format!("{:.1} hours", seconds / hour)
    } else if seconds < year {
        format!("{:.1} days", seconds / day)
    } else {
        format!("{:.2} years", seconds / year)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_flops_uses_six_times_params_times_tokens() {
        let flops = training_flops(1.0e9, 2.0e9).unwrap();
        assert!((flops - 12.0e18).abs() < 1.0);
    }

    #[test]
    fn estimates_seconds_from_gpu_counts_and_utilization() {
        let seconds = estimate_seconds(1.0e9, 1.0e9, 50.0, &[1, 0, 0, 0, 0, 0]).unwrap();
        let expected = 6.0e18 / (989.0e12 * 0.5);
        assert!((seconds - expected).abs() < 0.001);
    }

    #[test]
    fn rejects_empty_gpu_selection() {
        assert_eq!(
            estimate_seconds(1.0e9, 1.0e9, 50.0, &[0, 0, 0, 0, 0, 0]),
            Err(EstimateError::NoGpuThroughput)
        );
    }
}
