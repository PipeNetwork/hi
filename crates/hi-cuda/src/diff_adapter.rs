//! Adapters from the existing local engines to the normalized Diff Lab core.

use anyhow::Result;
use hi_diff::{
    BackendKind, Checkpoint, CheckpointCapabilities, CheckpointRecord, CheckpointSink, LocalCase,
    LocalImplementation, LocalOutcome, ProbeLevel, TensorSummary,
};
use std::collections::BTreeMap;

use crate::qwen_cpu::{QwenCpuReference, QwenCpuRunOptions};

const LOGIT_CHECKPOINTS: &[&str] = &["logits"];

fn options(case: &LocalCase, probe: ProbeLevel) -> QwenCpuRunOptions {
    QwenCpuRunOptions {
        max_tokens: case.decode_steps,
        seed: Some(case.seed),
        include_logits: !matches!(probe, ProbeLevel::Summary),
        ..QwenCpuRunOptions::default()
    }
}

fn finish(
    output: crate::qwen_cpu::QwenCpuRunOutput,
    sink: &mut dyn CheckpointSink,
    probe: ProbeLevel,
) -> Result<LocalOutcome> {
    let mut checkpoints = Vec::new();
    if !matches!(probe, ProbeLevel::Summary)
        && let Some(logits) = output.logits.as_deref()
    {
        let summary = TensorSummary::from_values(vec![logits.len()], logits);
        sink.checkpoint(Checkpoint {
            name: "logits",
            step: 0,
            shape: vec![logits.len()],
            values: Some(logits),
            summary: summary.clone(),
            artifact: None,
        })?;
        checkpoints.push(CheckpointRecord {
            name: "logits".into(),
            step: 0,
            summary,
            artifact: None,
        });
    }
    let checkpoint_values = output
        .logits
        .as_ref()
        .map(|logits| BTreeMap::from([(String::from("logits"), logits.clone())]))
        .unwrap_or_default();
    Ok(LocalOutcome {
        generated_tokens: output.generated_tokens,
        next_token: Some(output.next_token),
        logits: output.logits,
        checkpoints,
        checkpoint_values,
    })
}

impl LocalImplementation for QwenCpuReference {
    fn metadata(&self) -> hi_diff::ImplementationMetadata {
        hi_diff::ImplementationMetadata::new("cpu-reference", BackendKind::Cpu)
    }

    fn capabilities(&self) -> CheckpointCapabilities {
        CheckpointCapabilities {
            checkpoints: LOGIT_CHECKPOINTS
                .iter()
                .map(|name| (*name).to_string())
                .collect(),
            supports_full_values: true,
        }
    }

    fn run_case(
        &self,
        case: &LocalCase,
        probe: ProbeLevel,
        sink: &mut dyn CheckpointSink,
    ) -> Result<LocalOutcome> {
        finish(
            self.run_tokens(&case.input_tokens, options(case, probe))?,
            sink,
            probe,
        )
    }
}

#[cfg(feature = "native-cuda")]
pub struct CudaTarget {
    pub engine: std::sync::Mutex<crate::dsv4_gpu::DeepSeekV4GpuEngine>,
}

#[cfg(feature = "native-cuda")]
impl CudaTarget {
    pub fn new(engine: crate::dsv4_gpu::DeepSeekV4GpuEngine) -> Self {
        Self {
            engine: std::sync::Mutex::new(engine),
        }
    }
}

#[cfg(feature = "native-cuda")]
impl LocalImplementation for CudaTarget {
    fn metadata(&self) -> hi_diff::ImplementationMetadata {
        hi_diff::ImplementationMetadata::new("cuda-dsv4", BackendKind::Cuda)
    }

    fn capabilities(&self) -> CheckpointCapabilities {
        CheckpointCapabilities {
            checkpoints: LOGIT_CHECKPOINTS
                .iter()
                .map(|name| (*name).to_string())
                .collect(),
            supports_full_values: true,
        }
    }

    fn run_case(
        &self,
        case: &LocalCase,
        probe: ProbeLevel,
        sink: &mut dyn CheckpointSink,
    ) -> Result<LocalOutcome> {
        let engine = self
            .engine
            .lock()
            .map_err(|_| anyhow::anyhow!("CUDA engine mutex poisoned"))?;
        finish(
            engine.run_tokens(&case.input_tokens, options(case, probe))?,
            sink,
            probe,
        )
    }
}
