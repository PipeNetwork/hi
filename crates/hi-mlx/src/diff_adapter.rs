//! MLX adapter for the normalized Diff Lab local target contract.
//!
//! MLX currently exposes token-level streaming through `NativeRuntime`; the
//! adapter uses that path for exact generated-token comparison. Intermediate
//! tensors remain explicitly unavailable until the native model tap API is
//! shared with the CUDA checkpoint registry.

use std::collections::BTreeMap;

use anyhow::{Result, anyhow};
use hi_diff::{
    BackendKind, CheckpointCapabilities, LocalCase, LocalImplementation, LocalOutcome, ProbeLevel,
};
use hi_local_core::backend::{GenerationEvent, GenerationRequest};

use crate::generate::TokenizerRuntime;
use crate::models::NativeRuntime;

pub struct MlxTarget {
    pub runtime: std::sync::Mutex<NativeRuntime>,
    pub tokenizer: TokenizerRuntime,
}

impl MlxTarget {
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        Ok(Self {
            runtime: std::sync::Mutex::new(NativeRuntime::from_path(path)?),
            tokenizer: TokenizerRuntime::load(path)?,
        })
    }
}

impl LocalImplementation for MlxTarget {
    fn metadata(&self) -> hi_diff::ImplementationMetadata {
        hi_diff::ImplementationMetadata::new("mlx", BackendKind::Mlx)
    }

    fn capabilities(&self) -> CheckpointCapabilities {
        CheckpointCapabilities {
            checkpoints: Vec::new(),
            supports_full_values: false,
        }
    }

    fn run_case(
        &self,
        case: &LocalCase,
        _probe: ProbeLevel,
        _sink: &mut dyn hi_diff::CheckpointSink,
    ) -> Result<LocalOutcome> {
        let prompt = self.tokenizer.decode(&case.input_tokens)?;
        let request = GenerationRequest {
            prompt,
            max_tokens: case.decode_steps as u32,
            temperature: 0.0,
            top_p: 1.0,
            top_k: None,
            seed: Some(case.seed),
            stop_sequences: Vec::new(),
            media_inputs: Vec::new(),
            messages: Vec::new(),
        };
        let mut runtime = self
            .runtime
            .lock()
            .map_err(|_| anyhow!("MLX runtime mutex poisoned"))?;
        let mut generated_tokens = Vec::new();
        let _output = runtime.stream_generate(request, |event| {
            if let GenerationEvent::TokenDelta { token_id, .. } = event {
                generated_tokens.push(token_id);
            }
            Ok(())
        })?;
        Ok(LocalOutcome {
            next_token: generated_tokens.first().copied(),
            generated_tokens,
            logits: None,
            checkpoints: Vec::new(),
            checkpoint_values: BTreeMap::new(),
        })
    }
}
