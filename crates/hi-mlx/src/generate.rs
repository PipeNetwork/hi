use anyhow::{Result, anyhow};

#[derive(Clone, Debug)]
pub struct TokenizerRuntime {
    inner: tokenizers::Tokenizer,
}

impl TokenizerRuntime {
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path.as_ref().join("tokenizer.json"))
            .map_err(|err| anyhow!("loading tokenizer.json: {err}"))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|err| anyhow!("tokenizing prompt: {err}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.inner
            .decode(tokens, true)
            .map_err(|err| anyhow!("decoding generated tokens: {err}"))
    }

    pub fn token_count(&self, text: &str) -> Result<u64> {
        Ok(self.encode(text)?.len() as u64)
    }
}

#[derive(Clone, Debug)]
pub struct LogitsProcessor {
    temperature: f32,
    top_p: f32,
    repetition_penalty: f32,
    rng: Lcg,
}

impl LogitsProcessor {
    pub fn new(temperature: f32, top_p: f32, repetition_penalty: f32, seed: u64) -> Self {
        Self {
            temperature: temperature.max(0.0),
            top_p: top_p.clamp(0.0, 1.0),
            repetition_penalty: repetition_penalty.max(0.0),
            rng: Lcg::new(seed),
        }
    }

    pub fn sample(&mut self, logits: &[f32], previous_tokens: &[u32]) -> Option<u32> {
        if logits.is_empty() {
            return None;
        }
        if self.temperature <= f32::EPSILON {
            return argmax(logits).map(|idx| idx as u32);
        }
        let mut adjusted = logits.to_vec();
        if (self.repetition_penalty - 1.0).abs() > f32::EPSILON {
            for token in previous_tokens {
                if let Some(value) = adjusted.get_mut(*token as usize) {
                    if *value < 0.0 {
                        *value *= self.repetition_penalty;
                    } else {
                        *value /= self.repetition_penalty;
                    }
                }
            }
        }
        for value in &mut adjusted {
            *value /= self.temperature;
        }
        let max = adjusted.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probs = adjusted
            .iter()
            .enumerate()
            .map(|(idx, logit)| (idx as u32, (*logit - max).exp()))
            .collect::<Vec<_>>();
        let sum: f32 = probs.iter().map(|(_, p)| *p).sum();
        if !sum.is_finite() || sum <= 0.0 {
            return argmax(logits).map(|idx| idx as u32);
        }
        for (_, p) in &mut probs {
            *p /= sum;
        }
        probs.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        if self.top_p > 0.0 && self.top_p < 1.0 {
            let mut cumulative = 0.0;
            let mut keep = 0;
            for (_, prob) in &probs {
                cumulative += *prob;
                keep += 1;
                if cumulative >= self.top_p {
                    break;
                }
            }
            probs.truncate(keep.max(1));
        }
        let kept_sum: f32 = probs.iter().map(|(_, p)| *p).sum();
        let mut draw = self.rng.next_f32() * kept_sum;
        for (idx, prob) in probs {
            if draw <= prob {
                return Some(idx);
            }
            draw -= prob;
        }
        argmax(logits).map(|idx| idx as u32)
    }
}

pub fn hit_stop(tokens: &[u32], stop_tokens: &[u32]) -> bool {
    tokens
        .last()
        .is_some_and(|token| stop_tokens.iter().any(|stop| stop == token))
}

fn argmax(values: &[f32]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(idx, _)| idx)
}

#[derive(Clone, Debug)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9e37_79b9_7f4a_7c15,
        }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (self.state >> 40) as u32;
        (bits as f32) / ((1u32 << 24) as f32)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
pub mod mlx {
    use anyhow::Result;
    use mlx_rs::ops::indexing::IndexOp;
    use mlx_rs::{Array, transforms};

    use super::LogitsProcessor;

    pub fn last_token_logits(logits: &Array) -> Result<Vec<f32>> {
        let shape = logits.shape();
        let seq = shape[shape.len() - 2];
        let last = logits
            .index((.., seq - 1, ..))
            .reshape(&[-1])?
            .as_type::<f32>()?;
        transforms::eval([&last])?;
        Ok(last.as_slice::<f32>().to_vec())
    }

    pub fn sample_next_token(
        logits: &Array,
        processor: &mut LogitsProcessor,
        previous_tokens: &[u32],
    ) -> Result<Option<u32>> {
        Ok(processor.sample(&last_token_logits(logits)?, previous_tokens))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_sampling_uses_argmax() {
        let mut sampler = LogitsProcessor::new(0.0, 1.0, 1.0, 7);

        assert_eq!(sampler.sample(&[0.1, 3.0, 2.9], &[]), Some(1));
    }

    #[test]
    fn fixed_seed_sampling_is_stable() {
        let mut a = LogitsProcessor::new(0.8, 0.9, 1.0, 42);
        let mut b = LogitsProcessor::new(0.8, 0.9, 1.0, 42);
        let logits = [0.2, 0.1, 4.0, 3.0, 2.0];

        let left = (0..8)
            .filter_map(|_| a.sample(&logits, &[]))
            .collect::<Vec<_>>();
        let right = (0..8)
            .filter_map(|_| b.sample(&logits, &[]))
            .collect::<Vec<_>>();

        assert_eq!(left, right);
    }

    #[test]
    fn detects_stop_token() {
        assert!(hit_stop(&[1, 2, 3], &[3, 4]));
        assert!(!hit_stop(&[1, 2], &[3, 4]));
    }
}
