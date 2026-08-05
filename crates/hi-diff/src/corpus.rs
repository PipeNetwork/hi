use std::collections::BTreeMap;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::{CaseId, LocalCase};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MutationKind {
    InsertToken,
    RemoveToken,
    ChangeToken,
    ChangeDecodeSteps,
    ChangeSeed,
}

pub struct CaseGenerator {
    seed: u64,
    rng: StdRng,
    next: u64,
    vocab_size: u32,
    max_len: usize,
}

impl CaseGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            rng: StdRng::seed_from_u64(seed),
            next: 0,
            vocab_size: 32_000,
            max_len: 128,
        }
    }

    pub fn with_limits(mut self, vocab_size: u32, max_len: usize) -> Self {
        self.vocab_size = vocab_size.max(2);
        self.max_len = max_len.max(1);
        self
    }

    pub fn next_case(&mut self) -> LocalCase {
        let length = self.rng.gen_range(1..=self.max_len);
        let input_tokens = (0..length)
            .map(|_| self.rng.gen_range(0..self.vocab_size))
            .collect();
        let id = format!("generated-{}-{}", self.seed, self.next);
        self.next += 1;
        LocalCase {
            id,
            input_tokens,
            decode_steps: self.rng.gen_range(0..=8),
            seed: self.rng.r#gen(),
            metadata: BTreeMap::new(),
        }
    }
}

pub fn shrink_local_case<F>(original: &LocalCase, mut reproduces: F) -> LocalCase
where
    F: FnMut(&LocalCase) -> bool,
{
    let mut current = original.clone();
    let mut changed = true;
    while changed {
        changed = false;
        if current.input_tokens.len() > 1 {
            for end in (1..current.input_tokens.len()).rev() {
                let mut candidate = current.clone();
                candidate.input_tokens.truncate(end);
                candidate.id = CaseId::from(format!("{}-shrink-{end}", original.id));
                if reproduces(&candidate) {
                    current = candidate;
                    changed = true;
                    break;
                }
            }
        }
        if current.decode_steps > 0 {
            let mut candidate = current.clone();
            candidate.decode_steps -= 1;
            candidate.id = format!("{}-shrink-steps", original.id);
            if reproduces(&candidate) {
                current = candidate;
                changed = true;
            }
        }
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_is_reproducible() {
        let mut a = CaseGenerator::new(42).with_limits(100, 8);
        let mut b = CaseGenerator::new(42).with_limits(100, 8);
        assert_eq!(a.next_case().input_tokens, b.next_case().input_tokens);
        assert_eq!(a.next_case().decode_steps, b.next_case().decode_steps);
    }

    #[test]
    fn shrink_keeps_reproducing_prefix() {
        let original = LocalCase {
            id: "x".into(),
            input_tokens: vec![1, 2, 3, 4],
            decode_steps: 3,
            seed: 1,
            metadata: BTreeMap::new(),
        };
        let shrunk = shrink_local_case(&original, |case| {
            case.input_tokens.len() >= 2 && case.decode_steps >= 1
        });
        assert_eq!(shrunk.input_tokens.len(), 2);
        assert_eq!(shrunk.decode_steps, 1);
    }
}
