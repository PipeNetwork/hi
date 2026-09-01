pub struct Backoff {
    base_ms: u64,
    cap_ms: u64,
}

impl Backoff {
    pub fn new(base_ms: u64, cap_ms: u64) -> Self {
        Self { base_ms, cap_ms }
    }

    pub fn delay_ms(&self, attempt: u32) -> u64 {
        let multiplier = 1u64.checked_shl(attempt).unwrap_or(u64::MAX);
        self.base_ms.saturating_mul(multiplier).min(self.cap_ms)
    }
}
