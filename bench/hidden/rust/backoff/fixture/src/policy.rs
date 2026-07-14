pub struct Backoff {
    base_ms: u64,
    cap_ms: u64,
}

impl Backoff {
    pub fn new(base_ms: u64, cap_ms: u64) -> Self {
        Self { base_ms, cap_ms }
    }

    pub fn delay_ms(&self, attempt: u32) -> u64 {
        self.base_ms * u64::from(attempt)
    }
}
