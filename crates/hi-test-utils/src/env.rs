//! Environment-variable test knobs.

/// Parse a `usize` env knob, falling back to `default` when unset or
/// unparseable.
pub fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
