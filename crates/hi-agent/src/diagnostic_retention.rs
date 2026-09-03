//! Memory-safe retention for diagnostic-only turn trails.
//!
//! Productive execution is intentionally allowed to continue without an
//! ordinary step ceiling. Diagnostic vectors therefore cannot grow one entry
//! per model/tool round forever. This log keeps an exact prefix (useful for
//! intent/first-action analysis) and an exact rolling suffix (useful for the
//! terminal diagnosis), while counting every omitted middle entry.

use std::ops::{Deref, DerefMut};

#[derive(Clone, Debug)]
pub(crate) struct BoundedDiagnosticLog<T, const LIMIT: usize, const HEAD: usize> {
    entries: Vec<T>,
    dropped: u64,
}

impl<T, const LIMIT: usize, const HEAD: usize> Default for BoundedDiagnosticLog<T, LIMIT, HEAD> {
    fn default() -> Self {
        debug_assert!(LIMIT > 0, "diagnostic retention needs a non-zero limit");
        debug_assert!(HEAD < LIMIT, "diagnostic prefix must leave suffix space");
        Self {
            entries: Vec::new(),
            dropped: 0,
        }
    }
}

impl<T, const LIMIT: usize, const HEAD: usize> BoundedDiagnosticLog<T, LIMIT, HEAD> {
    pub(crate) fn push(&mut self, entry: T) {
        if self.entries.len() < LIMIT {
            self.entries.push(entry);
            return;
        }

        // Preserve [0, HEAD) forever and roll the suffix. Removing at HEAD
        // discards the oldest unpinned middle/suffix record.
        self.entries.remove(HEAD);
        self.entries.push(entry);
        self.dropped = self.dropped.saturating_add(1);
    }

    pub(crate) fn dropped(&self) -> u64 {
        self.dropped
    }

    pub(crate) fn total(&self) -> u64 {
        (self.entries.len() as u64).saturating_add(self.dropped)
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        &self.entries
    }
}

impl<T, const LIMIT: usize, const HEAD: usize> Deref for BoundedDiagnosticLog<T, LIMIT, HEAD> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.entries
    }
}

impl<T, const LIMIT: usize, const HEAD: usize> DerefMut for BoundedDiagnosticLog<T, LIMIT, HEAD> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.entries
    }
}

/// Apply the same prefix+suffix retention to a public `Vec` field whose type
/// cannot be changed without breaking downstream callers.
pub(crate) fn push_bounded_vec<T>(
    entries: &mut Vec<T>,
    entry: T,
    dropped: &mut u64,
    limit: usize,
    head: usize,
) {
    debug_assert!(limit > 0);
    debug_assert!(head < limit);
    if entries.len() < limit {
        entries.push(entry);
        return;
    }
    entries.remove(head);
    entries.push(entry);
    *dropped = dropped.saturating_add(1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_log_keeps_exact_prefix_and_latest_suffix() {
        let mut log = BoundedDiagnosticLog::<u32, 6, 2>::default();
        for value in 0..10 {
            log.push(value);
        }

        assert_eq!(log.as_slice(), &[0, 1, 6, 7, 8, 9]);
        assert_eq!(log.dropped(), 4);
        assert_eq!(log.total(), 10);
    }

    #[test]
    fn bounded_vec_reports_omitted_middle_entries() {
        let mut entries = Vec::new();
        let mut dropped = 0;
        for value in 0..8 {
            push_bounded_vec(&mut entries, value, &mut dropped, 5, 1);
        }

        assert_eq!(entries, vec![0, 4, 5, 6, 7]);
        assert_eq!(dropped, 3);
    }
}
