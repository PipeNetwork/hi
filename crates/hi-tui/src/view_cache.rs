//! Cached transcript flatten + wrap measurements.
//!
//! Spinner ticks and unrelated chrome redraws must not re-flatten thousands of
//! transcript entries. Structural changes (push, fold toggle, width, density,
//! block-nav marker) invalidate the cache; pure animation does not.

/// Flattened transcript ready for scroll/render, plus the indices sticky headers
/// and block-nav need.
#[derive(Clone, Default)]
pub(crate) struct TranscriptViewCache {
    /// Generation of `App.transcript_gen` this cache was built against.
    pub generation: u64,
    /// Theme revision used to style the flattened lines.
    pub theme_revision: u64,
    pub width: u16,
    pub show_reasoning: bool,
    pub show_tool_output: bool,
    pub density: crate::Density,
    /// Selected block ordinal when block-nav is on (marker line injected).
    pub nav_selected: Option<usize>,
    /// Pending streaming line fingerprint (len + last few chars) so live tokens
    /// rebuild without comparing full strings every time.
    pub pending_fp: u64,
    /// Exact number of lines compacted into the marker. The count is part of
    /// cache identity because the marker text changes while the boolean
    /// "marker is present" state remains true.
    pub trimmed: u64,

    pub lines: Vec<ratatui::text::Line<'static>>,
    /// `prefix[i]` = wrapped rows above flattened line `i`; `prefix[len]` = total.
    pub prefix: Vec<u32>,
    pub prompt_line_starts: Vec<usize>,
    /// `(line_start, line_end, block_ord)` for each tool-output block.
    pub block_line_ranges: Vec<(usize, usize, usize)>,
    /// How many `App.transcript` entries were folded into `lines` (excludes the
    /// live pending stream line). Used by the incremental append path.
    pub committed_entries: usize,
    /// Flattened line count corresponding to `committed_entries` (no pending).
    pub committed_flat_lines: usize,
}

/// The inputs that determine cached transcript geometry. Scroll position is
/// intentionally absent: scrolling changes which cached lines are painted but
/// does not change selection or sticky-prompt maps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ViewCacheKey {
    pub(crate) generation: u64,
    pub(crate) theme_revision: u64,
    pub(crate) width: u16,
    pub(crate) show_reasoning: bool,
    pub(crate) show_tool_output: bool,
    pub(crate) density: crate::Density,
    pub(crate) nav_selected: Option<usize>,
    pub(crate) pending_fp: u64,
    pub(crate) trimmed: u64,
}

impl TranscriptViewCache {
    pub(crate) fn key(&self) -> ViewCacheKey {
        ViewCacheKey {
            generation: self.generation,
            theme_revision: self.theme_revision,
            width: self.width,
            show_reasoning: self.show_reasoning,
            show_tool_output: self.show_tool_output,
            density: self.density,
            nav_selected: self.nav_selected,
            pending_fp: self.pending_fp,
            trimmed: self.trimmed,
        }
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "cache identity intentionally compares every render input without allocation"
    )]
    pub(crate) fn matches(
        &self,
        generation: u64,
        theme_revision: u64,
        width: u16,
        show_reasoning: bool,
        show_tool_output: bool,
        density: crate::Density,
        nav_selected: Option<usize>,
        pending_fp: u64,
        trimmed: u64,
    ) -> bool {
        self.generation == generation
            && self.theme_revision == theme_revision
            && self.width == width
            && self.show_reasoning == show_reasoning
            && self.show_tool_output == show_tool_output
            && self.density == density
            && self.nav_selected == nav_selected
            && self.pending_fp == pending_fp
            && self.trimmed == trimmed
            && !self.prefix.is_empty()
    }

    pub(crate) fn total_rows(&self) -> u16 {
        self.prefix
            .last()
            .copied()
            .unwrap_or(0)
            .min(u16::MAX as u32) as u16
    }
}

/// Fingerprint the in-progress streamed line so cache invalidates as tokens arrive
/// without storing the full pending string in the key.
pub(crate) fn pending_fingerprint(pending: &Option<(ratatui::style::Style, bool, String)>) -> u64 {
    let Some((_, md, text)) = pending else {
        return 0;
    };
    let mut h = text.len() as u64;
    h ^= u64::from(*md) << 32;
    // Mix in the tail so mid-line edits (rare) still bust the cache.
    for (i, b) in text.bytes().rev().take(16).enumerate() {
        h = h.wrapping_mul(131).wrapping_add(b as u64 + i as u64);
    }
    h
}

/// First flattened line index whose wrapped span intersects `row` (0-based from
/// the top of the full transcript). Clamps to the last line when `row` is past
/// the end.
pub(crate) fn line_index_at_row(prefix: &[u32], row: u32) -> usize {
    if prefix.len() < 2 {
        return 0;
    }
    let n = prefix.len() - 1;
    // prefix[i] <= row < prefix[i+1]
    let mut lo = 0usize;
    let mut hi = n;
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if prefix[mid] <= row {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    if prefix[lo] <= row { lo.min(n - 1) } else { 0 }
}

/// Visible line slice for a viewport starting at wrapped-row `scroll` with
/// `height` rows, plus overscan lines above/below. Returns
/// `(line_lo, line_hi_exclusive, scroll_within_slice)`.
pub(crate) fn visible_line_window(
    prefix: &[u32],
    scroll: u16,
    height: u16,
    overscan: usize,
) -> (usize, usize, u16) {
    if prefix.len() < 2 || height == 0 {
        return (0, 0, 0);
    }
    let n_lines = prefix.len() - 1;
    let total = *prefix.last().unwrap_or(&0);
    let start_row = scroll as u32;
    let end_row = (scroll as u32)
        .saturating_add(height as u32)
        .min(total.max(1));

    let mut lo = line_index_at_row(prefix, start_row);
    let mut hi = line_index_at_row(prefix, end_row.saturating_sub(1)) + 1;
    lo = lo.saturating_sub(overscan);
    hi = (hi + overscan).min(n_lines);
    if lo >= hi {
        hi = (lo + 1).min(n_lines);
    }
    let scroll_adj = start_row.saturating_sub(prefix[lo]) as u16;
    (lo, hi, scroll_adj)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_index_at_row_basic() {
        // lines of height 2, 1, 3 → prefix 0,2,3,6
        let prefix = vec![0, 2, 3, 6];
        assert_eq!(line_index_at_row(&prefix, 0), 0);
        assert_eq!(line_index_at_row(&prefix, 1), 0);
        assert_eq!(line_index_at_row(&prefix, 2), 1);
        assert_eq!(line_index_at_row(&prefix, 5), 2);
    }

    #[test]
    fn visible_window_with_overscan() {
        let prefix = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let (lo, hi, adj) = visible_line_window(&prefix, 3, 2, 1);
        assert_eq!(lo, 2); // line 3 - overscan 1
        assert_eq!(hi, 6); // lines 3,4 + overscan
        assert_eq!(adj, 1); // scroll 3, prefix[2]=2 → adj 1
    }

    #[test]
    fn theme_revision_is_part_of_cache_identity() {
        let mut cache = TranscriptViewCache {
            generation: 1,
            theme_revision: 7,
            width: 40,
            show_reasoning: false,
            show_tool_output: false,
            density: crate::Density::Comfortable,
            nav_selected: None,
            pending_fp: 0,
            trimmed: 0,
            lines: vec![ratatui::text::Line::raw("line")],
            prefix: vec![0, 1],
            ..Default::default()
        };
        assert!(cache.matches(
            1,
            7,
            40,
            false,
            false,
            crate::Density::Comfortable,
            None,
            0,
            0
        ));
        assert!(!cache.matches(
            1,
            8,
            40,
            false,
            false,
            crate::Density::Comfortable,
            None,
            0,
            0
        ));
        cache.theme_revision = 8;
        assert_eq!(cache.key().theme_revision, 8);
    }
}
