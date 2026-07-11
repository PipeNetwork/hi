//! Cross-request prefix KV cache — reuse decision and page bookkeeping.
//!
//! An agent loop resends the growing conversation each turn, so consecutive
//! requests share a long common token prefix (system prompt + tool schemas +
//! prior turns). Re-prefilling that prefix every request costs O(n) matmul and
//! O(n^2) attention. This module decides how much of a new request's prompt can
//! reuse a previously-computed prefix's KV, at whole-page granularity, and owns
//! the physical KV pages retained between requests.
//!
//! The pure decision logic here is engine-agnostic and unit-tested; the CUDA
//! integration (retaining pages out of the allocator, building a page table
//! that points the prefix at retained pages, and prefilling only the suffix
//! through the existing paged decode-append path) lives in `lib.rs`.

/// Length of the longest common prefix of two token sequences.
/// (Superseded by [`block_hash_chain`] matching on the live path; kept with its
/// tests as the readable specification of the reuse decision.)
#[cfg(test)]
pub(crate) fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// How many leading tokens of `new_tokens` may reuse a cached sequence's KV.
///
/// Reuse is whole-page only: a cached page holds `page_size` positions of KV, so
/// a page is reusable only if *every* position in it matches. We also always
/// leave at least one token for the model to actually process (otherwise there
/// are no logits to generate the next token from). The result is therefore a
/// multiple of `page_size`, in `[0, new_tokens.len())`.
#[cfg(test)]
pub(crate) fn reusable_prefix_tokens(
    new_tokens: &[u32],
    cached_tokens: &[u32],
    page_size: usize,
) -> usize {
    if page_size == 0 {
        return 0;
    }
    let lcp = common_prefix_len(new_tokens, cached_tokens);
    // Keep >= 1 new token to run through the model.
    let capped = lcp.min(new_tokens.len().saturating_sub(1));
    (capped / page_size) * page_size
}

/// Number of whole pages covering `token_count` positions (ceiling division).
#[cfg(test)]
pub(crate) fn pages_for_tokens(token_count: usize, page_size: usize) -> usize {
    if page_size == 0 {
        return 0;
    }
    token_count.div_ceil(page_size)
}

/// Chained per-page hashes over the FULL pages of a token stream (vLLM-style):
/// `hash[i]` covers tokens `[0, (i+1)*page_size)`, so equal hashes imply equal
/// whole-prefix content (up to 64-bit collision odds; FNV-1a over the token
/// bytes, chained so a page's hash commits to everything before it). A trailing
/// partial page contributes no hash — its KV can't be shared whole.
pub(crate) fn block_hash_chain(tokens: &[u32], page_size: usize) -> Vec<u64> {
    if page_size == 0 {
        return Vec::new();
    }
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let full_pages = tokens.len() / page_size;
    let mut hashes = Vec::with_capacity(full_pages);
    let mut hash = FNV_OFFSET;
    for page in 0..full_pages {
        for token in &tokens[page * page_size..(page + 1) * page_size] {
            for byte in token.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(FNV_PRIME);
            }
        }
        hashes.push(hash);
    }
    hashes
}

/// Refcounted block-hash index over retained KV pages — the multi-entry
/// replacement for the single-slot retained prefix. Sits ABOVE the free-list
/// allocator: pages it owns are, from the allocator's view, simply allocated;
/// eviction hands them back via the caller. Ownership invariant: every physical
/// page is owned by exactly one of {allocator free list, one lease's exclusive
/// tail, this index} — leases additionally *reference* index-owned pages as
/// their shared head, tracked by the per-entry refcount.
pub(crate) struct PrefixBlockIndex {
    entries: std::collections::HashMap<u64, BlockEntry>,
    tick: u64,
}

struct BlockEntry {
    page: usize,
    refcount: usize,
    last_use: u64,
}

impl PrefixBlockIndex {
    pub(crate) fn new() -> Self {
        Self {
            entries: std::collections::HashMap::new(),
            tick: 0,
        }
    }

    /// Pages for the longest prefix of `chain` present in the index, bumping each
    /// matched entry's refcount (the caller now references those pages).
    pub(crate) fn match_prefix(&mut self, chain: &[u64]) -> Vec<usize> {
        self.tick += 1;
        let mut pages = Vec::new();
        for hash in chain {
            let Some(entry) = self.entries.get_mut(hash) else {
                break;
            };
            entry.refcount += 1;
            entry.last_use = self.tick;
            pages.push(entry.page);
        }
        pages
    }

    /// Offer `(chain[i], pages[i])` pairs from a completed request. New hashes:
    /// the index takes ownership of the page. Hash present with the SAME page
    /// (the request's shared head — the index already owns it): no-op. Hash
    /// present with a DIFFERENT page (another request computed the same prefix
    /// independently): the offered page is a duplicate, returned for the caller
    /// to free. Entries start unreferenced; live sharers hold their own refs.
    pub(crate) fn insert_chain(&mut self, chain: &[u64], pages: &[usize]) -> Vec<usize> {
        self.tick += 1;
        let mut duplicates = Vec::new();
        for (hash, page) in chain.iter().zip(pages.iter()) {
            match self.entries.get_mut(hash) {
                None => {
                    self.entries.insert(
                        *hash,
                        BlockEntry {
                            page: *page,
                            refcount: 0,
                            last_use: self.tick,
                        },
                    );
                }
                Some(entry) if entry.page == *page => {
                    entry.last_use = self.tick;
                }
                Some(entry) => {
                    entry.last_use = self.tick;
                    duplicates.push(*page);
                }
            }
        }
        duplicates
    }

    /// Drop one reference on each of the first `count` entries of `chain`
    /// (the retiring request's shared head).
    pub(crate) fn release_prefix(&mut self, chain: &[u64], count: usize) {
        for hash in chain.iter().take(count) {
            if let Some(entry) = self.entries.get_mut(hash) {
                entry.refcount = entry.refcount.saturating_sub(1);
            }
        }
    }

    /// Evict up to `want` unreferenced entries, least-recently-used first,
    /// returning their pages for the caller to hand back to the allocator.
    pub(crate) fn evict_lru(&mut self, want: usize) -> Vec<usize> {
        if want == 0 {
            return Vec::new();
        }
        let mut candidates: Vec<(u64, u64)> = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.refcount == 0)
            .map(|(hash, entry)| (entry.last_use, *hash))
            .collect();
        candidates.sort_unstable();
        let mut pages = Vec::new();
        for (_, hash) in candidates.into_iter().take(want) {
            if let Some(entry) = self.entries.remove(&hash) {
                pages.push(entry.page);
            }
        }
        pages
    }

    pub(crate) fn entry_count(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn referenced_count(&self) -> usize {
        self.entries
            .values()
            .filter(|entry| entry.refcount > 0)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_prefix_len_counts_matching_head() {
        assert_eq!(common_prefix_len(&[1, 2, 3, 4], &[1, 2, 9, 4]), 2);
        assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 3]), 3);
        assert_eq!(common_prefix_len(&[], &[1]), 0);
        assert_eq!(common_prefix_len(&[9], &[1]), 0);
        // Divergence at the very first token.
        assert_eq!(common_prefix_len(&[5, 5, 5], &[6, 5, 5]), 0);
    }

    #[test]
    fn reuse_is_page_aligned_and_leaves_one_token() {
        let page = 16;
        // Identical 100-token prompts: reuse must leave >= 1 token, and round
        // down to a page boundary. min(100, 99) = 99 -> floor(99/16)*16 = 96.
        let a: Vec<u32> = (0..100).collect();
        assert_eq!(reusable_prefix_tokens(&a, &a, page), 96);

        // Shared prefix of 50 tokens, then divergence: floor(50/16)*16 = 48.
        let mut b = a.clone();
        b[50] = 9999;
        assert_eq!(reusable_prefix_tokens(&b, &a, page), 48);

        // Shared prefix shorter than one page -> no reuse.
        let mut c = a.clone();
        c[5] = 9999;
        assert_eq!(reusable_prefix_tokens(&c, &a, page), 0);
    }

    #[test]
    fn reuse_handles_growing_agent_conversation() {
        // Turn 1's prompt is a strict prefix of turn 2's (the loop appended the
        // tool call + result), so turn 2 reuses almost all of turn 1's KV.
        let page = 16;
        let turn1: Vec<u32> = (0..640).collect(); // 40 full pages
        let mut turn2 = turn1.clone();
        turn2.extend(700..760); // + 60 new suffix tokens
        // LCP = 640; capped by len-1 = 699 -> 640; page-aligned -> 640.
        assert_eq!(reusable_prefix_tokens(&turn2, &turn1, page), 640);
        // Only the 60-token suffix needs prefilling instead of all 700.
    }

    #[test]
    fn reuse_zero_when_no_cache_or_empty() {
        assert_eq!(reusable_prefix_tokens(&[1, 2, 3], &[], 16), 0);
        assert_eq!(reusable_prefix_tokens(&[], &[1, 2, 3], 16), 0);
        assert_eq!(reusable_prefix_tokens(&[1, 2, 3], &[1, 2, 3], 0), 0);
    }

    #[test]
    fn pages_for_tokens_is_ceiling() {
        assert_eq!(pages_for_tokens(0, 16), 0);
        assert_eq!(pages_for_tokens(1, 16), 1);
        assert_eq!(pages_for_tokens(16, 16), 1);
        assert_eq!(pages_for_tokens(17, 16), 2);
        assert_eq!(pages_for_tokens(96, 16), 6);
    }

    #[test]
    fn block_hash_chain_is_chained_and_full_pages_only() {
        let a: Vec<u32> = (0..40).collect();
        let chain = block_hash_chain(&a, 16);
        assert_eq!(chain.len(), 2); // 40 tokens = 2 full pages + partial

        // Same prefix -> same chain prefix; divergence in page 2 changes only
        // hashes from that page on.
        let mut b = a.clone();
        b[20] = 999;
        let chain_b = block_hash_chain(&b, 16);
        assert_eq!(chain[0], chain_b[0]);
        assert_ne!(chain[1], chain_b[1]);

        // Chaining: identical page-2 CONTENT after a different page 1 must hash
        // differently (the hash commits to everything before it).
        let mut c = a.clone();
        c[0] = 777;
        let chain_c = block_hash_chain(&c, 16);
        assert_ne!(chain[1], chain_c[1]);

        assert!(block_hash_chain(&a[..15], 16).is_empty());
        assert!(block_hash_chain(&a, 0).is_empty());
    }

    #[test]
    fn index_match_insert_release_evict_invariants() {
        let page_size = 4;
        let tokens: Vec<u32> = (0..16).collect(); // 4 full pages
        let chain = block_hash_chain(&tokens, page_size);
        let mut index = PrefixBlockIndex::new();

        // Empty index: no match.
        assert!(index.match_prefix(&chain).is_empty());

        // Insert 4 pages; all new, no duplicates returned.
        assert!(index.insert_chain(&chain, &[10, 11, 12, 13]).is_empty());
        assert_eq!(index.entry_count(), 4);
        assert_eq!(index.referenced_count(), 0);

        // Match the full chain: pages in order, refcounts bumped.
        assert_eq!(index.match_prefix(&chain), vec![10, 11, 12, 13]);
        assert_eq!(index.referenced_count(), 4);

        // A diverging chain matches only the shared prefix.
        let mut other = tokens.clone();
        other[9] = 999; // diverges in page 3
        let other_chain = block_hash_chain(&other, page_size);
        assert_eq!(index.match_prefix(&other_chain), vec![10, 11]);

        // Referenced entries never evict.
        assert!(index.evict_lru(10).is_empty());

        // Re-offering the index's own pages is a no-op; a different page for an
        // existing hash comes back as a duplicate to free.
        assert!(index.insert_chain(&chain[..2], &[10, 11]).is_empty());
        assert_eq!(index.insert_chain(&chain[..1], &[77]), vec![77]);
        assert_eq!(index.entry_count(), 4);

        // Release all references; eviction now drains LRU-first, each page
        // exactly once.
        index.release_prefix(&chain, 4);
        index.release_prefix(&other_chain, 2);
        assert_eq!(index.referenced_count(), 0);
        let mut evicted = index.evict_lru(2);
        evicted.extend(index.evict_lru(10));
        evicted.sort_unstable();
        assert_eq!(evicted, vec![10, 11, 12, 13]);
        assert_eq!(index.entry_count(), 0);
        assert!(index.match_prefix(&chain).is_empty());
    }

    #[test]
    fn index_partial_release_keeps_shared_head_referenced() {
        let page_size = 4;
        let tokens: Vec<u32> = (0..12).collect();
        let chain = block_hash_chain(&tokens, page_size);
        let mut index = PrefixBlockIndex::new();
        index.insert_chain(&chain, &[1, 2, 3]);
        // Two consumers of the first two pages, one of the third.
        index.match_prefix(&chain[..2]);
        index.match_prefix(&chain);
        index.release_prefix(&chain, 3); // second consumer retires fully
        assert_eq!(index.referenced_count(), 2);
        // Only the third page is evictable.
        assert_eq!(index.evict_lru(10), vec![3]);
        index.release_prefix(&chain, 2);
        let mut rest = index.evict_lru(10);
        rest.sort_unstable();
        assert_eq!(rest, vec![1, 2]);
    }
}
