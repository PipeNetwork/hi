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
pub(crate) fn pages_for_tokens(token_count: usize, page_size: usize) -> usize {
    if page_size == 0 {
        return 0;
    }
    token_count.div_ceil(page_size)
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
}
