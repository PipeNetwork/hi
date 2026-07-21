//! GGUF tokenizer models and streaming decode.
//!
//! Extracted from `lib.rs` as a pure code move; all public items are
//! re-exported from the crate root so `hi_gguf::X` paths are unchanged.

use std::collections::{BTreeSet, HashMap};
use std::sync::OnceLock;

use anyhow::{Context, Result, anyhow, bail};
use serde::Serialize;

use crate::{GgufFile, ensure_token_id_in_range};

#[derive(Clone, Debug)]
pub struct GgufTokenizer {
    model: Option<String>,
    tokens: Vec<String>,
    token_to_id: HashMap<String, u32>,
    pub(crate) merge_ranks: HashMap<(String, String), usize>,
    /// Integer-keyed mirror of `merge_ranks` for the BPE hot loop: every string
    /// appearing on either side of a merge (and every merge result) is interned
    /// to a symbol id in `bpe_symbol_ids`, and `(left, right) -> (rank, merged)`
    /// lets the encoder test/apply merges without allocating or hashing strings.
    bpe_symbol_ids: HashMap<String, u32>,
    bpe_pair_merges: HashMap<(u32, u32), (u32, u32)>,
    scores: Option<Vec<f32>>,
    token_types: Option<Vec<i32>>,
    pub(crate) special_ids: BTreeSet<u32>,
    special_tokens: Vec<(String, u32)>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    unknown_token_id: Option<u32>,
    padding_token_id: Option<u32>,
    add_bos_token: bool,
    add_eos_token: bool,
}

impl GgufTokenizer {
    pub(crate) fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let model = gguf
            .metadata_string("tokenizer.ggml.model")
            .map(ToString::to_string);
        let tokens = gguf
            .metadata_string_array("tokenizer.ggml.tokens")?
            .ok_or_else(|| anyhow!("GGUF metadata missing tokenizer.ggml.tokens"))?;
        if tokens.is_empty() {
            bail!("GGUF tokenizer has no tokens");
        }

        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (idx, token) in tokens.iter().enumerate() {
            let id = u32::try_from(idx).context("GGUF tokenizer token count exceeds u32")?;
            if token_to_id.insert(token.clone(), id).is_some() {
                bail!("GGUF tokenizer contains duplicate token {token:?}");
            }
        }

        let merges = gguf
            .metadata_string_array("tokenizer.ggml.merges")?
            .unwrap_or_default();
        let mut merge_ranks = HashMap::with_capacity(merges.len());
        let mut bpe_symbol_ids: HashMap<String, u32> = HashMap::new();
        let mut bpe_pair_merges: HashMap<(u32, u32), (u32, u32)> = HashMap::new();
        for (rank, merge) in merges.iter().enumerate() {
            let (left, right) = merge
                .split_once(' ')
                .ok_or_else(|| anyhow!("invalid GGUF tokenizer merge {merge:?}"))?;
            if left.is_empty() || right.is_empty() {
                bail!("invalid GGUF tokenizer merge {merge:?}");
            }
            merge_ranks.insert((left.to_string(), right.to_string()), rank);
            let mut intern = |s: &str| -> Result<u32> {
                if let Some(id) = bpe_symbol_ids.get(s) {
                    return Ok(*id);
                }
                let id = u32::try_from(bpe_symbol_ids.len())
                    .context("GGUF tokenizer merge symbol count exceeds u32")?;
                bpe_symbol_ids.insert(s.to_string(), id);
                Ok(id)
            };
            let left_id = intern(left)?;
            let right_id = intern(right)?;
            let merged_id = intern(&format!("{left}{right}"))?;
            let rank = u32::try_from(rank).context("GGUF tokenizer merge rank exceeds u32")?;
            // Duplicate pairs keep last-wins semantics, matching `merge_ranks`.
            bpe_pair_merges.insert((left_id, right_id), (rank, merged_id));
        }

        let scores = gguf.metadata_f32_array("tokenizer.ggml.scores")?;
        if let Some(scores) = &scores
            && scores.len() != tokens.len()
        {
            bail!(
                "GGUF tokenizer score count {} does not match token count {}",
                scores.len(),
                tokens.len()
            );
        }
        let token_types = gguf.metadata_i32_array("tokenizer.ggml.token_type")?;
        if let Some(token_types) = &token_types
            && token_types.len() != tokens.len()
        {
            bail!(
                "GGUF tokenizer token_type count {} does not match token count {}",
                token_types.len(),
                tokens.len()
            );
        }

        let bos_token_id = gguf.metadata_u32("tokenizer.ggml.bos_token_id");
        let eos_token_id = gguf.metadata_u32("tokenizer.ggml.eos_token_id");
        let unknown_token_id = gguf.metadata_u32("tokenizer.ggml.unknown_token_id");
        let padding_token_id = gguf.metadata_u32("tokenizer.ggml.padding_token_id");
        let mut special_ids = BTreeSet::new();
        for key in [
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.unknown_token_id",
            "tokenizer.ggml.separator_token_id",
            "tokenizer.ggml.padding_token_id",
            "tokenizer.ggml.mask_token_id",
        ] {
            if let Some(id) = gguf.metadata_u32(key) {
                ensure_token_id_in_range(id, tokens.len(), key)?;
                special_ids.insert(id);
            }
        }
        if let Some(token_types) = &token_types {
            for (idx, token_type) in token_types.iter().enumerate() {
                if matches!(*token_type, 2..=5) {
                    special_ids.insert(idx as u32);
                }
            }
        }
        for (idx, token) in tokens.iter().enumerate() {
            if token.starts_with("<|") && token.ends_with("|>") {
                special_ids.insert(idx as u32);
            }
        }
        let mut special_tokens = special_ids
            .iter()
            .filter_map(|id| tokens.get(*id as usize).map(|token| (token.clone(), *id)))
            .filter(|(token, _)| !token.is_empty())
            .collect::<Vec<_>>();
        special_tokens.sort_by_key(|(token, _)| std::cmp::Reverse(token.len()));

        Ok(Self {
            model,
            tokens,
            token_to_id,
            merge_ranks,
            bpe_symbol_ids,
            bpe_pair_merges,
            scores,
            token_types,
            special_ids,
            special_tokens,
            bos_token_id,
            eos_token_id,
            unknown_token_id,
            padding_token_id,
            add_bos_token: gguf
                .metadata_bool("tokenizer.ggml.add_bos_token")
                .unwrap_or(false),
            add_eos_token: gguf
                .metadata_bool("tokenizer.ggml.add_eos_token")
                .unwrap_or(false),
        })
    }

    pub fn summary(&self) -> GgufTokenizerSummary {
        GgufTokenizerSummary {
            model: self.model.clone(),
            token_count: self.tokens.len(),
            merge_count: self.merge_ranks.len(),
            has_scores: self.scores.is_some(),
            has_token_types: self.token_types.is_some(),
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            unknown_token_id: self.unknown_token_id,
            padding_token_id: self.padding_token_id,
            add_bos_token: self.add_bos_token,
            add_eos_token: self.add_eos_token,
        }
    }

    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }

    pub fn token(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = self.encode_with_special_tokens(text)?;

        // Prepend BOS when the tokenizer is configured to, but never double it: some
        // chat templates emit the BOS token as leading text (Gemma `<bos>`, Llama-3
        // `<|begin_of_text|>`, the Zephyr/llama family prompt's `<s>`), which already
        // tokenizes to `bos_token_id` at position 0.
        if self.add_bos_token
            && let Some(id) = self.bos_token_id
            && ids.first() != Some(&id)
        {
            ids.insert(0, id);
        }

        if self.add_eos_token
            && let Some(id) = self.eos_token_id
        {
            ids.push(id);
        }
        Ok(ids)
    }

    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        self.decode_with_options(token_ids, true)
    }

    pub fn decode_with_options(&self, token_ids: &[u32], skip_special: bool) -> Result<String> {
        let mut text = String::new();
        for id in token_ids {
            ensure_token_id_in_range(*id, self.tokens.len(), "token id")?;
            if skip_special && self.special_ids.contains(id) {
                continue;
            }
            text.push_str(&self.tokens[*id as usize]);
        }

        if self.is_sentencepiece_unigram() || self.is_gemma4_escaped_bpe() {
            decode_sentencepiece_text(&text)
        } else if self.is_byte_level_bpe() {
            decode_byte_level_text(&text)
        } else {
            Ok(text.replace('\u{2581}', " "))
        }
    }

    /// Incremental detokenizer: `push` one token at a time and receive only the
    /// newly completed text, in O(token piece) per call instead of re-decoding the
    /// whole sequence. The concatenation of all pushed deltas plus `finish()` equals
    /// `decode(&all_ids)` byte-for-byte: multi-token UTF-8 sequences (byte-level BPE
    /// and `<0xXX>` byte-fallback) are held until complete, and a trailing incomplete
    /// character is dropped at `finish()` exactly like the batch decoder drops it.
    pub fn streaming_decoder(&self, skip_special: bool) -> StreamingTokenDecoder {
        let family = if self.is_sentencepiece_unigram() || self.is_gemma4_escaped_bpe() {
            StreamingDecodeFamily::SentencePiece
        } else if self.is_byte_level_bpe() {
            StreamingDecodeFamily::ByteLevel
        } else {
            StreamingDecodeFamily::Plain
        };
        StreamingTokenDecoder {
            skip_special,
            family,
            pending_bytes: Vec::new(),
            carry: String::new(),
        }
    }

    fn is_byte_level_bpe(&self) -> bool {
        self.model
            .as_deref()
            .is_some_and(|model| matches!(model, "gpt2" | "bpe" | "qwen2"))
            || !self.merge_ranks.is_empty()
    }

    fn encode_with_special_tokens(&self, text: &str) -> Result<Vec<u32>> {
        if self.special_tokens.is_empty() {
            return self.encode_plain(text);
        }
        let mut ids = Vec::new();
        let mut offset = 0usize;
        while offset < text.len() {
            let remaining = &text[offset..];
            let Some((relative_idx, token_len, token_id)) = self.find_next_special(remaining)
            else {
                ids.extend(self.encode_plain(remaining)?);
                break;
            };
            if relative_idx > 0 {
                ids.extend(self.encode_plain(&remaining[..relative_idx])?);
            }
            ids.push(token_id);
            offset = offset
                .checked_add(relative_idx)
                .and_then(|value| value.checked_add(token_len))
                .context("special token encoding offset overflows usize")?;
        }
        Ok(ids)
    }

    fn find_next_special(&self, text: &str) -> Option<(usize, usize, u32)> {
        let mut best: Option<(usize, usize, u32)> = None;
        for (token, id) in &self.special_tokens {
            let Some(index) = text.find(token) else {
                continue;
            };
            let candidate = (index, token.len(), *id);
            if best.is_none_or(|current| {
                candidate.0 < current.0 || (candidate.0 == current.0 && candidate.1 > current.1)
            }) {
                best = Some(candidate);
            }
        }
        best
    }

    fn encode_plain(&self, text: &str) -> Result<Vec<u32>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        if self.is_sentencepiece_unigram() {
            self.encode_sentencepiece_unigram(text)
        } else if self.is_gemma4_escaped_bpe() {
            self.encode_escaped_bpe(text)
        } else if self.is_byte_level_bpe() {
            self.encode_byte_level_bpe(text)
        } else {
            self.encode_greedy(text)
        }
    }

    /// Gemma-4's tokenizer (`tokenizer.ggml.model = "gemma4"`) is BPE over
    /// whitespace-escaped text (llama.cpp LLAMA_VOCAB_TYPE_BPE with
    /// `escape_whitespaces` and no space prefix), with sentencepiece-style
    /// `<0xXX>` byte fallback for characters the merges never cover.
    fn is_gemma4_escaped_bpe(&self) -> bool {
        self.model.as_deref() == Some("gemma4")
    }

    fn encode_escaped_bpe(&self, text: &str) -> Result<Vec<u32>> {
        let mut escaped = String::with_capacity(text.len() + 3);
        for ch in text.chars() {
            if ch == ' ' {
                escaped.push('\u{2581}');
            } else {
                escaped.push(ch);
            }
        }
        let mut ids = Vec::new();
        for symbol in self.apply_bpe(&escaped) {
            if let Some(id) = self.token_to_id.get(symbol).copied() {
                ids.push(id);
                continue;
            }
            for byte in symbol.bytes() {
                let name = format!("<0x{byte:02X}>");
                match self.token_to_id.get(&name).copied() {
                    Some(id) => ids.push(id),
                    None => match self.unknown_token_id {
                        Some(id) => ids.push(id),
                        None => bail!(
                            "gemma4 BPE produced token {symbol:?} with no byte fallback in GGUF vocab"
                        ),
                    },
                }
            }
        }
        Ok(ids)
    }

    fn is_sentencepiece_unigram(&self) -> bool {
        self.model
            .as_deref()
            .is_some_and(|model| matches!(model, "llama" | "spm"))
    }

    fn encode_sentencepiece_unigram(&self, text: &str) -> Result<Vec<u32>> {
        let normalized = sentencepiece_normalize(text);
        if normalized.is_empty() {
            return Ok(Vec::new());
        }

        #[derive(Clone)]
        struct Node {
            score: f32,
            next: usize,
            ids: Vec<u32>,
        }

        let mut dp: Vec<Option<Node>> = vec![None; normalized.len() + 1];
        dp[normalized.len()] = Some(Node {
            score: 0.0,
            next: normalized.len(),
            ids: Vec::new(),
        });
        let offsets = normalized
            .char_indices()
            .map(|(idx, _)| idx)
            .collect::<Vec<_>>();
        let scores = self.scores.as_ref();

        for offset in offsets.iter().rev().copied() {
            let remaining = &normalized[offset..];
            let mut best: Option<Node> = None;
            for (token, id) in &self.token_to_id {
                let id_usize = *id as usize;
                if self.special_ids.contains(id)
                    || byte_fallback_value(token).is_some()
                    || token.is_empty()
                    || !remaining.starts_with(token)
                {
                    continue;
                }
                let next = offset + token.len();
                let Some(tail) = dp.get(next).and_then(Option::as_ref) else {
                    continue;
                };
                let score_bias = scores
                    .and_then(|scores| scores.get(id_usize))
                    .copied()
                    .unwrap_or_default()
                    * 0.000_001;
                let score = tail.score - 1.0 + score_bias;
                if best.as_ref().is_none_or(|current| score > current.score) {
                    best = Some(Node {
                        score,
                        next,
                        ids: vec![*id],
                    });
                }
            }

            if let Some((ch, next)) = remaining
                .chars()
                .next()
                .map(|ch| (ch, offset + ch.len_utf8()))
            {
                let Some(tail) = dp.get(next).and_then(Option::as_ref) else {
                    continue;
                };
                let fallback_ids = self.sentencepiece_byte_fallback(ch)?;
                let score = tail.score - 100.0 * fallback_ids.len() as f32;
                if best.as_ref().is_none_or(|current| score > current.score) {
                    best = Some(Node {
                        score,
                        next,
                        ids: fallback_ids,
                    });
                }
            }
            dp[offset] = best;
        }

        let mut ids = Vec::new();
        let mut offset = 0usize;
        while offset < normalized.len() {
            let node = dp[offset].as_ref().ok_or_else(|| {
                anyhow!("GGUF llama tokenizer cannot encode text at byte offset {offset}")
            })?;
            ids.extend_from_slice(&node.ids);
            offset = node.next;
        }
        Ok(ids)
    }

    fn sentencepiece_byte_fallback(&self, ch: char) -> Result<Vec<u32>> {
        let mut buffer = [0; 4];
        let mut ids = Vec::new();
        for byte in ch.encode_utf8(&mut buffer).as_bytes() {
            let token = format!("<0x{byte:02X}>");
            match self.token_to_id.get(&token).copied() {
                Some(id) => ids.push(id),
                None => match self.unknown_token_id {
                    Some(id) => {
                        ids.push(id);
                        break;
                    }
                    None => bail!("GGUF llama tokenizer has no byte fallback token {token}"),
                },
            }
        }
        Ok(ids)
    }

    fn encode_byte_level_bpe(&self, text: &str) -> Result<Vec<u32>> {
        let encoded = encode_byte_level_text(text.as_bytes());
        let mut ids = Vec::new();
        for symbol in self.apply_bpe(&encoded) {
            match self.token_to_id.get(symbol).copied() {
                Some(id) => ids.push(id),
                None => match self.unknown_token_id {
                    Some(id) => ids.push(id),
                    None => bail!("BPE produced token {symbol:?} that is missing from GGUF vocab"),
                },
            }
        }
        Ok(ids)
    }

    /// BPE merge loop over the byte-level-encoded text, returning the final
    /// symbols as slices of `encoded` (merges are always adjacent, so every
    /// symbol is a contiguous substring).
    ///
    /// Semantics are exactly the naive algorithm's — repeatedly merge the
    /// lowest-rank pair present, leftmost first among equal ranks — but run as
    /// a binary heap over a doubly-linked symbol list with integer-interned
    /// symbols, so a pass is O(n log n) with no per-iteration allocation. The
    /// naive rescan (clone + hash two Strings per pair, per merge) was O(n^2)
    /// with heavy constants: ~2.7s for a 5k-token prompt, dominating TTFT.
    pub(crate) fn apply_bpe<'a>(&self, encoded: &'a str) -> Vec<&'a str> {
        struct Node {
            start: usize,
            end: usize,
            sym: u32,
            prev: i32,
            next: i32,
            alive: bool,
        }
        /// Min-heap entry: (rank, left position, merged symbol, left symbol,
        /// right symbol) under `Reverse` ordering.
        type MergeEntry = std::cmp::Reverse<(u32, u32, u32, u32, u32)>;

        // Interned ids >= this are per-call locals for chars that appear in no
        // merge rule (they can never merge, but must survive as symbols).
        let local_base = u32::try_from(self.bpe_symbol_ids.len()).unwrap_or(u32::MAX);
        let mut local_chars: HashMap<char, u32> = HashMap::new();
        let mut char_buf = [0u8; 4];
        let mut nodes: Vec<Node> = Vec::with_capacity(encoded.chars().count());
        for (offset, ch) in encoded.char_indices() {
            let sym = match self
                .bpe_symbol_ids
                .get(ch.encode_utf8(&mut char_buf) as &str)
            {
                Some(id) => *id,
                None => {
                    let next_local = local_base.saturating_add(local_chars.len() as u32);
                    *local_chars.entry(ch).or_insert(next_local)
                }
            };
            let idx = nodes.len() as i32;
            nodes.push(Node {
                start: offset,
                end: offset + ch.len_utf8(),
                sym,
                prev: idx - 1,
                next: idx + 1,
                alive: true,
            });
        }
        if let Some(last) = nodes.last_mut() {
            last.next = -1;
        }

        if nodes.len() >= 2 && !self.bpe_pair_merges.is_empty() {
            // Reverse-ordered min-heap entries: (rank, left position, merged
            // symbol, left symbol, right symbol). Position tiebreak = leftmost
            // among equal ranks; node indices are stable and order-preserving
            // (a merge keeps the left node), matching the naive scan order.
            let mut heap: std::collections::BinaryHeap<MergeEntry> =
                std::collections::BinaryHeap::with_capacity(nodes.len());
            for idx in 0..nodes.len() - 1 {
                let pair = (nodes[idx].sym, nodes[idx + 1].sym);
                if let Some((rank, merged)) = self.bpe_pair_merges.get(&pair) {
                    heap.push(std::cmp::Reverse((
                        *rank, idx as u32, *merged, pair.0, pair.1,
                    )));
                }
            }
            while let Some(std::cmp::Reverse((_, left_idx, merged, left_sym, right_sym))) =
                heap.pop()
            {
                let left_idx = left_idx as usize;
                // Stale entries: either node died, or a neighboring merge
                // changed the pair since this entry was pushed.
                if !nodes[left_idx].alive || nodes[left_idx].sym != left_sym {
                    continue;
                }
                let right_idx = nodes[left_idx].next;
                if right_idx < 0 {
                    continue;
                }
                let right_idx = right_idx as usize;
                if nodes[right_idx].sym != right_sym {
                    continue;
                }
                // Merge right into left.
                nodes[left_idx].end = nodes[right_idx].end;
                nodes[left_idx].sym = merged;
                nodes[right_idx].alive = false;
                let after = nodes[right_idx].next;
                nodes[left_idx].next = after;
                if after >= 0 {
                    nodes[after as usize].prev = left_idx as i32;
                }
                // New candidate pairs with both neighbors.
                let before = nodes[left_idx].prev;
                if before >= 0 {
                    let pair = (nodes[before as usize].sym, merged);
                    if let Some((rank, m)) = self.bpe_pair_merges.get(&pair) {
                        heap.push(std::cmp::Reverse((
                            *rank,
                            before as u32,
                            *m,
                            pair.0,
                            pair.1,
                        )));
                    }
                }
                if after >= 0 {
                    let pair = (merged, nodes[after as usize].sym);
                    if let Some((rank, m)) = self.bpe_pair_merges.get(&pair) {
                        heap.push(std::cmp::Reverse((
                            *rank,
                            left_idx as u32,
                            *m,
                            pair.0,
                            pair.1,
                        )));
                    }
                }
            }
        }

        let mut symbols = Vec::new();
        let mut cursor = if nodes.is_empty() { -1 } else { 0i32 };
        while cursor >= 0 {
            let node = &nodes[cursor as usize];
            symbols.push(&encoded[node.start..node.end]);
            cursor = node.next;
        }
        symbols
    }

    fn encode_greedy(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        let mut offset = 0usize;
        while offset < text.len() {
            let remaining = &text[offset..];
            let mut best: Option<(&str, u32)> = None;
            for (token, id) in &self.token_to_id {
                if remaining.starts_with(token)
                    && best.is_none_or(|(best_token, _)| token.len() > best_token.len())
                {
                    best = Some((token.as_str(), *id));
                }
            }

            let Some((token, id)) = best else {
                if let Some(id) = self.unknown_token_id {
                    ids.push(id);
                    offset += remaining
                        .chars()
                        .next()
                        .map(char::len_utf8)
                        .unwrap_or_default();
                    continue;
                }
                bail!("GGUF tokenizer cannot encode text at byte offset {offset}");
            };
            ids.push(id);
            offset += token.len();
        }
        Ok(ids)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct GgufTokenizerSummary {
    pub model: Option<String>,
    pub token_count: usize,
    pub merge_count: usize,
    pub has_scores: bool,
    pub has_token_types: bool,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub unknown_token_id: Option<u32>,
    pub padding_token_id: Option<u32>,
    pub add_bos_token: bool,
    pub add_eos_token: bool,
}
pub(crate) fn encode_byte_level_text(bytes: &[u8]) -> String {
    let encoder = byte_encoder();
    bytes.iter().map(|byte| encoder[*byte as usize]).collect()
}

/// Decode a byte buffer that a tokenizer reconstructed from byte-fallback or
/// byte-level tokens, without failing on incomplete UTF-8.
///
/// A multi-byte character is often emitted across several tokens, so a partial
/// tail is expected mid-stream (the server re-decodes the whole token list each
/// step and diffs for the delta) and also occurs when generation is cut off
/// mid-character at `max_tokens`. A trailing sequence that is a valid *prefix* of
/// a UTF-8 character is dropped — during streaming it reappears once its
/// continuation bytes arrive, and for a truncated final response the partial
/// character is correctly omitted. Genuinely invalid bytes become U+FFFD, matching
/// how reference detokenizers (llama.cpp, HF tokenizers) behave, rather than
/// failing the whole request.
pub(crate) fn decode_tokenizer_bytes_lenient(bytes: &[u8]) -> String {
    let mut out = String::new();
    let mut rest = bytes;
    loop {
        match std::str::from_utf8(rest) {
            Ok(valid) => {
                out.push_str(valid);
                break;
            }
            Err(error) => {
                let valid_up_to = error.valid_up_to();
                out.push_str(
                    std::str::from_utf8(&rest[..valid_up_to])
                        .expect("bytes up to valid_up_to are valid UTF-8"),
                );
                match error.error_len() {
                    // Genuinely invalid bytes: emit a replacement and skip them.
                    Some(invalid_len) => {
                        out.push('\u{FFFD}');
                        rest = &rest[valid_up_to + invalid_len..];
                    }
                    // Incomplete trailing sequence: drop it (completed by a later
                    // token during streaming, or cut off at max_tokens).
                    None => break,
                }
            }
        }
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StreamingDecodeFamily {
    ByteLevel,
    SentencePiece,
    Plain,
}

/// Incremental detokenizer state; see [`GgufTokenizer::streaming_decoder`]. Holds no
/// reference to the tokenizer so it can live inside long-lived request state; every
/// `push` must pass the same tokenizer the decoder was created from. Semantics match
/// the batch decoder exactly:
/// - Byte-level BPE: every token piece maps to raw bytes; the longest valid UTF-8
///   prefix is emitted and the incomplete tail is held for the next push
///   (`decode_tokenizer_bytes_lenient` is left-to-right prefix-stable, so incremental
///   emission with a held tail reproduces the batch output byte-for-byte).
/// - SentencePiece: `<0xXX>` byte-fallback runs accumulate bytes (across token
///   boundaries) and drain incrementally like the above; a regular character flushes
///   the run, dropping any incomplete tail — the same drop the batch decoder performs.
///   A piece suffix that could be a split byte-fallback marker is carried to the next
///   push so marker parsing never depends on token boundaries. `▁` becomes a space.
/// - Plain: pieces are emitted directly with `▁` replaced by a space.
pub struct StreamingTokenDecoder {
    skip_special: bool,
    family: StreamingDecodeFamily,
    // Bytes reconstructed from byte-level/byte-fallback tokens that do not yet end on
    // a UTF-8 character boundary.
    pending_bytes: Vec<u8>,
    // SentencePiece only: trailing piece characters that could be the prefix of a
    // `<0xXX>` marker split across token pieces.
    carry: String,
}

impl StreamingTokenDecoder {
    /// Feed one token; returns the newly completed text (possibly empty).
    /// `tokenizer` must be the tokenizer this decoder was created from.
    pub fn push(&mut self, tokenizer: &GgufTokenizer, token_id: u32) -> Result<String> {
        ensure_token_id_in_range(token_id, tokenizer.tokens.len(), "token id")?;
        if self.skip_special && tokenizer.special_ids.contains(&token_id) {
            return Ok(String::new());
        }
        let piece = tokenizer.tokens[token_id as usize].as_str();
        match self.family {
            StreamingDecodeFamily::Plain => Ok(piece.replace('\u{2581}', " ")),
            StreamingDecodeFamily::ByteLevel => {
                let decoder = byte_decoder();
                for ch in piece.chars() {
                    if let Some(byte) = decoder.get(&ch) {
                        self.pending_bytes.push(*byte);
                    } else {
                        let mut buffer = [0; 4];
                        self.pending_bytes
                            .extend_from_slice(ch.encode_utf8(&mut buffer).as_bytes());
                    }
                }
                Ok(self.drain_pending_valid_prefix())
            }
            StreamingDecodeFamily::SentencePiece => {
                let mut text = std::mem::take(&mut self.carry);
                text.push_str(piece);
                let mut out = String::new();
                let mut offset = 0usize;
                while offset < text.len() {
                    let remaining = &text[offset..];
                    if let Some(byte) = remaining.get(..6).and_then(byte_fallback_value) {
                        self.pending_bytes.push(byte);
                        out.push_str(&self.drain_pending_valid_prefix());
                        offset += 6;
                        continue;
                    }
                    // A short tail that is a strict prefix of "<0xXX>" may be completed
                    // by the next piece (the batch decoder parses the concatenation, so
                    // markers are boundary-blind); hold it instead of emitting.
                    if remaining.len() < 6 && could_prefix_byte_fallback(remaining) {
                        self.carry = remaining.to_string();
                        break;
                    }
                    // Regular character: terminates any byte-fallback run, dropping an
                    // incomplete tail exactly like the batch decoder's flush.
                    self.pending_bytes.clear();
                    let ch = remaining
                        .chars()
                        .next()
                        .expect("remaining string is non-empty");
                    out.push(if ch == '\u{2581}' { ' ' } else { ch });
                    offset += ch.len_utf8();
                }
                Ok(out)
            }
        }
    }

    /// End of stream: emit whatever remains. An incomplete UTF-8 tail is dropped
    /// (matching the batch decoder's behavior for output truncated mid-character);
    /// a held SentencePiece carry that never became a byte-fallback marker is
    /// emitted as regular text.
    pub fn finish(mut self) -> String {
        self.pending_bytes.clear();
        let carry = std::mem::take(&mut self.carry);
        if carry.is_empty() {
            String::new()
        } else {
            carry.replace('\u{2581}', " ")
        }
    }

    // Emit the longest valid UTF-8 prefix of `pending_bytes` (with U+FFFD for
    // definitively invalid sequences), holding an incomplete trailing sequence.
    fn drain_pending_valid_prefix(&mut self) -> String {
        let mut out = String::new();
        let mut consumed = 0usize;
        loop {
            let rest = &self.pending_bytes[consumed..];
            match std::str::from_utf8(rest) {
                Ok(valid) => {
                    if self.family == StreamingDecodeFamily::SentencePiece {
                        for ch in valid.chars() {
                            out.push(if ch == '\u{2581}' { ' ' } else { ch });
                        }
                    } else {
                        out.push_str(valid);
                    }
                    consumed = self.pending_bytes.len();
                    break;
                }
                Err(error) => {
                    let valid_up_to = error.valid_up_to();
                    let valid = std::str::from_utf8(&rest[..valid_up_to])
                        .expect("bytes up to valid_up_to are valid UTF-8");
                    if self.family == StreamingDecodeFamily::SentencePiece {
                        for ch in valid.chars() {
                            out.push(if ch == '\u{2581}' { ' ' } else { ch });
                        }
                    } else {
                        out.push_str(valid);
                    }
                    consumed += valid_up_to;
                    match error.error_len() {
                        Some(invalid_len) => {
                            out.push('\u{FFFD}');
                            consumed += invalid_len;
                        }
                        // Incomplete trailing sequence: hold for the next push.
                        None => break,
                    }
                }
            }
        }
        self.pending_bytes.drain(..consumed);
        out
    }
}

/// Is `text` a strict prefix of a `<0xXX>` byte-fallback marker (e.g. `<`, `<0`,
/// `<0xA`)? Used to hold a piece-spanning marker candidate until the next token.
fn could_prefix_byte_fallback(text: &str) -> bool {
    if text.is_empty() || text.len() >= 6 {
        return false;
    }
    let bytes = text.as_bytes();
    let pattern: [fn(u8) -> bool; 5] = [
        |b| b == b'<',
        |b| b == b'0',
        |b| b == b'x',
        |b| b.is_ascii_hexdigit(),
        |b| b.is_ascii_hexdigit(),
    ];
    bytes
        .iter()
        .zip(pattern.iter())
        .all(|(byte, check)| check(*byte))
}

pub(crate) fn decode_byte_level_text(text: &str) -> Result<String> {
    let decoder = byte_decoder();
    let mut bytes = Vec::with_capacity(text.len());
    for ch in text.chars() {
        if let Some(byte) = decoder.get(&ch) {
            bytes.push(*byte);
        } else {
            let mut buffer = [0; 4];
            bytes.extend_from_slice(ch.encode_utf8(&mut buffer).as_bytes());
        }
    }
    Ok(decode_tokenizer_bytes_lenient(&bytes))
}

fn sentencepiece_normalize(text: &str) -> String {
    let mut normalized = String::with_capacity(text.len() + 3);
    normalized.push('\u{2581}');
    for ch in text.chars() {
        if ch == ' ' {
            normalized.push('\u{2581}');
        } else {
            normalized.push(ch);
        }
    }
    normalized
}

pub(crate) fn decode_sentencepiece_text(text: &str) -> Result<String> {
    let mut out = String::new();
    let mut bytes = Vec::new();
    let mut offset = 0usize;
    while offset < text.len() {
        let remaining = &text[offset..];
        if let Some(byte) = remaining.get(..6).and_then(byte_fallback_value) {
            bytes.push(byte);
            offset += 6;
            continue;
        }
        if !bytes.is_empty() {
            out.push_str(&decode_tokenizer_bytes_lenient(&std::mem::take(&mut bytes)));
        }
        let ch = remaining
            .chars()
            .next()
            .expect("remaining string is non-empty");
        out.push(ch);
        offset += ch.len_utf8();
    }
    if !bytes.is_empty() {
        out.push_str(&decode_tokenizer_bytes_lenient(&bytes));
    }
    // The SentencePiece space marker ▁ (U+2581) denotes a space in detokenized
    // output, whether it came from a normal token piece or from byte-fallback
    // tokens (some models emit U+2581 itself via byte fallback).
    Ok(out.replace('\u{2581}', " "))
}

fn byte_fallback_value(token: &str) -> Option<u8> {
    let hex = token
        .strip_prefix("<0x")
        .and_then(|value| value.strip_suffix('>'))?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

fn byte_encoder() -> &'static [char; 256] {
    static ENCODER: OnceLock<[char; 256]> = OnceLock::new();
    ENCODER.get_or_init(|| {
        let mut bytes = Vec::new();
        bytes.extend(b'!'..=b'~');
        bytes.extend(0xA1..=0xAC);
        bytes.extend(0xAE..=0xFF);

        let mut codepoints = bytes
            .iter()
            .map(|byte| u32::from(*byte))
            .collect::<Vec<_>>();
        let mut extra = 0u32;
        for byte in 0u8..=u8::MAX {
            if !bytes.contains(&byte) {
                bytes.push(byte);
                codepoints.push(256 + extra);
                extra += 1;
            }
        }

        let mut encoder = ['\0'; 256];
        for (byte, codepoint) in bytes.into_iter().zip(codepoints) {
            encoder[byte as usize] = char::from_u32(codepoint).expect("valid GPT-2 byte mapping");
        }
        encoder
    })
}

fn byte_decoder() -> &'static HashMap<char, u8> {
    static DECODER: OnceLock<HashMap<char, u8>> = OnceLock::new();
    DECODER.get_or_init(|| {
        byte_encoder()
            .iter()
            .enumerate()
            .map(|(byte, ch)| (*ch, byte as u8))
            .collect()
    })
}
