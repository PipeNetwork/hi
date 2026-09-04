//! Retire old opaque content without manufacturing invalid provider blocks.

use hi_ai::Content;

const ELIDE_THINKING_MIN_CHARS: usize = 400;

pub(crate) fn repair_legacy_elided_thinking(block: &mut Content) {
    if matches!(block, Content::Thinking { text, .. } if text.starts_with("[elided thinking")) {
        elide_old_thinking_in(block);
    }
}

pub(super) fn elide_old_thinking_in(block: &mut Content) -> usize {
    let Content::Thinking { text, signature } = block else {
        return 0;
    };
    let chars = text.chars().count();
    // Also repair historical elisions produced before the complete signed
    // block was retired. Their signature no longer authenticates their text.
    if chars <= ELIDE_THINKING_MIN_CHARS && !text.starts_with("[elided thinking") {
        return 0;
    }
    let bytes = text.len() + signature.as_deref().map_or(0, str::len);
    let marker = if text.starts_with("[elided thinking") {
        text.clone()
    } else {
        format!("[elided thinking — was {chars} chars]")
    };
    let freed = bytes.saturating_sub(marker.len());
    // Signatures bind the original reasoning bytes. Keeping one while
    // replacing its text makes the next Anthropic request invalid and resends
    // an opaque, often large signature that no longer serves any purpose.
    *block = Content::Text(marker);
    freed
}

pub(super) fn elide_old_image(block: &mut Content) -> usize {
    let Content::Image { data, .. } = block else {
        return 0;
    };
    let n = data.chars().count();
    let freed = data.len();
    *block = Content::Text(format!("[elided image — was {n} chars]"));
    freed
}
