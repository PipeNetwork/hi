//! Bounded compatibility telemetry; this does not make retry decisions.

pub(super) const COMPAT_FALLBACK_LIMIT: usize = 64;
pub(super) const COMPAT_FALLBACK_PREFIX: usize = 62;
pub(super) const COMPAT_FALLBACK_OMITTED_PREFIX: &str = "[diagnostic truncation: ";

pub(super) fn record_compat_fallback(fallbacks: &mut Vec<String>, fallback: String) {
    if fallbacks.iter().any(|seen| seen == &fallback) {
        return;
    }
    if fallbacks.len() < COMPAT_FALLBACK_LIMIT {
        fallbacks.push(fallback);
        return;
    }

    let already_compacted = fallbacks
        .last()
        .and_then(|marker| marker.strip_prefix(COMPAT_FALLBACK_OMITTED_PREFIX))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|count| count.parse::<u64>().ok());
    let dropped = match already_compacted {
        Some(dropped) => {
            fallbacks[COMPAT_FALLBACK_PREFIX] = fallback;
            dropped.saturating_add(1)
        }
        None => {
            // The marker itself consumes one slot: retain the first 62 and the
            // newest event, and explicitly account for the two displaced rows.
            fallbacks.truncate(COMPAT_FALLBACK_PREFIX);
            fallbacks.push(fallback);
            fallbacks.push(String::new());
            2
        }
    };
    let last = fallbacks
        .last_mut()
        .expect("bounded compatibility trail always retains a marker slot");
    *last = format!(
        "{COMPAT_FALLBACK_OMITTED_PREFIX}{dropped} additional compatibility events omitted]"
    );
}

#[cfg(test)]
mod diagnostic_tests {
    use super::*;

    #[test]
    fn compatibility_fallbacks_are_deduplicated_and_bounded() {
        let mut fallbacks = Vec::new();
        for index in 0..100 {
            record_compat_fallback(&mut fallbacks, format!("fallback-{index}"));
        }
        record_compat_fallback(&mut fallbacks, "fallback-0".into());

        assert_eq!(fallbacks.len(), COMPAT_FALLBACK_LIMIT);
        assert_eq!(fallbacks.first().map(String::as_str), Some("fallback-0"));
        assert_eq!(
            fallbacks.get(COMPAT_FALLBACK_PREFIX).map(String::as_str),
            Some("fallback-99")
        );
        assert!(
            fallbacks
                .last()
                .is_some_and(|marker| marker.contains("37 additional compatibility events omitted")),
            "exact dropped count is surfaced in the bounded diagnostic: {fallbacks:?}"
        );
    }
}
