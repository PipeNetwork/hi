//! Process-local optional request-shape capabilities learned from accepted
//! retries. Tools and caller-selected strict behavior are never degraded here.

use std::collections::HashMap;
use std::sync::Mutex;

use super::request::{
    RequestAttempt, is_deepseek_strict_schema_text, is_unsupported_frequency_penalty_text,
    is_unsupported_output_token_text,
};
use crate::ProviderErrorKind;
use crate::{ChatRequest, CompatMode, OutputTokenParameter};

#[derive(Clone, Copy, Default)]
struct OptionalFields {
    omit_usage: bool,
    omit_frequency_penalty: bool,
}

#[derive(Default)]
pub(super) struct CompatibilityCache(Mutex<HashMap<String, OptionalFields>>);

/// Given the attempt that just failed (at `current`) and its error, the index of
/// the next attempt to try — the one whose degradation actually addresses this
/// error — or `None` to stop and surface the error. Tool rejection is never
/// downgraded to chat-only: a coding-agent turn that advertised tools cannot
/// reliably complete after losing workspace access.
pub(crate) fn next_degraded_attempt(
    attempts: &[RequestAttempt],
    current: usize,
    kind: ProviderErrorKind,
    text: &str,
) -> Option<usize> {
    let cur = attempts[current];
    let after = current + 1;
    if cur.strict_fallback {
        return None;
    }
    if cur.output_token_parameter != OutputTokenParameter::Auto
        && !cur.output_token_fallback
        && is_unsupported_output_token_text(text)
    {
        return attempts[after..]
            .iter()
            .position(|a| {
                a.output_token_fallback
                    && a.include_usage == cur.include_usage
                    && a.include_tools == cur.include_tools
                    && a.include_frequency_penalty == cur.include_frequency_penalty
                    && a.strict_tools == cur.strict_tools
                    && a.deepseek_thinking == cur.deepseek_thinking
            })
            .map(|i| after + i);
    }
    if cur.strict_tools && is_deepseek_strict_schema_text(text) {
        return attempts[after..]
            .iter()
            // Preserve earlier degradations. For example, if the provider
            // already rejected `stream_options`, the strict-schema fallback
            // must not re-add it and pay for another avoidable 400.
            .position(|a| {
                !a.strict_tools
                    && a.include_tools
                    && a.include_usage == cur.include_usage
                    && a.include_frequency_penalty == cur.include_frequency_penalty
            })
            .map(|i| after + i);
    }
    // Usage streaming rejected → retry without it (keeping tools).
    if cur.include_usage
        && ["stream_options", "include_usage"]
            .iter()
            .any(|field| text.to_ascii_lowercase().contains(field))
    {
        return attempts[after..]
            .iter()
            .position(|a| {
                !a.include_usage
                    && a.include_frequency_penalty == cur.include_frequency_penalty
                    && a.strict_tools == cur.strict_tools
                    && a.deepseek_thinking == cur.deepseek_thinking
                    && a.output_token_parameter == cur.output_token_parameter
            })
            .map(|i| after + i);
    }
    // frequency_penalty rejected (xAI: "does not support parameter frequencyPenalty")
    // → retry without it. Keep tools and stream_options.
    if cur.include_frequency_penalty && is_unsupported_frequency_penalty_text(text) {
        return attempts[after..]
            .iter()
            .position(|a| {
                !a.include_frequency_penalty
                    && a.include_usage == cur.include_usage
                    && a.strict_tools == cur.strict_tools
                    && a.deepseek_thinking == cur.deepseek_thinking
                    && a.output_token_parameter == cur.output_token_parameter
            })
            .map(|i| after + i);
    }
    // Tool schema rejected → fail fast. Use `--tool-mode chat-only` for an
    // explicit no-tools request.
    if cur.include_tools
        && matches!(
            kind,
            ProviderErrorKind::UnsupportedTools | ProviderErrorKind::UnsupportedRequestShape
        )
    {
        return None;
    }
    // Provider/transport failures never justify mutating and replaying the
    // payload against the same route. The outer route/fallback policy may move
    // to another compatible backend when the typed error permits it.
    None
}

impl CompatibilityCache {
    pub(super) fn apply(&self, request: &mut ChatRequest) {
        if request.profile.compat != CompatMode::Auto {
            return;
        }
        let Some(fields) = self
            .0
            .lock()
            .ok()
            .and_then(|cache| cache.get(&request.model).copied())
        else {
            return;
        };
        if fields.omit_usage && request.profile.stream_usage.is_none() {
            request.profile.stream_usage = Some(false);
        }
        if fields.omit_frequency_penalty {
            request.frequency_penalty = None;
        }
    }

    pub(super) fn remember(&self, request: &ChatRequest, accepted: RequestAttempt) {
        if request.profile.compat != CompatMode::Auto {
            return;
        }
        let omit_usage = request.profile.stream_usage.is_none() && !accepted.include_usage;
        let omit_frequency_penalty =
            request.frequency_penalty.is_some() && !accepted.include_frequency_penalty;
        if (omit_usage || omit_frequency_penalty)
            && let Ok(mut cache) = self.0.lock()
        {
            let fields = cache.entry(request.model.clone()).or_default();
            fields.omit_usage |= omit_usage;
            fields.omit_frequency_penalty |= omit_frequency_penalty;
        }
    }
}
