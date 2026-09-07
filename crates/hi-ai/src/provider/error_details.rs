//! Provider contracts remain accessible through arbitrary error wrappers.

use super::*;

/// Typed provider details survive context and settlement error wrappers.
pub fn provider_error_details(err: &anyhow::Error) -> Option<&ProviderError> {
    err.downcast_ref::<ProviderError>().or_else(|| {
        err.chain()
            .find_map(|source| source.downcast_ref::<ProviderError>())
    })
}

pub fn provider_error_kind(err: &anyhow::Error) -> Option<ProviderErrorKind> {
    provider_error_details(err).map(|e| e.kind)
}

/// True when a models/probe request failed because the credential was refused
/// (HTTP 401/403), as opposed to a transport timeout the user might still save.
pub fn is_http_auth_rejection(err: &anyhow::Error) -> bool {
    if provider_error_kind(err) == Some(ProviderErrorKind::Auth) {
        return true;
    }
    let msg = err.to_string();
    msg.contains("returned 401") || msg.contains("returned 403")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputCapError {
    pub available_output_tokens: Option<u32>,
}

pub fn provider_output_cap_error(err: &anyhow::Error) -> Option<OutputCapError> {
    output_cap_error_from_text(&provider_error_text(err))
}

pub fn provider_retry_after_seconds(err: &anyhow::Error) -> Option<u64> {
    if let Some(error) = provider_error_details(err)
        && error.retry_after_seconds.is_some()
    {
        return error.retry_after_seconds;
    }
    retry_after_seconds_from_text(&provider_error_text(err))
}

pub fn provider_error_retryable(err: &anyhow::Error) -> Option<bool> {
    provider_error_details(err)
        .and_then(|error| error.retryable)
        .or_else(|| json_bool_field(&provider_error_text(err), "retryable"))
}

pub(super) fn provider_error_text(err: &anyhow::Error) -> String {
    provider_error_details(err)
        .map(|e| e.message.clone())
        .unwrap_or_else(|| err.to_string())
}

pub fn provider_error_usage(err: &anyhow::Error) -> Usage {
    provider_error_details(err)
        .map(|e| e.usage)
        .unwrap_or_default()
}
