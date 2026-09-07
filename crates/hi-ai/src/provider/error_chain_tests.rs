use super::*;

#[test]
fn provider_contract_survives_custom_error_source_wrappers() {
    #[derive(Debug)]
    struct Receipt(anyhow::Error);
    impl std::fmt::Display for Receipt {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("settled failure")
        }
    }
    impl std::error::Error for Receipt {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            Some(self.0.as_ref())
        }
    }
    let usage = Usage {
        input_tokens: 12,
        output_tokens: 7,
        ..Default::default()
    };
    let original: anyhow::Error = ProviderError::new(
        ProviderErrorKind::RateLimit,
        "max_tokens must be less than or equal to 512",
    )
    .with_api_contract(Some("quota".into()), Some(false), Some(13))
    .with_usage(usage)
    .into();
    let output_cap = provider_output_cap_error(&original);
    assert!(output_cap.is_some());
    let wrapped: anyhow::Error = Receipt(original).into();
    assert_eq!(
        provider_error_kind(&wrapped),
        Some(ProviderErrorKind::RateLimit)
    );
    assert_eq!(provider_error_retryable(&wrapped), Some(false));
    assert_eq!(provider_retry_after_seconds(&wrapped), Some(13));
    assert_eq!(provider_error_usage(&wrapped).input_tokens, 12);
    assert_eq!(provider_error_usage(&wrapped).output_tokens, 7);
    assert_eq!(
        provider_error_details(&wrapped).unwrap().code.as_deref(),
        Some("quota")
    );
    assert_eq!(provider_output_cap_error(&wrapped), output_cap);
}
