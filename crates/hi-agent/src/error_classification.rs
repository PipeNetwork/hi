pub fn error_counts_as_model_issue(err: &anyhow::Error) -> bool {
    if err
        .downcast_ref::<hi_workspace::AdmissionDenied>()
        .is_some()
    {
        return false;
    }
    !matches!(
        hi_ai::provider_error_kind(err),
        Some(
            hi_ai::ProviderErrorKind::CapacityUnavailable
                | hi_ai::ProviderErrorKind::ModelUnavailable
                | hi_ai::ProviderErrorKind::Outage
                | hi_ai::ProviderErrorKind::QualityRejected
                | hi_ai::ProviderErrorKind::ToolProtocol
                | hi_ai::ProviderErrorKind::PaymentRequired
        )
    )
}
