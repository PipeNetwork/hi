use super::deepseek_schema_recovery_enabled;

fn routing_with_capability(
    compat: hi_ai::DeepSeekCompat,
    provider: Option<&str>,
    capability: Option<&str>,
    model: &str,
) -> crate::config::AgentRouting {
    crate::config::AgentRouting {
        provider_route: provider.map(str::to_string),
        capability_route: capability.map(str::to_string),
        model: model.to_string(),
        deepseek_compat: compat,
        ..crate::config::AgentRouting::default()
    }
}

fn routing(
    compat: hi_ai::DeepSeekCompat,
    provider: Option<&str>,
    model: &str,
) -> crate::config::AgentRouting {
    routing_with_capability(compat, provider, None, model)
}

#[test]
fn deepseek_schema_recovery_matches_the_provider_auto_identity_rules() {
    assert!(deepseek_schema_recovery_enabled(&routing(
        hi_ai::DeepSeekCompat::On,
        Some("openai"),
        "custom-alias",
    )));
    assert!(!deepseek_schema_recovery_enabled(&routing(
        hi_ai::DeepSeekCompat::Off,
        Some("deepseek"),
        "deepseek-v4-flash",
    )));
    assert!(deepseek_schema_recovery_enabled(&routing(
        hi_ai::DeepSeekCompat::Auto,
        Some("deepseek"),
        "custom-alias",
    )));
    assert!(deepseek_schema_recovery_enabled(&routing(
        hi_ai::DeepSeekCompat::Auto,
        Some("openai"),
        "DeepSeek_V4_Pro_0813",
    )));
    assert!(!deepseek_schema_recovery_enabled(&routing(
        hi_ai::DeepSeekCompat::Auto,
        Some("openai"),
        "DeepSeek-Coder-V2-Lite",
    )));
    assert!(!deepseek_schema_recovery_enabled(&routing(
        hi_ai::DeepSeekCompat::Auto,
        Some("not-deepseek"),
        "generic-model",
    )));
    assert!(deepseek_schema_recovery_enabled(&routing_with_capability(
        hi_ai::DeepSeekCompat::Auto,
        Some("openai"),
        Some("deepseek@endpoint:blake3:opaque"),
        "custom-alias",
    )));
    assert!(!deepseek_schema_recovery_enabled(&routing_with_capability(
        hi_ai::DeepSeekCompat::Off,
        Some("openai"),
        Some("deepseek@endpoint:blake3:opaque"),
        "custom-alias",
    )));
    assert!(!deepseek_schema_recovery_enabled(&routing_with_capability(
        hi_ai::DeepSeekCompat::Auto,
        Some("openai"),
        Some("not-deepseek@endpoint:blake3:opaque"),
        "custom-alias",
    )));
}
