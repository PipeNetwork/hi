use super::*;

#[test]
fn models_cache_key_does_not_retain_endpoint_credentials() {
    let endpoint = "https://wire-user:wire-pass@example.invalid/v1?api_key=query-secret";
    let key = cache_key("openai", endpoint);
    assert_eq!(key, crate::endpoint_capability_route("openai", endpoint));
    for secret in ["wire-user", "wire-pass", "query-secret", "example.invalid"] {
        assert!(!key.contains(secret), "leaked {secret}: {key}");
    }
}

#[tokio::test]
async fn models_transport_error_does_not_retain_endpoint_credentials() {
    let endpoint = "https://wire-user:wire-pass@?api_key=query-secret";
    let error = fetch_models(agent_http_client_quick().get(endpoint))
        .await
        .unwrap_err();
    assert_eq!(format!("{error:#}"), "model discovery transport failed");
}

#[tokio::test]
async fn models_http_auth_status_remains_classifiable() {
    let Some(server) =
        crate::test_support::FakeOpenAiServer::new(vec![crate::test_support::Response::json(
            401,
            r#"{"error":"bad key"}"#,
        )])
    else {
        return;
    };
    let error = fetch_models(agent_http_client_quick().get(server.url()))
        .await
        .unwrap_err();
    assert!(crate::is_http_auth_rejection(&error), "{error:#}");
}
