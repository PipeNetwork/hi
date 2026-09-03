use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use url::{Host, Url};

/// Non-secret provider routing needed to reproduce a live smoke run.
///
/// Credentials deliberately do not belong in this type. Replays always obtain
/// `HI_API_KEY` from the environment of the replaying process.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LiveRoute {
    pub provider: String,
    pub model: String,
    pub base_url: String,
}

impl LiveRoute {
    pub(crate) fn new(provider: &str, model: &str, base_url: &str) -> Result<Self> {
        let provider = provider.trim();
        let provider = if provider.is_empty() {
            "openai"
        } else {
            provider
        };
        ensure_metadata_value("live provider", provider)?;
        match provider {
            "openai" | "anthropic" | "pipenetwork" | "pipe" | "xai" | "ollama" | "local" => {}
            other => bail!(
                "unsupported live provider {other:?}; expected openai, anthropic, pipenetwork, pipe, xai, ollama, or local"
            ),
        }

        let model = model.trim();
        ensure_metadata_value("live model", model)?;

        let base_url = base_url.trim();
        ensure_metadata_value("live base URL", base_url)?;
        let parsed = Url::parse(base_url)
            .map_err(|error| anyhow::anyhow!("invalid live base URL {base_url:?}: {error}"))?;
        ensure!(
            matches!(parsed.scheme(), "http" | "https"),
            "live base URL must use http or https"
        );
        ensure!(
            parsed.username().is_empty() && parsed.password().is_none(),
            "live base URL must not contain credentials"
        );
        ensure!(
            parsed.query().is_none() && parsed.fragment().is_none(),
            "live base URL must not contain a query or fragment"
        );

        Ok(Self {
            provider: provider.to_owned(),
            model: model.to_owned(),
            base_url: normalize_base_url(base_url)?,
        })
    }

    /// Validate the credential-free transport evidence emitted by the full
    /// TUI against the exact live route selected by the harness.
    ///
    /// A rejected compatibility attempt is valid evidence when its HTTP
    /// status is non-successful. Conversely, an accepted attempt must carry a
    /// successful status. This preserves bounded provider-recovery scenarios
    /// while preventing a live case from passing against a different model or
    /// endpoint than the one recorded in its replay metadata.
    pub(crate) fn validate_provider_requests(&self, requests: &[Value]) -> Result<()> {
        for (index, request) in requests.iter().enumerate() {
            self.validate_provider_request(request).with_context(|| {
                format!("live provider evidence invariant failed at request {index}")
            })?;
        }
        ensure!(
            requests
                .iter()
                .any(|request| request.get("accepted").and_then(Value::as_bool) == Some(true)),
            "live provider evidence contained no accepted HTTP request"
        );
        Ok(())
    }

    fn validate_provider_request(&self, request: &Value) -> Result<()> {
        let audit = serde_json::from_value::<hi_ai::WireAudit>(request.clone())
            .context("provider_request did not contain a valid typed wire audit")?;
        ensure!(
            audit.request_body.is_none(),
            "provider_request unexpectedly retained a request body"
        );
        ensure!(
            audit.provider == self.expected_wire_provider(),
            "provider_request adapter mismatch: expected {:?}, got {:?}",
            self.expected_wire_provider(),
            audit.provider
        );
        ensure!(
            audit.model == self.model,
            "provider_request model mismatch: expected {:?}, got {:?}",
            self.model,
            audit.model
        );
        let audit_route = normalize_base_url(&audit.route)
            .context("provider_request route was not a valid base URL")?;
        ensure!(
            audit_route == self.base_url,
            "provider_request route mismatch: expected {:?}, got {:?}",
            self.base_url,
            audit_route
        );
        ensure!(
            audit.request_attempt > 0,
            "provider_request request_attempt must be positive"
        );
        let status = audit
            .response_status
            .context("provider_request did not record an HTTP response status")?;
        let status_was_accepted = (200..300).contains(&status);
        ensure!(
            audit.accepted == status_was_accepted,
            "provider_request acceptance/status mismatch: accepted={}, response_status={status}",
            audit.accepted
        );
        Ok(())
    }

    fn expected_wire_provider(&self) -> &'static str {
        match self.provider.as_str() {
            "xai" => "xai",
            "anthropic" => "anthropic",
            // Pipe, Ollama, and local endpoints all use the production
            // OpenAI-compatible adapter selected by hi's provider routing.
            _ => "openai_compatible",
        }
    }

    /// Defense in depth for accidentally embedding the separately supplied
    /// credential in a route field before that route is persisted.
    pub(crate) fn ensure_excludes_secret(&self, secret: &str) -> Result<()> {
        ensure!(!secret.is_empty(), "live API key must not be empty");
        for (name, value) in [
            ("provider", self.provider.as_str()),
            ("model", self.model.as_str()),
            ("base URL", self.base_url.as_str()),
        ] {
            ensure!(
                !value.contains(secret),
                "live {name} must not contain HI_API_KEY"
            );
        }
        Ok(())
    }
}

fn normalize_base_url(base_url: &str) -> Result<String> {
    let parsed = Url::parse(base_url)
        .map_err(|error| anyhow::anyhow!("invalid live base URL {base_url:?}: {error}"))?;
    ensure!(
        matches!(parsed.scheme(), "http" | "https"),
        "live base URL must use http or https"
    );
    ensure!(
        parsed.username().is_empty() && parsed.password().is_none(),
        "live base URL must not contain credentials"
    );
    ensure!(
        parsed.query().is_none() && parsed.fragment().is_none(),
        "live base URL must not contain a query or fragment"
    );
    ensure!(
        parsed.scheme() == "https" || (parsed.scheme() == "http" && is_loopback(&parsed)),
        "live base URL must use https unless it targets loopback"
    );
    Ok(parsed.as_str().trim_end_matches('/').to_owned())
}

fn is_loopback(url: &Url) -> bool {
    match url.host() {
        Some(Host::Ipv4(address)) => address.is_loopback(),
        Some(Host::Ipv6(address)) => address.is_loopback(),
        Some(Host::Domain(domain)) => {
            domain.eq_ignore_ascii_case("localhost")
                || domain
                    .to_ascii_lowercase()
                    .strip_suffix(".localhost")
                    .is_some_and(|prefix| !prefix.is_empty())
        }
        None => false,
    }
}

fn ensure_metadata_value(name: &str, value: &str) -> Result<()> {
    ensure!(!value.is_empty(), "{name} must not be empty");
    ensure!(
        !value.chars().any(char::is_control),
        "{name} must not contain control characters"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(route: &str, model: &str, accepted: bool, status: u16) -> Value {
        serde_json::to_value(hi_ai::WireAudit {
            provider: "openai_compatible".into(),
            route: route.into(),
            model: model.into(),
            output_token_parameter: "max_tokens".into(),
            max_output_tokens: 512,
            request_attempt: 1,
            accepted,
            response_status: Some(status),
            ..hi_ai::WireAudit::default()
        })
        .unwrap()
    }

    #[test]
    fn route_is_non_secret_and_replay_safe() {
        let route = LiveRoute::new(
            "pipenetwork",
            "pipe/deepseek-v4-flash-0731",
            "https://api.pipenetwork.ai/v1",
        )
        .unwrap();
        route.ensure_excludes_secret("not-present-secret").unwrap();
        assert!(route.ensure_excludes_secret("deepseek").is_err());
        assert!(LiveRoute::new("openai", "model", "https://key@example.test/v1").is_err());
        assert!(LiveRoute::new("openai", "model", "https://example.test/v1?key=x").is_err());
        assert!(LiveRoute::new("openai\n# injected", "model", "https://example.test/v1").is_err());
        assert!(LiveRoute::new("pipenetwork", "model", "http://api.pipenetwork.ai/v1").is_err());
        assert!(LiveRoute::new("openai", "model", "http://127.0.0.1:8080/v1").is_ok());
        assert!(LiveRoute::new("local", "model", "http://[::1]:8080/v1").is_ok());
        assert!(LiveRoute::new("ollama", "model", "http://worker.localhost:11434/v1").is_ok());
    }

    #[test]
    fn typed_provider_requests_match_the_normalized_live_route() {
        let route = LiveRoute::new(
            "pipenetwork",
            "pipe/deepseek-v4-flash-0731",
            "https://API.pipenetwork.ai:443/v1/",
        )
        .unwrap();
        assert_eq!(route.base_url, "https://api.pipenetwork.ai/v1");

        route
            .validate_provider_requests(&[
                request(
                    "https://api.pipenetwork.ai/v1",
                    "pipe/deepseek-v4-flash-0731",
                    false,
                    503,
                ),
                request(
                    "https://api.pipenetwork.ai/v1/",
                    "pipe/deepseek-v4-flash-0731",
                    true,
                    200,
                ),
            ])
            .unwrap();
    }

    #[test]
    fn typed_provider_request_mismatches_are_rejected() {
        let route = LiveRoute::new(
            "pipenetwork",
            "pipe/deepseek-v4-flash-0731",
            "https://api.pipenetwork.ai/v1",
        )
        .unwrap();
        let wrong_model = request("https://api.pipenetwork.ai/v1", "wrong-model", true, 200);
        assert!(
            format!(
                "{:#}",
                route
                    .validate_provider_requests(&[wrong_model])
                    .unwrap_err()
            )
            .contains("model mismatch")
        );

        let wrong_route = request(
            "https://other.example/v1",
            "pipe/deepseek-v4-flash-0731",
            true,
            200,
        );
        assert!(
            format!(
                "{:#}",
                route
                    .validate_provider_requests(&[wrong_route])
                    .unwrap_err()
            )
            .contains("route mismatch")
        );

        let mut wrong_adapter = request(
            "https://api.pipenetwork.ai/v1",
            "pipe/deepseek-v4-flash-0731",
            true,
            200,
        );
        wrong_adapter["provider"] = serde_json::json!("xai");
        assert!(
            format!(
                "{:#}",
                route
                    .validate_provider_requests(&[wrong_adapter])
                    .unwrap_err()
            )
            .contains("adapter mismatch")
        );

        let inconsistent = request(
            "https://api.pipenetwork.ai/v1",
            "pipe/deepseek-v4-flash-0731",
            true,
            503,
        );
        assert!(
            format!(
                "{:#}",
                route
                    .validate_provider_requests(&[inconsistent])
                    .unwrap_err()
            )
            .contains("acceptance/status mismatch")
        );

        assert!(
            format!(
                "{:#}",
                route
                    .validate_provider_requests(&[serde_json::json!({"audit_valid": false})])
                    .unwrap_err()
            )
            .contains("valid typed wire audit")
        );

        assert!(
            format!("{:#}", route.validate_provider_requests(&[]).unwrap_err())
                .contains("no accepted HTTP request")
        );

        let rejected_only = request(
            "https://api.pipenetwork.ai/v1",
            "pipe/deepseek-v4-flash-0731",
            false,
            503,
        );
        assert!(
            format!(
                "{:#}",
                route
                    .validate_provider_requests(&[rejected_only])
                    .unwrap_err()
            )
            .contains("no accepted HTTP request")
        );
    }
}
