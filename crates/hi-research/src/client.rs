use std::net::IpAddr;
use std::time::Duration;

use reqwest::{Client, Method, RequestBuilder, StatusCode};
use serde::de::DeserializeOwned;
use uuid::Uuid;

use crate::error::ResearchError;
use crate::types::*;

#[derive(Clone, Debug)]
pub struct ResearchClientConfig {
    pub origin: String,
    pub api_key: String,
}

#[derive(Clone, Debug)]
pub struct ResearchClient {
    http: Client,
    origin: String,
    api_key: String,
}

impl ResearchClient {
    pub fn from_process_defaults() -> Result<Self, ResearchError> {
        if let Some(config) = crate::process_defaults() {
            return Self::new(config.clone());
        }
        let api_key = std::env::var("PIPENETWORK_API_KEY").unwrap_or_default();
        let origin = std::env::var("PIPENETWORK_API_BASE")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_ORIGIN.to_string());
        Self::new(ResearchClientConfig { origin, api_key })
    }

    pub fn new(config: ResearchClientConfig) -> Result<Self, ResearchError> {
        if config.api_key.trim().is_empty() {
            return Err(ResearchError::fail_open(
                "Research API key missing (set PIPENETWORK_API_KEY or an active Pipe provider key)",
            ));
        }
        let origin = crate::normalize_origin(&config.origin);
        if origin.is_empty() {
            return Err(ResearchError::fail_open("Research API base URL is empty"));
        }
        validate_credential_origin(&origin)?;
        let http = Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(120))
            .build()
            .map_err(|error| ResearchError::hard(error.to_string()))?;
        Ok(Self {
            http,
            origin,
            api_key: config.api_key,
        })
    }

    pub fn origin(&self) -> &str {
        &self.origin
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.origin)
    }

    fn authorized(&self, builder: RequestBuilder) -> RequestBuilder {
        builder.bearer_auth(&self.api_key)
    }

    async fn send_json<T: DeserializeOwned>(
        &self,
        builder: RequestBuilder,
    ) -> Result<T, ResearchError> {
        let response = builder
            .send()
            .await
            .map_err(|error| ResearchError::hard(error.to_string()))?;
        let status = response.status();
        let bytes = response
            .bytes()
            .await
            .map_err(|error| ResearchError::hard(error.to_string()))?;
        if !status.is_success() {
            return Err(ResearchError::from_http(
                status.as_u16(),
                &String::from_utf8_lossy(&bytes),
            ));
        }
        serde_json::from_slice(&bytes)
            .map_err(|error| ResearchError::hard(format!("decoding research response: {error}")))
    }

    pub async fn research(&self, query: &str) -> Result<ResearchResponse, ResearchError> {
        self.send_json(
            self.authorized(self.http.request(Method::POST, self.url(RESEARCH_PATH)))
                .header("idempotency-key", Uuid::new_v4().to_string())
                .json(&ResearchRequest {
                    query: query.to_string(),
                    queries: None,
                    max_pages: None,
                    max_snippets: None,
                }),
        )
        .await
    }

    pub async fn read_page(
        &self,
        research_id: &str,
        page_id: &str,
    ) -> Result<ResearchPageResponse, ResearchError> {
        self.send_json(
            self.authorized(
                self.http
                    .get(self.url(&format!("{RESEARCH_PATH}/{research_id}/pages/{page_id}"))),
            ),
        )
        .await
    }

    /// Score drafts against the original question + snippet corpus. Fail-open
    /// returns `None` so the caller can keep the first completed draft.
    pub async fn pick_draft(
        &self,
        model: &str,
        question: &str,
        snippets: &str,
        drafts: &[String],
    ) -> Result<Option<usize>, ResearchError> {
        if drafts.is_empty() {
            return Ok(None);
        }
        let mut user = format!("Question:\n{question}\n\nSnippets:\n{snippets}\n\nDrafts:\n");
        for (index, draft) in drafts.iter().enumerate() {
            user.push_str(&format!("\n--- draft {} ---\n{draft}\n", index + 1));
        }
        user.push_str(
            "\nReply with the winning draft number on the first line (1-based). \
Then one short reason. Prefer grounded citations over confident guesses.",
        );
        let body = serde_json::json!({
            "model": model,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "system",
                    "content": "You pick the best research draft. Untrusted web snippets. First line is a single integer."
                },
                { "role": "user", "content": user }
            ]
        });
        let response = self
            .authorized(self.http.post(self.url("/v1/chat/completions")).json(&body))
            .send()
            .await
            .map_err(|error| ResearchError::hard(error.to_string()))?;
        let status = response.status();
        let bytes = response
            .bytes()
            .await
            .map_err(|error| ResearchError::hard(error.to_string()))?;
        if status == StatusCode::UNAUTHORIZED
            || status == StatusCode::NOT_FOUND
            || status == StatusCode::SERVICE_UNAVAILABLE
        {
            return Ok(None);
        }
        if !status.is_success() {
            return Err(ResearchError::from_http(
                status.as_u16(),
                &String::from_utf8_lossy(&bytes),
            ));
        }
        let value: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|error| ResearchError::hard(format!("decoding judge response: {error}")))?;
        let content = value
            .pointer("/choices/0/message/content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        Ok(parse_winning_draft(content, drafts.len()))
    }
}

fn validate_credential_origin(origin: &str) -> Result<(), ResearchError> {
    let url = reqwest::Url::parse(origin).map_err(|error| {
        ResearchError::fail_open(format!("Research API base URL is invalid: {error}"))
    })?;
    if !url.username().is_empty() || url.password().is_some() || url.host_str().is_none() {
        return Err(ResearchError::fail_open(
            "Research API base URL must be an absolute URL without userinfo",
        ));
    }
    let loopback = url.host_str().is_some_and(|host| {
        let host = host
            .strip_prefix('[')
            .and_then(|host| host.strip_suffix(']'))
            .unwrap_or(host);
        host.eq_ignore_ascii_case("localhost")
            || host
                .parse::<IpAddr>()
                .is_ok_and(|address| address.is_loopback())
    });
    if url.scheme() != "https" && !(url.scheme() == "http" && loopback) {
        return Err(ResearchError::fail_open(
            "Research API base URL must use HTTPS (loopback HTTP is allowed for local development)",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    async fn client(server: &MockServer) -> ResearchClient {
        ResearchClient::new(ResearchClientConfig {
            origin: server.uri(),
            api_key: "pk_test".into(),
        })
        .unwrap()
    }

    #[test]
    fn missing_key_fail_open() {
        let error = ResearchClient::new(ResearchClientConfig {
            origin: "http://127.0.0.1:1".into(),
            api_key: "  ".into(),
        })
        .unwrap_err();
        assert!(error.is_fail_open());
    }

    #[test]
    fn credential_transport_requires_https_or_exact_loopback_http() {
        for accepted in [
            "https://api.example.com",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            "http://[::1]:8080",
        ] {
            assert!(
                ResearchClient::new(ResearchClientConfig {
                    origin: accepted.into(),
                    api_key: "secret".into(),
                })
                .is_ok(),
                "{accepted}"
            );
        }
        for rejected in [
            "http://api.example.com",
            "http://localhost.evil.example",
            "ftp://127.0.0.1",
            "https://user@example.com",
            "not a url",
        ] {
            let error = ResearchClient::new(ResearchClientConfig {
                origin: rejected.into(),
                api_key: "secret".into(),
            })
            .unwrap_err();
            assert!(error.is_fail_open(), "{rejected}: {error}");
        }
    }

    #[tokio::test]
    async fn research_round_trip() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(RESEARCH_PATH))
            .and(header("authorization", "Bearer pk_test"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "research",
                "research_id": "res_1",
                "query": "zig http",
                "queries": ["zig http"],
                "snippets": [{
                    "snippet_id": "sn_00",
                    "page_id": "pg_a",
                    "url": "https://example.com",
                    "title": "Zig",
                    "text": "CANDIDATE snippet_id=sn_00\n---\nhello",
                    "score": 0.9
                }],
                "pages": [{
                    "page_id": "pg_a",
                    "url": "https://example.com",
                    "title": "Zig",
                    "fetched": true
                }]
            })))
            .mount(&server)
            .await;
        let got = client(&server).await.research("zig http").await.unwrap();
        assert_eq!(got.research_id, "res_1");
        assert_eq!(got.snippets.len(), 1);
    }

    #[tokio::test]
    async fn unavailable_is_fail_open() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(RESEARCH_PATH))
            .respond_with(ResponseTemplate::new(503).set_body_json(serde_json::json!({
                "error": { "code": "research_backend_unavailable", "message": "down" }
            })))
            .mount(&server)
            .await;
        let error = client(&server).await.research("q").await.unwrap_err();
        assert!(error.is_fail_open());
        assert_eq!(error.code.as_deref(), Some("research_backend_unavailable"));
    }

    #[tokio::test]
    async fn read_page_round_trip() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/research/res_1/pages/pg_a"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "research.page",
                "research_id": "res_1",
                "page_id": "pg_a",
                "url": "https://example.com",
                "title": "Zig",
                "markdown": "# hi"
            })))
            .mount(&server)
            .await;
        let page = client(&server)
            .await
            .read_page("res_1", "pg_a")
            .await
            .unwrap();
        assert_eq!(page.markdown, "# hi");
    }

    #[test]
    fn request_omits_unknown_keys() {
        let value = serde_json::to_value(&ResearchRequest {
            query: "q".into(),
            queries: None,
            max_pages: None,
            max_snippets: None,
        })
        .unwrap();
        let keys: Vec<_> = value.as_object().unwrap().keys().cloned().collect();
        assert_eq!(keys, vec!["query"]);
    }

    #[test]
    fn winning_draft_parser() {
        assert_eq!(parse_winning_draft("2\nbecause citations", 3), Some(1));
        assert_eq!(parse_winning_draft("draft 3 wins", 3), Some(2));
        assert_eq!(parse_winning_draft("9", 3), None);
    }
}
