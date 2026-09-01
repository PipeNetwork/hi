use std::net::IpAddr;
use std::time::Duration;

use reqwest::{Client, Method, RequestBuilder, StatusCode};
use serde::Serialize;
use serde::de::DeserializeOwned;
use uuid::Uuid;

use crate::error::{OutcomeError, OutcomeErrorKind};
use crate::types::*;

#[derive(Clone, Debug)]
pub struct OutcomeClientConfig {
    pub origin: String,
    pub api_key: String,
}

#[derive(Clone, Debug)]
pub struct OutcomeClient {
    http: Client,
    origin: String,
    api_key: String,
}

impl OutcomeClient {
    pub fn new(config: OutcomeClientConfig) -> Result<Self, OutcomeError> {
        if config.api_key.trim().is_empty() {
            return Err(OutcomeError::fail_open(
                "Outcome API key missing (set PIPENETWORK_API_KEY or an active Pipe provider key)",
            ));
        }
        let origin = crate::normalize_origin(&config.origin);
        if origin.is_empty() {
            return Err(OutcomeError::fail_open("Outcome API base URL is empty"));
        }
        validate_credential_origin(&origin)?;
        let http = Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(90))
            .build()
            .map_err(|error| OutcomeError::hard(error.to_string()))?;
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

    fn mutating(&self, method: Method, path: &str) -> RequestBuilder {
        self.authorized(self.http.request(method, self.url(path)))
            .header("idempotency-key", Uuid::new_v4().to_string())
    }

    async fn send_json<T: DeserializeOwned>(
        &self,
        builder: RequestBuilder,
    ) -> Result<T, OutcomeError> {
        let response = builder
            .send()
            .await
            .map_err(|error| OutcomeError::hard(error.to_string()))?;
        let status = response.status();
        let bytes = response
            .bytes()
            .await
            .map_err(|error| OutcomeError::hard(error.to_string()))?;
        if !status.is_success() {
            return Err(OutcomeError::from_http(
                status.as_u16(),
                &String::from_utf8_lossy(&bytes),
            ));
        }
        serde_json::from_slice(&bytes)
            .map_err(|error| OutcomeError::hard(format!("decoding Outcome response: {error}")))
    }

    pub async fn rsi_ready(&self) -> Result<bool, OutcomeError> {
        match self
            .send_json::<PublicRsiStatus>(
                self.authorized(self.http.get(self.url("/v1/rsi/status"))),
            )
            .await
        {
            Ok(status) => Ok(status.ready),
            Err(error) if error.is_fail_open() => Ok(false),
            Err(error) => Err(error),
        }
    }

    pub async fn create_quote(&self, maximum_cost_usd: f64) -> Result<QuoteList, OutcomeError> {
        self.send_json(
            self.mutating(Method::POST, QUOTES_PATH)
                .json(&QuoteCreateRequest {
                    requirements: QuoteRequirements {
                        maximum_cost_usd: crate::clamp_cost_usd(maximum_cost_usd),
                    },
                }),
        )
        .await
    }

    pub fn pick_offer(quotes: &QuoteList, offer: OutcomeOffer) -> Option<&QuoteOffer> {
        let wanted = offer.as_route();
        quotes
            .quotes
            .iter()
            .find(|quote| quote.route == wanted)
            .or_else(|| quotes.quotes.first())
    }

    pub async fn upload_repository(
        &self,
        gzip: Vec<u8>,
        blake3: &str,
        idempotency_key: &str,
    ) -> Result<RepositoryCreated, OutcomeError> {
        self.send_json(
            self.authorized(
                self.http
                    .post(self.url("/v1/rsi/repositories"))
                    .header("idempotency-key", idempotency_key)
                    .header("content-type", "application/gzip")
                    .header("x-content-blake3", blake3)
                    .body(gzip),
            ),
        )
        .await
    }

    pub async fn create_task(&self, request: &TaskCreateRequest) -> Result<TaskView, OutcomeError> {
        self.send_json(self.mutating(Method::POST, TASKS_PATH).json(request))
            .await
    }

    pub async fn get_task(&self, task_id: &str) -> Result<TaskView, OutcomeError> {
        self.send_json(self.authorized(self.http.get(self.url(&format!("/v1/tasks/{task_id}")))))
            .await
    }

    pub async fn task_events(&self, task_id: &str) -> Result<TaskEvents, OutcomeError> {
        self.send_json(
            self.authorized(
                self.http
                    .get(self.url(&format!("/v1/tasks/{task_id}/events"))),
            ),
        )
        .await
    }

    pub async fn cancel_task(&self, task_id: &str) -> Result<TaskView, OutcomeError> {
        self.send_json(
            self.mutating(Method::POST, &format!("/v1/tasks/{task_id}/cancel"))
                .json(&serde_json::json!({})),
        )
        .await
    }

    pub async fn feedback(
        &self,
        task_id: &str,
        outcome: &str,
        reason: Option<&str>,
    ) -> Result<serde_json::Value, OutcomeError> {
        self.send_json(
            self.mutating(Method::POST, &format!("/v1/tasks/{task_id}/feedback"))
                .json(&serde_json::json!({
                    "outcome": outcome,
                    "reason": reason,
                })),
        )
        .await
    }

    pub async fn create_repair(
        &self,
        task_id: &str,
        remaining_budget_usd: f64,
    ) -> Result<RepairView, OutcomeError> {
        self.send_json(
            self.mutating(Method::POST, REPAIRS_PATH)
                .json(&RepairCreateRequest {
                    task_id: task_id.to_owned(),
                    remaining_budget_usd: crate::clamp_cost_usd(remaining_budget_usd),
                }),
        )
        .await
    }

    pub async fn create_verification(
        &self,
        task_id: &str,
        contract: TaskOutcomeContract,
    ) -> Result<VerificationView, OutcomeError> {
        self.send_json(self.mutating(Method::POST, VERIFICATIONS_PATH).json(
            &VerificationCreateRequest {
                result: VerificationSubject {
                    task_id: Some(task_id.to_owned()),
                },
                contract,
            },
        ))
        .await
    }

    pub async fn verify_receipt(&self, task_id: &str) -> Result<ExecutionReceipt, OutcomeError> {
        self.send_json(self.mutating(Method::POST, RECEIPTS_VERIFY_PATH).json(
            &ReceiptVerifyRequest {
                task_id: Some(task_id.to_owned()),
                receipt_id: None,
            },
        ))
        .await
    }

    pub async fn download_task_patch(&self, task_id: &str) -> Result<Vec<u8>, OutcomeError> {
        let mut last_error = OutcomeError::hard("task patch artifact was not available");
        for path in [
            format!("/v1/tasks/{task_id}/artifacts/patch"),
            format!("/v1/rsi/runs/{task_id}/artifacts/patch"),
        ] {
            match self.download_bytes(&path).await {
                Ok(bytes) if !bytes.is_empty() => return Ok(bytes),
                Ok(_) => {
                    last_error = OutcomeError::hard("task patch artifact was empty");
                }
                Err(error) => last_error = error,
            }
        }
        Err(last_error)
    }

    async fn download_bytes(&self, path: &str) -> Result<Vec<u8>, OutcomeError> {
        let response = self
            .authorized(self.http.get(self.url(path)))
            .send()
            .await
            .map_err(|error| OutcomeError::hard(error.to_string()))?;
        let status = response.status();
        if status == StatusCode::NOT_FOUND {
            return Err(OutcomeError::from_http(404, "artifact not found"));
        }
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(OutcomeError::from_http(status.as_u16(), &body));
        }
        Ok(response
            .bytes()
            .await
            .map_err(|error| OutcomeError::hard(error.to_string()))?
            .to_vec())
    }

    pub async fn post_json<T: DeserializeOwned>(
        &self,
        path: &str,
        body: &impl Serialize,
    ) -> Result<T, OutcomeError> {
        self.send_json(self.mutating(Method::POST, path).json(body))
            .await
    }

    pub fn classify_queue_stall(status: TaskStatus, waited_secs: u64) -> Option<OutcomeError> {
        if status == TaskStatus::Queued && waited_secs >= crate::QUEUE_STALL_SECS {
            Some(OutcomeError {
                kind: OutcomeErrorKind::FailOpen,
                status: None,
                code: Some(TASKS_UNAVAILABLE_CODE.into()),
                message: "Outcome task stayed queued with no worker progress".into(),
            })
        } else {
            None
        }
    }
}

fn validate_credential_origin(origin: &str) -> Result<(), OutcomeError> {
    let url = reqwest::Url::parse(origin).map_err(|error| {
        OutcomeError::fail_open(format!("Outcome API base URL is invalid: {error}"))
    })?;
    if !url.username().is_empty() || url.password().is_some() || url.host_str().is_none() {
        return Err(OutcomeError::fail_open(
            "Outcome API base URL must be an absolute URL without userinfo",
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
        return Err(OutcomeError::fail_open(
            "Outcome API base URL must use HTTPS (loopback HTTP is allowed for local development)",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, header_exists, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    async fn client(server: &MockServer) -> OutcomeClient {
        OutcomeClient::new(OutcomeClientConfig {
            origin: server.uri(),
            api_key: "sk_test".into(),
        })
        .unwrap()
    }

    #[tokio::test]
    async fn create_task_sends_idempotency_and_code_change() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(TASKS_PATH))
            .and(header_exists("idempotency-key"))
            .and(header("authorization", "Bearer sk_test"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "task",
                "id": "task_123",
                "status": "queued",
                "type": "code.change",
                "goal": "fix tests",
                "maximum_cost_usd": 5.0
            })))
            .mount(&server)
            .await;
        let client = client(&server).await;
        let created = client
            .create_task(&TaskCreateRequest::code_change(
                "fix tests",
                "repo_1",
                vec![cargo_test_verifier(), cargo_clippy_verifier()],
                5.0,
                1800,
            ))
            .await
            .unwrap();
        assert_eq!(created.id, "task_123");
        assert_eq!(created.status, TaskStatus::Queued);
    }

    #[tokio::test]
    async fn tasks_unavailable_is_fail_open() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(TASKS_PATH))
            .respond_with(ResponseTemplate::new(503).set_body_json(serde_json::json!({
                "error": { "code": "tasks_unavailable", "message": "worker not ready" }
            })))
            .mount(&server)
            .await;
        let client = client(&server).await;
        let error = client
            .create_task(&TaskCreateRequest::code_change(
                "fix tests",
                "repo_1",
                vec![cargo_test_verifier()],
                1.0,
                1800,
            ))
            .await
            .unwrap_err();
        assert!(error.is_fail_open());
        assert_eq!(error.code.as_deref(), Some("tasks_unavailable"));
    }

    #[tokio::test]
    async fn quotes_repairs_and_receipts_round_trip() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(QUOTES_PATH))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "quotes": [
                    { "route": "route_cheap", "price_cap_usd": 1.0 },
                    { "route": "route_quality", "price_cap_usd": 5.0 }
                ]
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path(REPAIRS_PATH))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "repair_1",
                "task_id": "task_123",
                "status": "queued"
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path(RECEIPTS_VERIFY_PATH))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "rcpt_1",
                "task_id": "task_123",
                "contract_hash": "abc",
                "outcome": "succeeded",
                "verified": true
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/tasks/task_123/events"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "events": [{ "type": "executing", "sequence": 1, "stage": "executing" }]
            })))
            .mount(&server)
            .await;
        let client = client(&server).await;
        let quotes = client.create_quote(5.0).await.unwrap();
        assert_eq!(
            OutcomeClient::pick_offer(&quotes, OutcomeOffer::Quality)
                .unwrap()
                .route,
            "route_quality"
        );
        let repair = client.create_repair("task_123", 2.0).await.unwrap();
        assert_eq!(repair.id, "repair_1");
        let receipt = client.verify_receipt("task_123").await.unwrap();
        assert!(receipt.verified);
        assert_eq!(receipt.contract_hash, "abc");
        let events = client.task_events("task_123").await.unwrap();
        assert_eq!(events.events[0].event_type, "executing");
    }

    #[test]
    fn missing_key_fail_open() {
        let error = OutcomeClient::new(OutcomeClientConfig {
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
                OutcomeClient::new(OutcomeClientConfig {
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
            let error = OutcomeClient::new(OutcomeClientConfig {
                origin: rejected.into(),
                api_key: "secret".into(),
            })
            .unwrap_err();
            assert!(error.is_fail_open(), "{rejected}: {error}");
        }
    }

    #[test]
    fn create_request_omits_unknown_keys() {
        let request = TaskCreateRequest::code_change(
            "fix",
            "repo",
            vec![cargo_test_verifier(), review_verifier()],
            5.0,
            1800,
        );
        let value = serde_json::to_value(&request).unwrap();
        let keys: Vec<_> = value.as_object().unwrap().keys().cloned().collect();
        assert!(keys.contains(&"type".into()));
        assert!(keys.contains(&"outcome_contract".into()));
        assert!(keys.contains(&"execution_policy".into()));
        assert!(!keys.contains(&"webhook_url".into()));
        assert_eq!(value["type"], "code.change");
        assert_eq!(value["outcome_contract"]["verifiers"][1]["type"], "review");
        assert_eq!(
            value["outcome_contract"]["verifiers"][1]["rubric"],
            "secure-rust-v2"
        );
    }
}
