use std::time::Duration;

use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{PipeFsClient, PipeFsError, PipeFsLease};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeaseReceipt {
    pub lease: PipeFsLease,
    pub expires_at_unix: u64,
}

#[derive(Serialize)]
struct LeaseRequest<'a> {
    client_instance_id: &'a str,
    machine_id: &'a str,
    takeover: bool,
    lease_token: &'a str,
}

#[derive(Deserialize)]
struct LeaseResponse {
    lease_token: String,
    generation: u64,
    #[serde(default)]
    expires_at_unix: u64,
}

impl PipeFsClient {
    /// Acquire the session writer lease. The client-selected token remains
    /// stable across bounded transport retries, making a lost response safe.
    pub async fn acquire_writer_lease(
        &self,
        session_id: &str,
        machine_id: &str,
        takeover: bool,
    ) -> Result<LeaseReceipt, PipeFsError> {
        if !valid_identifier(session_id) || !valid_identifier(machine_id) {
            return Err(PipeFsError::Protocol(
                "session and machine identity must be bounded safe identifiers".into(),
            ));
        }
        let client_instance_id = format!("{machine_id}-{}", std::process::id());
        let requested_token = format!("hl_{}", Uuid::new_v4().simple());
        let body = LeaseRequest {
            client_instance_id: &client_instance_id,
            machine_id,
            takeover,
            lease_token: &requested_token,
        };
        let url = self
            .base_url
            .join(&format!("hi/sessions/{session_id}/lease"))
            .map_err(|error| PipeFsError::Protocol(error.to_string()))?;
        let mut last_error = None;
        for attempt in 0..3 {
            let response = self
                .http
                .post(url.clone())
                .header("x-api-key", &self.config.api_key)
                .json(&body)
                .timeout(self.config.request_timeout)
                .send()
                .await;
            match response {
                Ok(response) if response.status().is_success() => {
                    let receipt: LeaseResponse = response
                        .json()
                        .await
                        .map_err(|error| PipeFsError::Protocol(error.to_string()))?;
                    if receipt.lease_token.is_empty() || receipt.generation == 0 {
                        return Err(PipeFsError::Protocol(
                            "writer lease response is missing its token or generation".into(),
                        ));
                    }
                    return Ok(LeaseReceipt {
                        lease: PipeFsLease {
                            token: receipt.lease_token,
                            generation: receipt.generation,
                        },
                        expires_at_unix: receipt.expires_at_unix,
                    });
                }
                Ok(response) => {
                    let status = response.status();
                    let detail = response.text().await.unwrap_or_default();
                    return Err(match status {
                        StatusCode::CONFLICT => PipeFsError::LeaseLost(detail),
                        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                            PipeFsError::Authentication(detail)
                        }
                        StatusCode::NOT_FOUND | StatusCode::METHOD_NOT_ALLOWED => {
                            PipeFsError::Protocol(
                                "server does not support mandatory writer leases".into(),
                            )
                        }
                        _ => PipeFsError::Storage(format!("HTTP {status}: {detail}")),
                    });
                }
                Err(error) if attempt < 2 && (error.is_connect() || error.is_timeout()) => {
                    last_error = Some(error.to_string());
                    tokio::time::sleep(Duration::from_millis(150)).await;
                }
                Err(error) => return Err(PipeFsError::Network(error.to_string())),
            }
        }
        Err(PipeFsError::Network(last_error.unwrap_or_else(|| {
            "writer lease request exhausted retries".into()
        })))
    }
}

fn valid_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lease_path_identifiers_reject_traversal_and_unbounded_values() {
        assert!(valid_identifier("session-1"));
        for invalid in ["", "../session", "a/session", "with space"] {
            assert!(!valid_identifier(invalid));
        }
        assert!(!valid_identifier(&"a".repeat(129)));
    }
}
