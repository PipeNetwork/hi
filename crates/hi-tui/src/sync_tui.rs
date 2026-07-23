//! Lightweight RemoteUi for the TUI's `/sync on` command. Buffers serialized
//! `UiEvent`s and flushes them to ipop's live event endpoint. This is a
//! TUI-resident version of `hi_cli::sync::RemoteUi` — it doesn't depend on
//! `hi-cli`, so the TUI can create it mid-session without a restart.

use std::sync::Mutex;

use anyhow::{Context, Result, anyhow};

const MAX_PENDING_EVENTS: usize = 1_024;
const MAX_PENDING_BYTES: usize = 8 * 1024 * 1024;
const OMITTED_EVENT_TEXT: &str = "(older live events omitted; durable session record is unchanged)";

/// Configuration for syncing live events to ipop. Mirrors the subset of
/// `hi_cli::sync::SyncConfig` that the TUI's `RemoteUi` needs.
#[derive(Clone, Debug)]
pub struct SyncConfig {
    pub base_url: String,
    pub api_key: String,
}

/// Buffers serialized `UiEvent`s for flushing to ipop's live event endpoint.
/// Best-effort: if a flush fails, events are retained for retry.
pub struct RemoteUi {
    config: SyncConfig,
    session_id: String,
    client: reqwest::Client,
    pending: Mutex<Vec<String>>,
    /// Serializes flushes so shutdown waits for an in-flight background flush.
    flush_lock: tokio::sync::Mutex<()>,
}

impl RemoteUi {
    pub fn new(config: SyncConfig, session_id: String) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .http1_only()
            .build()
            .unwrap_or_else(|_| hi_ai::timed_http_client_fallback(5, 10));
        Self {
            config,
            session_id,
            client,
            pending: Mutex::new(Vec::new()),
            flush_lock: tokio::sync::Mutex::new(()),
        }
    }

    /// Serialize and buffer a UiEvent for the next flush.
    pub fn push_event(&self, event: crate::event::UiEvent) {
        if let Ok(json) = serde_json::to_string(&event) {
            let json = if json.len() <= 256_000 {
                json
            } else {
                serde_json::to_string(&crate::event::UiEvent::Status {
                    text: "(oversized live event omitted; durable session record is unchanged)"
                        .to_string(),
                })
                .unwrap_or_default()
            };
            let mut pending = self.pending.lock().unwrap();
            pending.push(json);
            trim_pending(&mut pending);
        }
    }

    /// Flush all buffered events to ipop. Best-effort. If a flush is already
    /// in-flight, returns immediately (events stay buffered for the next call)
    /// to preserve event ordering.
    pub async fn flush(&self) -> Result<()> {
        let _flush = self.flush_lock.lock().await;
        loop {
            let events: Vec<String> = {
                let mut pending = self.pending.lock().unwrap();
                if pending.is_empty() {
                    return Ok(());
                }
                let mut count = 0;
                let mut bytes = 0usize;
                for event in pending.iter().take(256) {
                    let next = bytes.saturating_add(event.len());
                    if count > 0 && next > 1_800_000 {
                        break;
                    }
                    count += 1;
                    bytes = next;
                }
                pending.drain(..count).collect()
            };

            let url = format!(
                "{}/hi/sessions/{}/events",
                self.config.base_url, self.session_id
            );
            let body = serde_json::json!({
                "events": events.iter().map(|e| {
                    serde_json::json!({ "event_json": e })
                }).collect::<Vec<_>>(),
            });

            let response = match self
                .client
                .post(&url)
                .header("x-api-key", &self.config.api_key)
                .json(&body)
                .send()
                .await
            {
                Ok(response) => response,
                Err(err) => {
                    self.requeue_front(events);
                    return Err(err).with_context(|| format!("flushing live events to {url}"));
                }
            };

            if !response.status().is_success() {
                self.requeue_front(events);
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(anyhow!("live event flush failed: {status} {body}"));
            }
        }
    }

    fn requeue_front(&self, mut events: Vec<String>) {
        let mut pending = self.pending.lock().unwrap();
        let protected = events.len();
        events.append(&mut *pending);
        *pending = events;
        trim_pending_after(&mut pending, protected);
    }
}

fn trim_pending(pending: &mut Vec<String>) {
    trim_pending_after(pending, 0);
}

fn trim_pending_after(pending: &mut Vec<String>, protected: usize) {
    let marker = serde_json::to_string(&crate::event::UiEvent::Status {
        text: OMITTED_EVENT_TEXT.to_string(),
    })
    .unwrap_or_default();
    let protected = protected.min(pending.len());
    let mut dropped = false;
    while pending.len() > MAX_PENDING_EVENTS
        || pending.iter().map(String::len).sum::<usize>() > MAX_PENDING_BYTES
    {
        if pending.len() <= protected {
            break;
        }
        pending.remove(protected);
        dropped = true;
    }
    if dropped && pending.get(protected) != Some(&marker) {
        pending.insert(protected, marker);
    }
    while pending.len() > MAX_PENDING_EVENTS
        || pending.iter().map(String::len).sum::<usize>() > MAX_PENDING_BYTES
    {
        if pending.len() <= protected + 1 {
            break;
        }
        pending.remove(protected + 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_queue_is_bounded_and_marks_omissions() {
        let mut pending = (0..=MAX_PENDING_EVENTS)
            .map(|index| format!("event-{index}"))
            .collect::<Vec<_>>();
        trim_pending(&mut pending);
        assert!(pending.len() <= MAX_PENDING_EVENTS);
        assert!(pending[0].contains(OMITTED_EVENT_TEXT));
        assert!(pending.last().is_some_and(|event| event == "event-1024"));
    }

    #[test]
    fn requeue_trim_preserves_failed_batch_order() {
        let failed = vec!["failed-0".to_string(), "failed-1".to_string()];
        let mut pending = failed.clone();
        pending.extend((0..MAX_PENDING_EVENTS).map(|index| format!("new-{index}")));
        trim_pending_after(&mut pending, failed.len());

        assert!(pending.len() <= MAX_PENDING_EVENTS);
        assert_eq!(&pending[..failed.len()], failed.as_slice());
        assert!(pending[failed.len()].contains(OMITTED_EVENT_TEXT));
    }
}
