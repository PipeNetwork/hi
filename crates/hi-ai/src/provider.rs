//! The `Provider` trait: the single seam every model backend implements.

use anyhow::Result;
use async_trait::async_trait;

use crate::types::{ChatRequest, Completion, StreamEvent};

/// A model the endpoint serves, with whatever live metadata it reports via its
/// `/models` route. Everything past `id` is best-effort — most endpoints report
/// only the id (then these stay `None`), but some (e.g. terminaili) also report
/// the context window, pricing, and a health status.
#[derive(Clone, Debug)]
pub struct ServedModel {
    pub id: String,
    /// Context window in tokens.
    pub context_window: Option<u32>,
    /// Pricing `(input, output)` in USD per 1M tokens.
    pub price: Option<(f64, f64)>,
    /// Health label as reported, e.g. "available" or "degraded".
    pub status: Option<String>,
    /// Whether the endpoint currently flags the model as usable.
    pub available: bool,
}

impl ServedModel {
    /// A short health label worth flagging, or `None` when the model is healthy
    /// (or the endpoint reported nothing). Used to warn before you rely on a
    /// degraded/limited model.
    pub fn health(&self) -> Option<&str> {
        match self.status.as_deref() {
            Some(s) if !s.eq_ignore_ascii_case("available") => Some(s),
            None if !self.available => Some("unavailable"),
            _ => None,
        }
    }
}

/// A model backend. Implementations own the wire-format translation and SSE
/// reassembly so the agent loop stays provider-agnostic.
///
/// `sink` is invoked for each incremental [`StreamEvent`] as it arrives; the
/// returned [`Completion`] is the fully-assembled assistant turn (text,
/// reasoning, and tool calls).
#[async_trait]
pub trait Provider: Send + Sync {
    async fn stream(
        &self,
        request: ChatRequest,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion>;

    /// The models this endpoint actually serves (via its `/models` route), with
    /// any live metadata reported. Default: empty, so callers fall back to the
    /// static models.dev catalog.
    async fn list_models(&self) -> Result<Vec<ServedModel>> {
        Ok(Vec::new())
    }
}
