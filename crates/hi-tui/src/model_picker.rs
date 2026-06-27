//! Interactive `/model` picker: a filterable, arrow-navigable list of model ids.

use std::collections::HashMap;

use hi_ai::ServedModel;

use crate::PICKER_ROWS;

/// A shared empty metadata instance used when a model id has no served info.
static EMPTY_META: ModelMeta = ModelMeta {
    window: None,
    price: None,
    health: None,
    capabilities: Vec::new(),
};

/// One row of model metadata for display: the id plus whatever the endpoint
/// reported (price, context window, health). Price/window/health come from the
/// `/models` route when available; otherwise they're blank.
#[derive(Clone, Debug, Default)]
pub(crate) struct ModelMeta {
    /// `id` is stored in `ModelPicker::all`; this holds only the extras.
    pub window: Option<u32>,
    pub price: Option<(f64, f64)>,
    pub health: Option<String>,
    /// Capability tags from the static catalog: "tools", "reasoning".
    pub capabilities: Vec<&'static str>,
}

impl ModelMeta {
    fn from_served(sm: &ServedModel) -> Self {
        Self {
            window: sm.context_window,
            price: sm.price,
            health: sm.health().map(|h| h.to_string()),
            capabilities: Vec::new(),
        }
    }
}

/// Format a token count compactly: 8192 → "8K", 200000 → "200K", 1000000 → "1M".
fn fmt_window(w: u32) -> String {
    if w >= 1_000_000 {
        format!("{}M", w / 1_000_000)
    } else if w >= 1000 {
        format!("{}K", w / 1000)
    } else {
        w.to_string()
    }
}

/// Format a price as "$in/$out" per 1M tokens, compactly.
fn fmt_price(p: (f64, f64)) -> String {
    let (inp, outp) = p;
    // Up to two decimals, with trailing zeros (and a bare trailing dot)
    // stripped: whole-dollar prices read compactly ("$3/15") while sub-dollar
    // ones keep their precision ("$0.15/0.6"). One leading "$" covers both
    // numbers — they're always the same unit.
    let fmt = |v: f64| {
        let s = format!("{v:.2}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    };
    format!("${}/{}", fmt(inp), fmt(outp))
}

/// A display row for the picker: the model id, its formatted metadata, and
/// whether this row is the highlighted one.
pub(crate) struct PickerRow<'a> {
    pub id: &'a str,
    pub meta: &'a ModelMeta,
    pub selected: bool,
}

/// Interactive `/model` picker: a filterable, arrow-navigable list of model ids.
pub(crate) struct ModelPicker {
    pub all: Vec<String>,
    /// The model in use when the picker opened — pre-selected and marked.
    pub current: String,
    /// Per-id metadata for display (health/price/window), from the endpoint's
    /// `/models` route when available. Keyed by model id.
    pub meta: HashMap<String, ModelMeta>,
    pub filter: String,
    /// Indices into `all` matching the current filter.
    pub matches: Vec<usize>,
    /// Index into `matches` of the highlighted row.
    pub selected: usize,
}

impl ModelPicker {
    pub fn new(
        all: Vec<String>,
        current: &str,
        tags: HashMap<String, String>,
        served: &HashMap<String, ServedModel>,
        capabilities: &HashMap<String, Vec<&'static str>>,
    ) -> Self {
        let matches: Vec<usize> = (0..all.len()).collect();
        // Open with the current model highlighted (and scrolled into view).
        let selected = all.iter().position(|id| id == current).unwrap_or(0);
        // Build per-id metadata: prefer served-model data from /models, fall
        // back to the health-only `tags` map.
        let meta = all
            .iter()
            .map(|id| {
                let m = served
                    .get(id)
                    .map(ModelMeta::from_served)
                    .unwrap_or_default();
                // If served didn't report health but the legacy tags map has it,
                // use that.
                let mut m = m;
                if m.health.is_none()
                    && let Some(h) = tags.get(id)
                {
                    m.health = Some(h.clone());
                }
                // Attach capability tags from the static catalog when available.
                if let Some(caps) = capabilities.get(id) {
                    m.capabilities = caps.clone();
                }
                (id.clone(), m)
            })
            .collect();
        Self {
            all,
            current: current.to_string(),
            meta,
            filter: String::new(),
            matches,
            selected,
        }
    }

    /// Recompute matches (case-insensitive substring) after the filter changes.
    fn refilter(&mut self) {
        let needle = self.filter.to_lowercase();
        self.matches = self
            .all
            .iter()
            .enumerate()
            .filter(|(_, id)| needle.is_empty() || id.to_lowercase().contains(&needle))
            .map(|(i, _)| i)
            .collect();
        self.selected = 0;
    }

    pub fn insert(&mut self, c: char) {
        self.filter.push(c);
        self.refilter();
    }
    pub fn backspace(&mut self) {
        self.filter.pop();
        self.refilter();
    }
    pub fn down(&mut self) {
        if self.selected + 1 < self.matches.len() {
            self.selected += 1;
        }
    }
    pub fn up(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }
    pub fn page_down(&mut self) {
        self.selected = (self.selected + PICKER_ROWS).min(self.matches.len().saturating_sub(1));
    }
    pub fn page_up(&mut self) {
        self.selected = self.selected.saturating_sub(PICKER_ROWS);
    }
    pub fn current(&self) -> Option<&str> {
        self.matches
            .get(self.selected)
            .map(|&i| self.all[i].as_str())
    }

    /// The visible window of rows, scrolled to keep the selection in view.
    pub fn visible(&self) -> (usize, Vec<PickerRow<'_>>) {
        let offset = if self.selected >= PICKER_ROWS {
            self.selected + 1 - PICKER_ROWS
        } else {
            0
        };
        let end = (offset + PICKER_ROWS).min(self.matches.len());
        let rows = (offset..end)
            .map(|vi| {
                let idx = self.matches[vi];
                let id = self.all[idx].as_str();
                PickerRow {
                    id,
                    meta: self
                        .meta
                        .get(id)
                        .map(|m| m as &ModelMeta)
                        .unwrap_or(&EMPTY_META),
                    selected: vi == self.selected,
                }
            })
            .collect();
        (offset, rows)
    }
}

/// Format a model's context window for display in the picker column.
pub(crate) fn display_window(meta: &ModelMeta) -> String {
    meta.window.map(fmt_window).unwrap_or_default()
}

/// Format a model's price for display in the picker column.
pub(crate) fn display_price(meta: &ModelMeta) -> String {
    meta.price.map(fmt_price).unwrap_or_default()
}

/// Format a model's health label for display in the picker column.
pub(crate) fn display_health(meta: &ModelMeta) -> String {
    meta.health.clone().unwrap_or_default()
}

/// Format capability tags as a compact string: "tools·reasoning" or "".
pub(crate) fn display_capabilities(meta: &ModelMeta) -> String {
    if meta.capabilities.is_empty() {
        String::new()
    } else {
        meta.capabilities.join("·")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_picker_filters_and_navigates() {
        let mut p = ModelPicker::new(
            vec![
                "anthropic/claude-sonnet-4".into(),
                "openai/gpt-4o".into(),
                "openai/gpt-4o-mini".into(),
                "google/gemini".into(),
            ],
            "google/gemini",
            HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        // Opens with the current model pre-selected.
        assert_eq!(p.current(), Some("google/gemini"));
        assert_eq!(p.matches.len(), 4);
        for c in "gpt".chars() {
            p.insert(c);
        }
        assert_eq!(p.matches.len(), 2, "only gpt-* match");
        assert_eq!(p.current(), Some("openai/gpt-4o"));
        p.down();
        assert_eq!(p.current(), Some("openai/gpt-4o-mini"));
        p.down(); // clamped at the end
        assert_eq!(p.current(), Some("openai/gpt-4o-mini"));
        p.up();
        assert_eq!(p.current(), Some("openai/gpt-4o"));
        p.backspace(); // "gp"
        p.backspace(); // "g" → matches both gpt-* and google
        assert_eq!(p.filter, "g");
        assert_eq!(p.matches.len(), 3);
    }

    #[test]
    fn fmt_window_compacts_large_numbers() {
        assert_eq!(fmt_window(8192), "8K");
        assert_eq!(fmt_window(200_000), "200K");
        assert_eq!(fmt_window(1_000_000), "1M");
        assert_eq!(fmt_window(512), "512");
    }

    #[test]
    fn fmt_price_formats_compactly() {
        assert_eq!(fmt_price((3.0, 15.0)), "$3/15");
        assert_eq!(fmt_price((0.15, 0.6)), "$0.15/0.6");
        assert_eq!(fmt_price((0.0, 0.0)), "$0/0");
    }
}
