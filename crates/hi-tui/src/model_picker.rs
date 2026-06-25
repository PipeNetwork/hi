//! Interactive `/model` picker: a filterable, arrow-navigable list of model ids.

use std::collections::HashMap;

use crate::PICKER_ROWS;

/// Interactive `/model` picker: a filterable, arrow-navigable list of model ids.
pub(crate) struct ModelPicker {
    pub all: Vec<String>,
    /// The model in use when the picker opened — pre-selected and marked.
    pub current: String,
    /// Health label per id (e.g. "degraded"), when the endpoint reported one.
    pub tags: HashMap<String, String>,
    pub filter: String,
    /// Indices into `all` matching the current filter.
    pub matches: Vec<usize>,
    /// Index into `matches` of the highlighted row.
    pub selected: usize,
}

impl ModelPicker {
    pub fn new(all: Vec<String>, current: &str, tags: HashMap<String, String>) -> Self {
        let matches: Vec<usize> = (0..all.len()).collect();
        // Open with the current model highlighted (and scrolled into view).
        let selected = all.iter().position(|id| id == current).unwrap_or(0);
        Self {
            all,
            current: current.to_string(),
            tags,
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

    /// The visible window of (id, is_selected) rows, scrolled to keep the
    /// selection in view.
    pub fn visible(&self) -> (usize, Vec<(&str, bool)>) {
        let offset = if self.selected >= PICKER_ROWS {
            self.selected + 1 - PICKER_ROWS
        } else {
            0
        };
        let end = (offset + PICKER_ROWS).min(self.matches.len());
        let rows = (offset..end)
            .map(|vi| (self.all[self.matches[vi]].as_str(), vi == self.selected))
            .collect();
        (offset, rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

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
}
