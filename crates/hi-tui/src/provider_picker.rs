//! Interactive `/provider` picker: a filterable, arrow-navigable list of the
//! things you can switch to — configured profiles first, then the built-in
//! provider presets.
//!
//! Presets are listed alongside profiles because a provider is usable without
//! one (`/provider xai` right after `/login xai`), and a list that only showed
//! profiles left no way to discover or reach them.

/// One selectable row: a configured profile or a built-in provider preset.
#[derive(Clone)]
pub(crate) struct ProviderEntry {
    /// The token passed to `/provider <name>` when this row is chosen.
    pub name: String,
    /// Right-hand detail: the profile's provider/model, or the preset's label.
    pub detail: String,
    /// Presets sort after profiles and are marked differently.
    pub is_preset: bool,
}

/// The built-in providers, in display order. Kept beside
/// `provider_form::PROVIDER_CHOICES` — that list is what you can *create a
/// profile for*, this one is what you can *switch to right now*, so it also
/// includes providers with no interactive setup form.
const PRESETS: &[(&str, &str)] = &[
    (
        "xai",
        "xAI (Grok) — subscription via /login xai, or XAI_API_KEY",
    ),
    ("pipenetwork", "pipenetwork.ai"),
    ("anthropic", "Anthropic (Claude)"),
    ("openai", "OpenAI-compatible (OpenRouter by default)"),
    ("ollama", "Ollama (local)"),
];

pub(crate) struct ProviderPicker {
    pub all: Vec<ProviderEntry>,
    /// The profile/provider in use when the picker opened, marked in the list.
    pub current: String,
    pub filter: String,
    /// Indices into `all` matching the current filter.
    pub matches: Vec<usize>,
    /// Index into `matches` of the highlighted row.
    pub selected: usize,
}

impl ProviderPicker {
    /// `profiles` is (name, detail) for each configured profile.
    pub fn new(profiles: Vec<(String, String)>, current: &str) -> Self {
        let mut all: Vec<ProviderEntry> = profiles
            .into_iter()
            .map(|(name, detail)| ProviderEntry {
                name,
                detail,
                is_preset: false,
            })
            .collect();
        // A preset whose name is already a profile would switch to the profile
        // anyway (profiles shadow presets in resolution), so listing it twice
        // would be a lie about what selecting it does.
        for (name, detail) in PRESETS {
            if all.iter().any(|entry| entry.name == *name) {
                continue;
            }
            all.push(ProviderEntry {
                name: (*name).to_string(),
                detail: (*detail).to_string(),
                is_preset: true,
            });
        }
        let selected = all.iter().position(|e| e.name == current).unwrap_or(0);
        let matches = (0..all.len()).collect();
        Self {
            all,
            current: current.to_string(),
            filter: String::new(),
            matches,
            selected,
        }
    }

    fn refilter(&mut self) {
        let needle = self.filter.to_lowercase();
        self.matches = self
            .all
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                needle.is_empty()
                    || entry.name.to_lowercase().contains(&needle)
                    || entry.detail.to_lowercase().contains(&needle)
            })
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
        if !self.matches.is_empty() && self.selected + 1 < self.matches.len() {
            self.selected += 1;
        }
    }

    pub fn up(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }

    pub fn page_down(&mut self) {
        if !self.matches.is_empty() {
            self.selected = (self.selected + 10).min(self.matches.len() - 1);
        }
    }

    pub fn page_up(&mut self) {
        self.selected = self.selected.saturating_sub(10);
    }

    /// The highlighted entry's name, for `/provider <name>`.
    pub fn current_name(&self) -> Option<&str> {
        self.matches
            .get(self.selected)
            .and_then(|i| self.all.get(*i))
            .map(|e| e.name.as_str())
    }

    /// Rows to render: (name, detail, is_preset, is_active, is_highlighted).
    pub fn visible(&self) -> Vec<(&str, &str, bool, bool, bool)> {
        self.matches
            .iter()
            .enumerate()
            .filter_map(|(row, index)| {
                let entry = self.all.get(*index)?;
                Some((
                    entry.name.as_str(),
                    entry.detail.as_str(),
                    entry.is_preset,
                    entry.name == self.current,
                    row == self.selected,
                ))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profiles() -> Vec<(String, String)> {
        vec![
            ("local".to_string(), "ollama · qwen2.5-coder".to_string()),
            (
                "work".to_string(),
                "anthropic · claude-sonnet-4".to_string(),
            ),
        ]
    }

    #[test]
    fn lists_profiles_first_then_presets() {
        let picker = ProviderPicker::new(profiles(), "local");
        let names: Vec<&str> = picker.all.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(&names[..2], &["local", "work"]);
        assert!(
            names.contains(&"xai"),
            "presets must be reachable: {names:?}"
        );
        assert!(picker.all[0].name == "local" && !picker.all[0].is_preset);
    }

    /// Profiles shadow presets when resolving, so showing both would misrepresent
    /// what selecting the preset row actually does.
    #[test]
    fn a_profile_named_after_a_provider_is_not_duplicated() {
        let picker = ProviderPicker::new(
            vec![("xai".to_string(), "xai · grok-4.5".to_string())],
            "xai",
        );
        let xai_rows = picker.all.iter().filter(|e| e.name == "xai").count();
        assert_eq!(xai_rows, 1);
        assert!(
            !picker.all[0].is_preset,
            "the profile should be the one kept"
        );
    }

    #[test]
    fn opens_with_the_active_entry_highlighted() {
        let picker = ProviderPicker::new(profiles(), "work");
        assert_eq!(picker.current_name(), Some("work"));
    }

    #[test]
    fn arrows_move_and_clamp_at_both_ends() {
        let mut picker = ProviderPicker::new(profiles(), "local");
        assert_eq!(picker.current_name(), Some("local"));
        picker.down();
        assert_eq!(picker.current_name(), Some("work"));
        picker.up();
        picker.up(); // already at the top — must not wrap or underflow
        assert_eq!(picker.current_name(), Some("local"));
        for _ in 0..100 {
            picker.down();
        }
        assert!(
            picker.current_name().is_some(),
            "paging past the end must not select a nonexistent row"
        );
    }

    #[test]
    fn filtering_matches_name_or_detail() {
        let mut picker = ProviderPicker::new(profiles(), "local");
        picker.insert('g');
        picker.insert('r');
        picker.insert('o');
        picker.insert('k');
        // "grok" appears in the xai preset's detail, not its name.
        assert_eq!(picker.current_name(), Some("xai"));
        picker.backspace();
        picker.backspace();
        picker.backspace();
        picker.backspace();
        assert_eq!(picker.matches.len(), picker.all.len());
    }

    #[test]
    fn a_filter_matching_nothing_selects_nothing_rather_than_panicking() {
        let mut picker = ProviderPicker::new(profiles(), "local");
        for c in "zzzzz".chars() {
            picker.insert(c);
        }
        assert!(picker.matches.is_empty());
        assert_eq!(picker.current_name(), None);
        picker.down();
        picker.page_down();
        assert_eq!(picker.current_name(), None);
    }
}
