//! Interactive `/provider` picker: a filterable, arrow-navigable list of the
//! things you can switch to — configured profiles first, then the built-in
//! provider presets.
//!
//! Presets are listed alongside profiles because a provider is usable without
//! one (`/provider xai` right after `/login xai`), and a list that only showed
//! profiles left no way to discover or reach them.

/// One selectable row: a configured profile, a managed local model, or a
/// built-in provider preset.
#[derive(Clone)]
pub(crate) struct ProviderEntry {
    /// The token passed to `/provider <name>` when this row is chosen.
    pub name: String,
    /// Right-hand detail: the profile's provider/model, or the preset's label.
    pub detail: String,
    /// Presets sort after profiles and are marked differently.
    pub is_preset: bool,
    pub is_local: bool,
    action: ProviderChoice,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ProviderChoice {
    Named(String),
    LocalModel(String),
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
    #[cfg(test)]
    pub fn new(profiles: Vec<(String, String)>, current: &str) -> Self {
        // Keep the legacy unit-test constructor environment-independent; the
        // production constructor performs the real Ollama socket probe.
        Self::new_with_local_status(profiles, Vec::new(), current, true)
    }

    /// Build the picker with managed local model actions between configured
    /// profiles and provider presets. Local rows are actions rather than
    /// profiles because selecting one may need a download and server startup.
    pub fn new_with_local(
        profiles: Vec<(String, String)>,
        local_models: Vec<(String, String, String)>,
        current: &str,
    ) -> Self {
        Self::new_with_local_status(profiles, local_models, current, ollama_is_running())
    }

    fn new_with_local_status(
        profiles: Vec<(String, String)>,
        local_models: Vec<(String, String, String)>,
        current: &str,
        ollama_running: bool,
    ) -> Self {
        let mut all: Vec<ProviderEntry> = profiles
            .into_iter()
            .filter(|(_, detail)| ollama_running || !profile_uses_ollama(detail))
            .map(|(name, detail)| ProviderEntry {
                action: ProviderChoice::Named(name.clone()),
                name,
                detail,
                is_preset: false,
                is_local: false,
            })
            .collect();
        for (name, detail, model) in local_models {
            all.push(ProviderEntry {
                name,
                detail,
                is_preset: false,
                is_local: true,
                action: ProviderChoice::LocalModel(model),
            });
        }
        // A preset whose name is already a profile would switch to the profile
        // anyway (profiles shadow presets in resolution), so listing it twice
        // would be a lie about what selecting it does.
        for (name, detail) in PRESETS {
            if *name == "ollama" && !ollama_running {
                continue;
            }
            if all.iter().any(|entry| entry.name == *name) {
                continue;
            }
            all.push(ProviderEntry {
                action: ProviderChoice::Named((*name).to_string()),
                name: (*name).to_string(),
                detail: (*detail).to_string(),
                is_preset: true,
                is_local: false,
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

    /// Replace the local-model rows after the background Hub catalog refresh
    /// completes. Profile and preset rows keep their order; an active filter
    /// is reapplied so the picker can update while it is open.
    pub fn replace_local_models(&mut self, local_models: Vec<(String, String, String)>) {
        let highlighted = self.current_choice().and_then(|choice| match choice {
            ProviderChoice::LocalModel(model) => Some(model),
            ProviderChoice::Named(_) => None,
        });
        self.all.retain(|entry| !entry.is_local);
        let insert_at = self
            .all
            .iter()
            .position(|entry| entry.is_preset)
            .unwrap_or(self.all.len());
        let rows = local_models
            .into_iter()
            .map(|(name, detail, model)| ProviderEntry {
                name,
                detail,
                is_preset: false,
                is_local: true,
                action: ProviderChoice::LocalModel(model),
            });
        self.all.splice(insert_at..insert_at, rows);
        self.refilter();
        if let Some(highlighted) = highlighted
            && let Some(row) = self.matches.iter().position(|index| {
                self.all.get(*index).is_some_and(|entry| {
                    entry.action == ProviderChoice::LocalModel(highlighted.clone())
                })
            })
        {
            self.selected = row;
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
    #[cfg(test)]
    pub fn current_name(&self) -> Option<&str> {
        self.matches
            .get(self.selected)
            .and_then(|i| self.all.get(*i))
            .and_then(|e| match &e.action {
                ProviderChoice::Named(name) => Some(name.as_str()),
                ProviderChoice::LocalModel(_) => None,
            })
    }

    /// The highlighted row's action. Local model selections are handled by
    /// the background runtime path instead of being sent through profile
    /// resolution.
    pub fn current_choice(&self) -> Option<ProviderChoice> {
        self.matches
            .get(self.selected)
            .and_then(|i| self.all.get(*i))
            .map(|entry| entry.action.clone())
    }

    /// Rows to render: (name, detail, is_preset, is_local, is_active,
    /// is_highlighted).
    pub fn visible(&self) -> Vec<(&str, &str, bool, bool, bool, bool)> {
        self.matches
            .iter()
            .enumerate()
            .filter_map(|(row, index)| {
                let entry = self.all.get(*index)?;
                Some((
                    entry.name.as_str(),
                    entry.detail.as_str(),
                    entry.is_preset,
                    entry.is_local,
                    entry.name == self.current,
                    row == self.selected,
                ))
            })
            .collect()
    }
}

/// Ollama is a local service, not just a provider preset. Keep the preset out
/// of `/provider` until its configured socket is reachable so selecting it
/// cannot immediately produce a confusing connection-refused error.
fn ollama_is_running() -> bool {
    use std::net::{TcpStream, ToSocketAddrs};
    use std::time::Duration;

    let configured = std::env::var("OLLAMA_HOST")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "127.0.0.1:11434".to_string());
    let host = configured
        .trim()
        .trim_start_matches("http://")
        .trim_start_matches("https://")
        .split('/')
        .next()
        .unwrap_or("127.0.0.1:11434");
    let address = if host.contains(':') {
        host.to_string()
    } else {
        format!("{host}:11434")
    };
    address
        .to_socket_addrs()
        .ok()
        .into_iter()
        .flatten()
        .any(|address| TcpStream::connect_timeout(&address, Duration::from_millis(40)).is_ok())
}

fn profile_uses_ollama(detail: &str) -> bool {
    detail
        .split('·')
        .next()
        .is_some_and(|provider| provider.trim().eq_ignore_ascii_case("ollama"))
}

/// Display rows for models that are actually selectable on this machine. The
/// live Pipe Network catalog is merged into the built-in fallback catalog once
/// the background refresh completes; entries that exceed either RAM or free
/// disk are omitted instead of letting a selection predictably fail later.
pub(crate) fn local_model_rows() -> Vec<(String, String, String)> {
    let ram = hi_agent::local_skeptic::system_ram_gb();
    let backend = hi_agent::local_skeptic::detect_backend_cached();
    if backend != Some(hi_agent::local_skeptic::LocalBackend::Mlx) || ram == 0 {
        return Vec::new();
    }
    let mut rows = hi_agent::local_skeptic::SUPPORTED_LOCAL_MODELS
        .iter()
        .filter_map(|entry| {
            let quant = entry.pick_mlx(ram)?;
            let dir = hi_tools::skeptic_model_dir(quant.repo);
            let spec = hi_agent::local_skeptic::LocalModelSpec {
                repo: quant.repo.to_string(),
                model_id: quant.model_id.to_string(),
                gguf_file: None,
                backend: hi_agent::local_skeptic::LocalBackend::Mlx,
            };
            let available = hi_tools::available_space_bytes(&dir);
            if available.is_some_and(|bytes| {
                bytes < quant.download_gb.saturating_add(1) * 1024 * 1024 * 1024
            }) {
                return None;
            }
            let status = if hi_agent::local_skeptic::model_present(&dir, &spec) {
                format!("MLX {} · ready", quant.quant)
            } else {
                format!("MLX {} · {:.0}GB download", quant.quant, quant.download_gb)
            };
            let display = if entry.name == "deepseek-coder-v2-lite" {
                "DeepSeek Coder V2 Lite".to_string()
            } else {
                entry.name.to_string()
            };
            Some((
                display,
                format!("{} · {status}", entry.label),
                entry.name.to_string(),
            ))
        })
        .collect::<Vec<_>>();

    if let Some(catalog) = hi_agent::local_skeptic::cached_pipenetwork_catalog() {
        rows.extend(catalog.into_iter().filter_map(|model| {
            let dir = hi_tools::skeptic_model_dir(&model.repo);
            let available = hi_tools::available_space_bytes(&dir);
            if !model.fits_machine(ram, available) {
                return None;
            }
            let spec = hi_agent::local_skeptic::LocalModelSpec {
                repo: model.repo.clone(),
                model_id: model.model_id.clone(),
                gguf_file: None,
                backend: hi_agent::local_skeptic::LocalBackend::Mlx,
            };
            let status = if hi_agent::local_skeptic::model_present(&dir, &spec) {
                "ready".to_string()
            } else {
                format!("{:.1}GB download", model.download_bytes as f64 / 1e9)
            };
            Some((
                model.display_name,
                format!("Pipe Network · {} · {}", model.collection, status),
                model.repo,
            ))
        }));
    }
    rows
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

    #[test]
    fn local_rows_are_actions_separate_from_named_profiles() {
        let picker = ProviderPicker::new_with_local(
            profiles(),
            vec![(
                "DeepSeek Coder V2 Lite".into(),
                "MLX 4bit · download on select".into(),
                "deepseek-coder-v2-lite".into(),
            )],
            "local",
        );
        let local_index = picker.all.iter().position(|entry| entry.is_local).unwrap();
        let mut picker = picker;
        picker.selected = picker
            .matches
            .iter()
            .position(|i| *i == local_index)
            .unwrap();
        assert_eq!(
            picker.current_choice(),
            Some(ProviderChoice::LocalModel("deepseek-coder-v2-lite".into()))
        );
        assert_eq!(picker.current_name(), None);
    }

    #[test]
    fn background_catalog_replaces_local_rows_without_losing_profiles_or_presets() {
        let mut picker = ProviderPicker::new_with_local_status(
            profiles(),
            vec![(
                "DeepSeek Coder V2 Lite".into(),
                "MLX 4bit · 9GB download".into(),
                "deepseek-coder-v2-lite".into(),
            )],
            "local",
            true,
        );
        picker.replace_local_models(vec![(
            "VISTA 9B".into(),
            "Pipe Network · VISTA MLX · ready".into(),
            "pipenetwork/VISTA-9B-MLX-4bit".into(),
        )]);

        assert!(picker.all.iter().any(|entry| entry.name == "VISTA 9B"));
        assert!(
            !picker
                .all
                .iter()
                .any(|entry| entry.name == "DeepSeek Coder V2 Lite")
        );
        assert!(picker.all.iter().any(|entry| entry.name == "local"));
        assert!(picker.all.iter().any(|entry| entry.name == "ollama"));
    }

    #[test]
    fn unavailable_ollama_preset_is_not_listed() {
        let picker = ProviderPicker::new_with_local_status(profiles(), Vec::new(), "openai", false);
        assert!(!picker.all.iter().any(|entry| entry.name == "ollama"));
        assert!(!picker.all.iter().any(|entry| entry.name == "local"));
        assert!(picker.all.iter().any(|entry| entry.name == "work"));
        assert!(picker.all.iter().any(|entry| entry.name == "openai"));
    }
}
