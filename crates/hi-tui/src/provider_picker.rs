//! Interactive `/provider` picker: a filterable, arrow-navigable list of the
//! things you can switch to — configured profiles first, then the built-in
//! provider presets, then local models.
//!
//! Presets are listed alongside profiles because a provider is usable without
//! one (`/provider xai` right after `/login xai`), and a list that only showed
//! profiles left no way to discover or reach them. Hosted presets stay above
//! the (often long) local-model catalog so `/provider` can still select
//! `pipenetwork` after `/login pipenetwork`.

use crate::PICKER_ROWS;

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
    /// Persisted managed-local profiles are local actions too, but catalog
    /// refreshes must retain them alongside the newly discovered rows.
    managed_local_profile: bool,
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
///
/// Shared with `/provider` slash-completion so typing `/provider pipe` can
/// select the hosted preset instead of only matching profile names.
pub(crate) const PRESETS: &[(&str, &str)] = &[
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

    /// Build the picker from full profile metadata so a managed local profile
    /// remains a local provisioning action after switching away from it. Its
    /// persisted endpoint may be stale after the server is stopped or the app
    /// restarts; the repository is the durable intent.
    pub fn new_with_profile_infos(
        profiles: Vec<crate::ProfileInfo>,
        local_models: Vec<(String, String, String)>,
        current: &str,
    ) -> Self {
        let managed = profiles
            .iter()
            .filter_map(|profile| {
                profile
                    .managed_local_path
                    .as_ref()
                    .map(|path| (profile.name.clone(), path.to_string_lossy().into_owned()))
                    .or_else(|| {
                        profile
                            .managed_local_repo
                            .as_ref()
                            .map(|repo| (profile.name.clone(), repo.clone()))
                    })
            })
            .collect::<std::collections::HashMap<_, _>>();
        let profile_rows = profiles
            .into_iter()
            .map(|profile| {
                (
                    profile.name,
                    format!(
                        "{} · {}",
                        profile.provider,
                        profile.model.as_deref().unwrap_or("(no model set)")
                    ),
                )
            })
            .collect();
        let mut picker = Self::new_with_local(profile_rows, local_models, current);
        for entry in &mut picker.all {
            if let Some(repo) = managed.get(&entry.name) {
                entry.is_local = true;
                entry.managed_local_profile = true;
                entry.action = ProviderChoice::LocalModel(repo.clone());
                entry.detail.push_str(" · managed MLX");
            }
        }
        picker
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
                managed_local_profile: false,
            })
            .collect();
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
                managed_local_profile: false,
            });
        }
        // Local catalog rows come last so a long MLX list cannot push hosted
        // presets like `pipenetwork` off the first page.
        for (name, detail, model) in local_models {
            all.push(ProviderEntry {
                name,
                detail,
                is_preset: false,
                is_local: true,
                managed_local_profile: false,
                action: ProviderChoice::LocalModel(model),
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
        let highlighted = self.current_choice();
        self.all
            .retain(|entry| !entry.is_local || entry.managed_local_profile);
        // Catalog rows stay after profiles and hosted presets.
        let rows = local_models
            .into_iter()
            .map(|(name, detail, model)| ProviderEntry {
                name,
                detail,
                is_preset: false,
                is_local: true,
                managed_local_profile: false,
                action: ProviderChoice::LocalModel(model),
            });
        self.all.extend(rows);
        self.refilter();
        if let Some(highlighted) = highlighted
            && let Some(row) = self.matches.iter().position(|index| {
                self.all
                    .get(*index)
                    .is_some_and(|entry| entry.action == highlighted)
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
    /// is_highlighted). The window scrolls so the highlighted row stays
    /// on-screen — same contract as the `/model` picker.
    pub fn visible(&self) -> Vec<(&str, &str, bool, bool, bool, bool)> {
        let offset = if self.selected >= PICKER_ROWS {
            self.selected + 1 - PICKER_ROWS
        } else {
            0
        };
        let end = (offset + PICKER_ROWS).min(self.matches.len());
        self.matches[offset..end]
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
                    offset + row == self.selected,
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

/// Structured models that are actually selectable on this machine. The live
/// Pipe Network catalog is merged into the built-in fallback catalog once the
/// background refresh completes; entries that exceed either RAM or free disk
/// are omitted instead of letting a selection predictably fail later.
pub(crate) fn local_model_options() -> Vec<hi_agent::local_skeptic::LocalModelOption> {
    let ram = hi_agent::local_skeptic::system_ram_gb();
    let backend = hi_agent::local_skeptic::detect_backend_cached();
    if backend != Some(hi_agent::local_skeptic::LocalBackend::Mlx) || ram == 0 {
        return Vec::new();
    }
    // Prefer live Pipe Network metadata whenever it is available. The
    // built-in entries are a fallback for offline startup, but their sizes
    // are estimates and otherwise duplicate every matching live row.
    let live_catalog = hi_agent::local_skeptic::cached_pipenetwork_catalog();
    let live_repos = live_catalog
        .as_ref()
        .map(|catalog| {
            catalog
                .iter()
                .map(|model| model.repo.as_str())
                .collect::<std::collections::HashSet<_>>()
        })
        .unwrap_or_default();
    let mut rows = hi_agent::local_skeptic::SUPPORTED_LOCAL_MODELS
        .iter()
        .filter_map(|entry| {
            let quant = entry.pick_mlx(ram)?;
            if live_repos.contains(quant.repo) {
                return None;
            }
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
            let display = if entry.name == "deepseek-coder-v2-lite" {
                "DeepSeek Coder V2 Lite".to_string()
            } else {
                entry.name.to_string()
            };
            Some(hi_agent::local_skeptic::LocalModelOption {
                display_name: display,
                model_id: quant.model_id.to_string(),
                source: hi_agent::local_skeptic::LocalModelSource::Hub {
                    repo: quant.repo.to_string(),
                },
                quantization: Some(quant.quant.to_string()),
                download_bytes: Some(quant.download_gb * 1024 * 1024 * 1024),
                resident_bytes: None,
                min_ram_gb: Some(quant.min_ram_gb),
                context_window: None,
                tool_support: hi_agent::local_skeptic::LocalToolSupport::ToolCapable,
                installed: hi_agent::local_skeptic::model_present(&dir, &spec),
            })
        })
        .collect::<Vec<_>>();

    if let Some(catalog) = live_catalog {
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
            Some(hi_agent::local_skeptic::LocalModelOption {
                display_name: model.display_name,
                model_id: model.model_id,
                source: hi_agent::local_skeptic::LocalModelSource::Hub { repo: model.repo },
                quantization: Some(model.quant),
                download_bytes: Some(model.download_bytes),
                resident_bytes: Some(model.resident_bytes),
                min_ram_gb: Some((model.resident_bytes / (1024 * 1024 * 1024)).saturating_add(8)),
                context_window: model.context_window,
                tool_support: model.tool_support,
                installed: hi_agent::local_skeptic::model_present(&dir, &spec),
            })
        }));
    }
    rows
}

/// Compatibility representation for the existing `/provider` picker.
pub(crate) fn local_model_rows() -> Vec<(String, String, String)> {
    local_model_options()
        .into_iter()
        .map(|model| {
            let status = if model.installed {
                "ready".to_string()
            } else {
                model
                    .download_bytes
                    .map(format_bytes)
                    .map(|size| format!("{size} download"))
                    .unwrap_or_else(|| "download on select".to_string())
            };
            let detail = format!(
                "MLX {} · {} · {}",
                model.quantization.as_deref().unwrap_or("unknown quant"),
                status,
                tool_support_label(model.tool_support)
            );
            let action = match model.source {
                hi_agent::local_skeptic::LocalModelSource::Hub { repo } => repo,
                hi_agent::local_skeptic::LocalModelSource::Directory { path } => {
                    path.to_string_lossy().into_owned()
                }
            };
            (model.display_name, detail, action)
        })
        .collect()
}

fn tool_support_label(support: hi_agent::local_skeptic::LocalToolSupport) -> &'static str {
    match support {
        hi_agent::local_skeptic::LocalToolSupport::ToolCapable => "tools",
        hi_agent::local_skeptic::LocalToolSupport::ChatOnly => "chat-only",
        hi_agent::local_skeptic::LocalToolSupport::Unknown => "tools unknown",
    }
}

fn format_bytes(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    if bytes as f64 >= GIB {
        format!("{:.1}GiB", bytes as f64 / GIB)
    } else {
        format!("{}MiB", bytes / (1024 * 1024))
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
    fn managed_profile_reuses_repository_action_after_switching_away() {
        let mut picker = ProviderPicker::new_with_profile_infos(
            vec![crate::ProfileInfo {
                name: "mlx-model".into(),
                provider: "openai".into(),
                model: Some("Model".into()),
                base_url: Some("http://127.0.0.1:8080/v1".into()),
                managed_local_repo: Some("org/model".into()),
                managed_local_path: None,
            }],
            Vec::new(),
            "openai",
        );
        picker.replace_local_models(vec![(
            "fresh-model".into(),
            "Pipe Network · ready".into(),
            "org/fresh-model".into(),
        )]);
        let entry = picker
            .all
            .iter()
            .find(|entry| entry.name == "mlx-model")
            .unwrap();
        assert!(entry.is_local);
        assert_eq!(entry.action, ProviderChoice::LocalModel("org/model".into()));
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

    #[test]
    fn hosted_presets_stay_above_the_local_catalog() {
        let picker = ProviderPicker::new_with_local(
            profiles(),
            vec![(
                "DeepSeek Coder V2 Lite".into(),
                "MLX 4bit · download on select".into(),
                "deepseek-coder-v2-lite".into(),
            )],
            "local",
        );
        let names: Vec<&str> = picker.all.iter().map(|e| e.name.as_str()).collect();
        let pipe = names
            .iter()
            .position(|name| *name == "pipenetwork")
            .expect("pipenetwork preset");
        let local = names
            .iter()
            .position(|name| *name == "DeepSeek Coder V2 Lite")
            .expect("local catalog row");
        assert!(
            pipe < local,
            "hosted pipenetwork must stay selectable above the local catalog: {names:?}"
        );
    }

    #[test]
    fn visible_window_scrolls_so_pipenetwork_can_be_selected() {
        let locals = (0..(PICKER_ROWS + 4))
            .map(|i| {
                (
                    format!("local-model-{i}"),
                    "MLX 4bit · download on select".into(),
                    format!("repo/model-{i}"),
                )
            })
            .collect();
        let mut picker = ProviderPicker::new_with_local(profiles(), locals, "local");
        let pipe_match = picker
            .matches
            .iter()
            .position(|index| picker.all[*index].name == "pipenetwork")
            .expect("pipenetwork in matches");
        picker.selected = pipe_match;
        let visible: Vec<&str> = picker
            .visible()
            .into_iter()
            .map(|(name, ..)| name)
            .collect();
        assert!(
            visible.contains(&"pipenetwork"),
            "highlighting pipenetwork must scroll it into the visible window: {visible:?}"
        );
        assert_eq!(
            picker.current_choice(),
            Some(ProviderChoice::Named("pipenetwork".into()))
        );
    }

    #[test]
    fn catalog_refresh_keeps_a_named_preset_highlight() {
        let mut picker = ProviderPicker::new_with_local(profiles(), Vec::new(), "local");
        let pipe_match = picker
            .matches
            .iter()
            .position(|index| picker.all[*index].name == "pipenetwork")
            .expect("pipenetwork in matches");
        picker.selected = pipe_match;
        picker.replace_local_models(vec![(
            "VISTA 9B".into(),
            "Pipe Network · VISTA MLX · ready".into(),
            "pipenetwork/VISTA-9B-MLX-4bit".into(),
        )]);
        assert_eq!(
            picker.current_choice(),
            Some(ProviderChoice::Named("pipenetwork".into()))
        );
    }
}
