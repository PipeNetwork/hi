//! Dedicated local-model picker used by `/local`.

use hi_agent::local_skeptic::{LocalModelOption, LocalModelSource};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum LocalChoice {
    Model(LocalModelOption),
    ExistingDirectory,
}

pub(crate) struct LocalModelPicker {
    pub all: Vec<LocalModelOption>,
    pub filter: String,
    pub matches: Vec<usize>,
    pub selected: usize,
}

impl LocalModelPicker {
    pub fn new(mut all: Vec<LocalModelOption>) -> Self {
        // Keep the most useful local choices first: installed models, then
        // smaller downloads, then larger models.
        all.sort_by_key(|model| {
            (
                !model.installed,
                model.download_bytes.unwrap_or(u64::MAX),
                model.display_name.to_ascii_lowercase(),
            )
        });
        let mut matches: Vec<usize> = (0..all.len()).collect();
        matches.push(usize::MAX);
        Self {
            all,
            filter: String::new(),
            matches,
            selected: 0,
        }
    }

    fn refilter(&mut self) {
        let needle = self.filter.to_ascii_lowercase();
        self.matches = self
            .all
            .iter()
            .enumerate()
            .filter(|(_, model)| {
                needle.is_empty()
                    || model.display_name.to_ascii_lowercase().contains(&needle)
                    || model.model_id.to_ascii_lowercase().contains(&needle)
                    || match &model.source {
                        LocalModelSource::Hub { repo } => {
                            repo.to_ascii_lowercase().contains(&needle)
                        }
                        LocalModelSource::Directory { path } => path
                            .to_string_lossy()
                            .to_ascii_lowercase()
                            .contains(&needle),
                    }
            })
            .map(|(index, _)| index)
            .collect();
        if needle.is_empty() || needle.contains("path") || needle.contains("directory") {
            self.matches.push(usize::MAX);
        }
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

    pub fn up(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }

    pub fn down(&mut self) {
        if !self.matches.is_empty() {
            self.selected = (self.selected + 1).min(self.matches.len() - 1);
        }
    }

    pub fn page_up(&mut self) {
        self.selected = self.selected.saturating_sub(crate::PICKER_ROWS);
    }

    pub fn page_down(&mut self) {
        if !self.matches.is_empty() {
            self.selected = (self.selected + crate::PICKER_ROWS).min(self.matches.len() - 1);
        }
    }

    pub fn current_choice(&self) -> Option<LocalChoice> {
        let index = *self.matches.get(self.selected)?;
        if index == usize::MAX {
            Some(LocalChoice::ExistingDirectory)
        } else {
            self.all.get(index).cloned().map(LocalChoice::Model)
        }
    }

    pub fn visible(&self) -> Vec<(Option<&str>, Option<&LocalModelOption>, bool)> {
        self.matches
            .iter()
            .enumerate()
            .filter_map(|(row, index)| {
                if *index == usize::MAX {
                    Some((
                        Some("Use existing MLX directory…"),
                        None,
                        row == self.selected,
                    ))
                } else {
                    self.all.get(*index).map(|model| {
                        (
                            Some(model.display_name.as_str()),
                            Some(model),
                            row == self.selected,
                        )
                    })
                }
            })
            .collect()
    }
}

pub(crate) fn option_detail(model: &LocalModelOption) -> String {
    let quant = model.quantization.as_deref().unwrap_or("quant unknown");
    let download = model
        .download_bytes
        .map(format_bytes)
        .unwrap_or_else(|| "download unknown".into());
    let memory = model
        .resident_bytes
        .map(format_bytes)
        .or_else(|| model.min_ram_gb.map(|gb| format!("≥{gb}GiB RAM")))
        .unwrap_or_else(|| "memory unknown".into());
    let context = model
        .context_window
        .map(format_window)
        .unwrap_or_else(|| "context unknown".into());
    let tools = match model.tool_support {
        hi_agent::local_skeptic::LocalToolSupport::ToolCapable => "tool-capable",
        hi_agent::local_skeptic::LocalToolSupport::ChatOnly => "chat-only",
        hi_agent::local_skeptic::LocalToolSupport::Unknown => "tools unknown",
    };
    let state = if model.installed {
        "installed"
    } else {
        "download"
    };
    format!("{quant} · {download} · {memory} · {context} · {tools} · {state}")
}

fn format_bytes(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    if bytes as f64 >= GIB {
        format!("{:.1}GiB", bytes as f64 / GIB)
    } else {
        format!("{}MiB", bytes / (1024 * 1024))
    }
}

fn format_window(window: u32) -> String {
    if window >= 1_000_000 {
        format!("{}M ctx", window / 1_000_000)
    } else if window >= 1_000 {
        format!("{}K ctx", window / 1_000)
    } else {
        format!("{window} ctx")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(name: &str, installed: bool) -> LocalModelOption {
        LocalModelOption {
            display_name: name.into(),
            model_id: name.into(),
            source: LocalModelSource::Hub {
                repo: format!("org/{name}"),
            },
            quantization: Some("4bit".into()),
            download_bytes: Some(4),
            resident_bytes: Some(5),
            min_ram_gb: Some(8),
            context_window: None,
            tool_support: Default::default(),
            installed,
        }
    }

    #[test]
    fn installed_models_sort_before_downloads_and_filter() {
        let mut picker = LocalModelPicker::new(vec![model("large", false), model("ready", true)]);
        assert_eq!(picker.visible()[0].0, Some("ready"));
        for c in "large".chars() {
            picker.insert(c);
        }
        assert_eq!(
            picker.current_choice().map(|choice| match choice {
                LocalChoice::Model(model) => model.display_name,
                LocalChoice::ExistingDirectory => String::new(),
            }),
            Some("large".into())
        );
    }
}
