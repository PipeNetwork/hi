//! Interactive `/provider add` and `/provider edit` form: a multi-field
//! input overlay rendered above the transcript. Tab/Shift-Tab cycles fields,
//! Enter submits, Esc cancels. Reuses `InputLine` for per-field editing.

use crate::input::InputLine;

/// The provider choices shown in the form, in display order.
const PROVIDER_CHOICES: &[(&str, &str)] = &[
    ("ollama", "Ollama (local)"),
    ("pipenetwork", "pipenetwork.ai"),
    ("anthropic", "Anthropic"),
    ("openai", "OpenRouter"),
];

/// One editable field in the form.
struct Field {
    label: &'static str,
    /// The current value being typed.
    input: InputLine,
    /// A short hint shown when the field is empty.
    placeholder: &'static str,
    /// Whether this is the provider-type field (rendered as a picker, not
    /// freeform text).
    is_provider_picker: bool,
}

/// The form state: a list of fields, the active field index, and whether we're
/// editing (vs creating). When editing, the name field is read-only.
pub(crate) struct ProviderForm {
    fields: Vec<Field>,
    active: usize,
    /// The profile name (read-only when editing, editable when adding).
    pub editing: bool,
    /// The provider index into `PROVIDER_CHOICES` (for the picker field).
    provider_idx: usize,
}

impl ProviderForm {
    /// Create a form for adding a new profile. Fields start empty.
    pub fn new_add() -> Self {
        Self {
            fields: vec![
                Field {
                    label: "Name",
                    input: InputLine::default(),
                    placeholder: "e.g. sonnet, local, work",
                    is_provider_picker: false,
                },
                Field {
                    label: "API key",
                    input: InputLine::default(),
                    placeholder: "paste key or env var name (e.g. ANTHROPIC_API_KEY)",
                    is_provider_picker: false,
                },
                Field {
                    label: "Model",
                    input: InputLine::default(),
                    placeholder: "optional — blank to pick via /model",
                    is_provider_picker: false,
                },
                Field {
                    label: "Base URL",
                    input: InputLine::default(),
                    placeholder: "blank for provider default",
                    is_provider_picker: false,
                },
            ],
            active: 0,
            editing: false,
            provider_idx: 0,
        }
    }

    /// Create a form for editing an existing profile, pre-filled with current
    /// values. The name field is read-only.
    pub fn new_edit(
        name: &str,
        provider: &str,
        api_key: &str,
        model: &str,
        base_url: &str,
    ) -> Self {
        let provider_idx = PROVIDER_CHOICES
            .iter()
            .position(|(id, _)| *id == provider)
            .unwrap_or(0);

        let mut form = Self {
            fields: vec![
                Field {
                    label: "Name",
                    input: InputLine::default(),
                    placeholder: "",
                    is_provider_picker: false,
                },
                Field {
                    label: "API key",
                    input: InputLine::default(),
                    placeholder: "blank = keep current",
                    is_provider_picker: false,
                },
                Field {
                    label: "Model",
                    input: InputLine::default(),
                    placeholder: "blank = keep current",
                    is_provider_picker: false,
                },
                Field {
                    label: "Base URL",
                    input: InputLine::default(),
                    placeholder: "blank = keep current",
                    is_provider_picker: false,
                },
            ],
            active: 1, // Start on the API key field (name is read-only).
            editing: true,
            provider_idx,
        };
        // Pre-fill the name (read-only display) and other fields.
        form.fields[0].input.set(name);
        if !api_key.is_empty() {
            form.fields[1].input.set(api_key);
        }
        if !model.is_empty() {
            form.fields[2].input.set(model);
        }
        if !base_url.is_empty() {
            form.fields[3].input.set(base_url);
        }
        form
    }

    /// The active field index.
    pub fn active(&self) -> usize {
        self.active
    }

    /// The provider picker index.
    pub fn provider_idx(&self) -> usize {
        self.provider_idx
    }

    /// Move to the next field (Tab). Cycles to submit after the last field.
    pub fn next_field(&mut self) {
        if self.active < self.fields.len() - 1 {
            self.active += 1;
        }
    }

    /// Move to the previous field (Shift-Tab / BackTab).
    pub fn prev_field(&mut self) {
        if self.active > 0 && !(self.editing && self.active == 1) {
            self.active -= 1;
        }
    }

    /// Cycle the provider picker to the next choice (only meaningful when the
    /// provider field is active — it's always shown as field 0 in the picker
    /// row, but we render it separately from the text fields).
    pub fn cycle_provider(&mut self) {
        self.provider_idx = (self.provider_idx + 1) % PROVIDER_CHOICES.len();
    }

    /// Cycle the provider picker to the previous choice.
    pub fn cycle_provider_prev(&mut self) {
        if self.provider_idx == 0 {
            self.provider_idx = PROVIDER_CHOICES.len() - 1;
        } else {
            self.provider_idx -= 1;
        }
    }

    /// Insert a character into the active text field.
    pub fn insert(&mut self, c: char) {
        if self.active < self.fields.len() {
            self.fields[self.active].input.insert(c);
        }
    }

    /// Backspace in the active text field.
    pub fn backspace(&mut self) {
        if self.active < self.fields.len() {
            self.fields[self.active].input.backspace();
        }
    }

    /// Clear the active text field.
    pub fn clear_field(&mut self) {
        if self.active < self.fields.len() {
            self.fields[self.active].input.clear();
        }
    }

    /// The text in the active field.
    pub fn active_text(&self) -> String {
        if self.active < self.fields.len() {
            self.fields[self.active].input.text()
        } else {
            String::new()
        }
    }

    /// The cursor position in the active field.
    pub fn active_cursor(&self) -> usize {
        if self.active < self.fields.len() {
            self.fields[self.active].input.cursor()
        } else {
            0
        }
    }

    /// Collect the form data into a `ProfileFormData`. Returns `None` if the
    /// name field is empty (when adding).
    pub fn data(&self) -> Option<super::ProfileFormData> {
        let name = self.fields[0].input.text();
        if !self.editing && name.is_empty() {
            return None;
        }
        let api_key = self.fields[1].input.text();
        let model = self.fields[2].input.text();
        let base_url = self.fields[3].input.text();
        let provider = PROVIDER_CHOICES[self.provider_idx].0.to_string();
        // Heuristic: if the key looks like an env var name (all caps + _),
        // store as env var reference.
        let store_as_env = !api_key.is_empty()
            && api_key
                .chars()
                .all(|c| c.is_uppercase() || c == '_' || c.is_ascii_digit())
            && api_key.contains('_');

        Some(super::ProfileFormData {
            name,
            provider,
            api_key,
            store_as_env,
            model,
            base_url,
        })
    }

    /// The provider choices for rendering.
    pub fn provider_choices(&self) -> &'static [(&'static str, &'static str)] {
        PROVIDER_CHOICES
    }

    /// The label and placeholder for each text field, for rendering.
    pub fn field_labels(&self) -> Vec<(&'static str, &'static str, String, bool)> {
        self.fields
            .iter()
            .enumerate()
            .map(|(i, f)| (f.label, f.placeholder, f.input.text(), i == self.active))
            .collect()
    }
}
