//! Interactive `/provider add` and `/provider edit` form: a multi-field
//! input overlay rendered above the transcript. Tab/Shift-Tab cycles fields,
//! Enter submits, Esc cancels. Reuses `InputLine` for per-field editing.

use crate::input::InputLine;

/// The provider choices shown in the form, in display order.
const PROVIDER_CHOICES: &[(&str, &str)] = &[
    ("pipenetwork", "pipenetwork.ai"),
    ("ollama", "Ollama (local)"),
    ("xai", "xAI (Grok)"),
];

/// One editable field in the form.
struct Field {
    label: &'static str,
    /// The current value being typed.
    input: InputLine,
    /// A short hint shown when the field is empty.
    placeholder: &'static str,
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
                },
                Field {
                    label: "API key",
                    input: InputLine::default(),
                    placeholder: "paste key or env var name (e.g. ANTHROPIC_API_KEY)",
                },
                Field {
                    label: "Model",
                    input: InputLine::default(),
                    placeholder: "optional",
                },
                Field {
                    label: "Base URL",
                    input: InputLine::default(),
                    placeholder: "blank for provider default",
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
                },
                Field {
                    label: "API key",
                    input: InputLine::default(),
                    placeholder: "blank = keep current",
                },
                Field {
                    label: "Model",
                    input: InputLine::default(),
                    placeholder: "blank = keep current",
                },
                Field {
                    label: "Base URL",
                    input: InputLine::default(),
                    placeholder: "blank = keep current",
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
        // Start focus past the API-key field when it's hidden (Ollama needs no
        // key). Without this the form opens focused on an undrawn field, so the
        // user's first keystrokes land invisibly in the API-key input.
        form.skip_hidden();
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

    /// The provider id string for the currently-selected provider.
    pub fn provider_id(&self) -> &'static str {
        PROVIDER_CHOICES[self.provider_idx].0
    }

    /// True when the current provider doesn't need an API key (Ollama runs
    /// locally without one). The form skips the API-key field for these.
    pub fn api_key_unneeded(&self) -> bool {
        self.provider_id() == "ollama"
    }

    /// The default base URL for the currently-selected provider, to show as the
    /// placeholder when the Base URL field is empty.
    pub fn default_base_url(&self) -> &'static str {
        match self.provider_id() {
            "ollama" => "http://localhost:11434/v1",
            "pipenetwork" => "https://api.pipenetwork.ai/v1",
            "anthropic" => "https://api.anthropic.com",
            "openai" => "https://openrouter.ai/api/v1",
            "xai" => "https://api.x.ai/v1",
            _ => "",
        }
    }

    /// Advance `self.active` past any field that should be skipped (the API-key
    /// field when the provider is Ollama). Called after provider cycling and
    /// after Tab navigation.
    fn skip_hidden(&mut self) {
        if self.api_key_unneeded() && self.active == 1 {
            // Skip forward past the API-key field.
            self.active = 2;
        }
    }

    /// Move to the next field (Tab). Cycles to submit after the last field.
    pub fn next_field(&mut self) {
        if self.active < self.fields.len() - 1 {
            self.active += 1;
            self.skip_hidden();
        }
    }

    /// Move to the previous field (Shift-Tab / BackTab).
    pub fn prev_field(&mut self) {
        if self.active > 0 && !(self.editing && self.active == 1) {
            self.active -= 1;
            // If we landed on the hidden API-key field, skip back past it.
            if self.api_key_unneeded() && self.active == 1 {
                self.active = 0;
            }
        }
    }

    /// Cycle the provider picker to the next choice (only meaningful when the
    /// provider field is active — it's always shown as field 0 in the picker
    /// row, but we render it separately from the text fields).
    pub fn cycle_provider(&mut self) {
        self.provider_idx = (self.provider_idx + 1) % PROVIDER_CHOICES.len();
        self.skip_hidden();
    }

    /// Cycle the provider picker to the previous choice.
    pub fn cycle_provider_prev(&mut self) {
        if self.provider_idx == 0 {
            self.provider_idx = PROVIDER_CHOICES.len() - 1;
        } else {
            self.provider_idx -= 1;
        }
        self.skip_hidden();
    }

    /// Move the cursor left within the active text field.
    pub fn cursor_left(&mut self) {
        if self.active < self.fields.len() {
            self.fields[self.active].input.left();
        }
    }

    /// Move the cursor right within the active text field.
    pub fn cursor_right(&mut self) {
        if self.active < self.fields.len() {
            self.fields[self.active].input.right();
        }
    }

    /// Insert a character into the active text field.
    pub fn insert(&mut self, c: char) {
        if self.active < self.fields.len() {
            self.fields[self.active].input.insert(c);
        }
    }

    /// Insert a (possibly multi-line) string into the active text field — used
    /// for pastes. Newlines are stripped so a pasted key stays on one line.
    pub fn insert_str(&mut self, s: &str) {
        if self.active < self.fields.len() {
            let single = s.replace("\r\n", "").replace(['\r', '\n'], "");
            for c in single.chars() {
                self.fields[self.active].input.insert(c);
            }
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
        // `store_as_env` is decided by `ProfileForm::to_profile` in hi-cli,
        // which checks whether the input is the name of an env var that's
        // actually set. The form can't make that call reliably (and a pasted
        // literal key that's all-caps+digits+underscores must not be mistaken
        // for an env var name), so we pass false here.
        let store_as_env = false;

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

    /// The label and placeholder for each text field, for rendering. The
    /// placeholder is dynamic: the API-key field shows "(not needed for Ollama)"
    /// when the provider is Ollama, and the Base URL field shows the provider's
    /// default URL.
    pub fn field_labels(&self) -> Vec<(&'static str, String, String, bool)> {
        let unneeded = self.api_key_unneeded();
        let default_url = self.default_base_url();
        self.fields
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let placeholder = if i == 1 && unneeded {
                    "(not needed for Ollama)".to_string()
                } else if i == 3 {
                    if default_url.is_empty() {
                        f.placeholder.to_string()
                    } else {
                        format!("blank for {default_url}")
                    }
                } else {
                    f.placeholder.to_string()
                };
                (f.label, placeholder, f.input.text(), i == self.active)
            })
            .collect()
    }
}
