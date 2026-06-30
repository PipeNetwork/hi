//! `App` methods: completion.



use crate::completion::{
    CompletionContext, CompletionItem, CompletionState, MODEL_CMD, MODEL_COMPLETION_MAX,
    PROVIDER_CMD, completion_context, completion_items_for,
};

impl crate::App {

    /// Rows for a completion context — model ids from the live catalog, profile
    /// names from the config, every other command's values from the static
    /// table.
    pub(crate) fn items_for_ctx(&self, ctx: &CompletionContext) -> Vec<CompletionItem> {
        if let CompletionContext::Arg { cmd, prefix } = ctx
            && *cmd == MODEL_CMD
        {
            return self.model_completion_items(prefix);
        }
        if let CompletionContext::Arg { cmd, prefix } = ctx
            && *cmd == PROVIDER_CMD
        {
            return self.provider_completion_items(prefix);
        }
        completion_items_for(ctx)
    }

    /// Up to [`MODEL_COMPLETION_MAX`] catalog ids starting with `prefix` (already
    /// lowercased), as `/model <id>` rows — inline type-ahead for `/model`.
    pub(crate) fn model_completion_items(&self, prefix: &str) -> Vec<CompletionItem> {
        self.model_ids
            .iter()
            .filter(|id| id.to_lowercase().starts_with(prefix))
            .take(MODEL_COMPLETION_MAX)
            .map(|id| CompletionItem {
                label: id.clone(),
                help: String::new(),
                insert: format!("/{MODEL_CMD} {id}"),
                submit_on_enter: true,
            })
            .collect()
    }

    /// Profile names + `add`/`edit`/`remove` subcommands matching `prefix`, as
    /// `/provider <name>` rows — inline type-ahead for `/provider`.
    pub(crate) fn provider_completion_items(&self, prefix: &str) -> Vec<CompletionItem> {
        let mut items: Vec<CompletionItem> = Vec::new();
        // Subcommands first.
        for sub in ["add", "edit", "remove"] {
            if sub.starts_with(prefix) {
                items.push(CompletionItem {
                    label: sub.to_string(),
                    help: match sub {
                        "add" => "create a new profile",
                        "edit" => "edit an existing profile",
                        "remove" => "remove a profile",
                        _ => "",
                    }
                    .to_string(),
                    insert: format!("/{PROVIDER_CMD} {sub}"),
                    submit_on_enter: true,
                });
            }
        }
        // Profile names.
        for p in &self.profiles {
            if p.name.starts_with(prefix) {
                let help = format!(
                    "{} · {}",
                    p.provider,
                    p.model.as_deref().unwrap_or("pick via /model")
                );
                items.push(CompletionItem {
                    label: p.name.clone(),
                    help,
                    insert: format!("/{PROVIDER_CMD} {}", p.name),
                    submit_on_enter: true,
                });
            }
        }
        items
    }

    /// The rows the completion menu currently offers (empty when closed).
    pub(crate) fn completion_items(&self) -> Vec<CompletionItem> {
        match &self.completion {
            Some(c) => self.items_for_ctx(&c.ctx),
            None => Vec::new(),
        }
    }

    /// Re-sync the completion menu to the current input: open/refresh it when the
    /// input is a slash-command name being typed (`/`, `/mo`, …) or the argument
    /// of a command with enumerable values (`/compact `, `/model gp`), with
    /// matches; otherwise close it. Called after every edit to the input line.
    pub(crate) fn sync_completion(&mut self) {
        match completion_context(&self.input.text()) {
            Some(ctx) if !self.items_for_ctx(&ctx).is_empty() => {
                // Reset the highlight only when the context actually changed, so
                // navigation survives unrelated redraws.
                if self.completion.as_ref().map(|c| &c.ctx) != Some(&ctx) {
                    self.completion = Some(CompletionState { ctx, selected: 0 });
                }
            }
            _ => self.completion = None,
        }
    }

    /// Move the completion highlight by `delta`, clamped to the match list.
    pub(crate) fn completion_move(&mut self, delta: isize) {
        let len = self.completion_items().len();
        if let Some(c) = &mut self.completion
            && len > 0
        {
            let last = len - 1;
            c.selected = match delta {
                d if d < 0 => c.selected.saturating_sub(1),
                _ => (c.selected + 1).min(last),
            };
        }
    }

    /// Accept the highlighted completion: replace the input with the row's
    /// insertion (`/name`, `/name ` for an arg-taking command, or `/cmd value`)
    /// and close the menu. When `submit` is set and the row is a complete line,
    /// return it to run immediately; otherwise leave it in the input.
    pub(crate) fn accept_completion(&mut self, submit: bool) -> Option<String> {
        let items = self.completion_items();
        let c = self.completion.as_ref()?;
        let item = items.get(c.selected)?;
        let submit_on_enter = item.submit_on_enter;
        self.input.set(&item.insert);
        self.completion = None;
        if submit && submit_on_enter {
            Some(self.input.submit())
        } else {
            None
        }
    }
}
