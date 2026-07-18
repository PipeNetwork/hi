//! `App` methods: completion.

use crossterm::event::{KeyCode, KeyEvent};

use crate::completion::{
    CompletionContext, CompletionItem, CompletionState, MODEL_CMD, MODEL_COMPLETION_MAX,
    PROVIDER_CMD, SESSIONS_ARCHIVE_CTX, SESSIONS_DELETE_CTX, SESSIONS_FAVORITE_CTX,
    SESSIONS_RENAME_CTX, SESSIONS_RESTORE_CTX, SESSIONS_SWITCH_CTX, completion_context,
    completion_items_for,
};

impl crate::App {
    /// Rows for a completion context — model ids from the live catalog, profile
    /// names from the config, every other command's values from the static
    /// table.
    pub(crate) fn items_for_ctx(&self, ctx: &CompletionContext) -> Vec<CompletionItem> {
        if let CompletionContext::Path { prefix } = ctx {
            return self.path_completion_items(prefix);
        }
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
        if let CompletionContext::Arg { cmd, prefix } = ctx
            && matches!(
                *cmd,
                SESSIONS_SWITCH_CTX
                    | SESSIONS_RENAME_CTX
                    | SESSIONS_FAVORITE_CTX
                    | SESSIONS_ARCHIVE_CTX
                    | SESSIONS_RESTORE_CTX
                    | SESSIONS_DELETE_CTX
            )
        {
            return self.session_completion_items(cmd, prefix);
        }
        completion_items_for(ctx)
    }

    fn session_completion_items(&self, command: &str, prefix: &str) -> Vec<CompletionItem> {
        let action = command.strip_prefix("sessions ").unwrap_or("switch");
        self.session_completion_cache
            .iter()
            .filter(|session| {
                session.id.to_lowercase().starts_with(prefix)
                    || session.title.to_lowercase().contains(prefix)
            })
            .take(8)
            .map(|session| CompletionItem {
                label: session.id.clone(),
                help: session.title.clone(),
                insert: format!(
                    "/sessions {action} {}{}",
                    session.id,
                    if matches!(action, "rename" | "delete") {
                        " "
                    } else {
                        ""
                    }
                ),
                submit_on_enter: matches!(action, "switch" | "favorite" | "archive" | "restore"),
            })
            .collect()
    }

    /// `@file` path completions: workspace-relative paths under the workspace
    /// root, resolved via `git ls-files` (so `.gitignore` is respected and the
    /// walk is fast). Falls back to a shallow directory walk outside a git repo.
    /// Matching is fuzzy: a prefix that matches the start of a path *or* the
    /// start of a filename (the last segment) ranks highest, then subsequence
    /// matches. Capped so a huge repo can't flood the menu. The `insert`
    /// replaces the `@prefix` token with `@path` followed by a space.
    fn path_completion_items(&self, prefix: &str) -> Vec<CompletionItem> {
        const MAX: usize = 12;
        let prefix_lower = prefix.to_lowercase();
        let pool: Vec<String> = if !self.path_completion_cache.is_empty() {
            self.path_completion_cache.clone()
        } else {
            // Not a git repo (or git unavailable): shallow walk fallback. Use
            // an empty prefix so we collect the full shallow set, then filter
            // below with the same fuzzy logic.
            shallow_walk(&self.workspace_root, "", 200)
        };
        let mut scored: Vec<(u8, String)> = pool
            .iter()
            .filter_map(|p| fuzzy_path_score(&p.to_lowercase(), &prefix_lower))
            .collect();
        scored.sort_by_key(|(rank, _)| *rank);
        let candidates: Vec<String> = scored.into_iter().take(MAX).map(|(_, p)| p).collect();
        // Build the insert: replace the trailing `@prefix` token with `@path `.
        // `accept_completion` does `input.set(insert)`, so we reconstruct the
        // full input with the completed token swapped in.
        let input = self.input.text();
        let token_start = input
            .rfind('@')
            .filter(|&i| i == 0 || input[..i].ends_with(char::is_whitespace));
        let before = match token_start {
            Some(i) => &input[..i],
            None => &input[..],
        };
        candidates
            .into_iter()
            .map(|path| CompletionItem {
                label: path.clone(),
                help: String::new(),
                insert: format!("{before}@{path} "),
                submit_on_enter: false,
            })
            .collect()
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
                    p.model.as_deref().unwrap_or("not configured")
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
            Some(ctx) => {
                let changed = self.completion.as_ref().map(|c| &c.ctx) != Some(&ctx);
                let entering_session_completion = match (&ctx, self.completion.as_ref()) {
                    (
                        CompletionContext::Arg { cmd, .. },
                        Some(CompletionState {
                            ctx: CompletionContext::Arg { cmd: previous, .. },
                            ..
                        }),
                    ) if matches!(*cmd, SESSIONS_SWITCH_CTX | SESSIONS_RENAME_CTX) => {
                        cmd != previous
                    }
                    (CompletionContext::Arg { cmd, .. }, _)
                        if matches!(*cmd, SESSIONS_SWITCH_CTX | SESSIONS_RENAME_CTX) =>
                    {
                        true
                    }
                    _ => false,
                };
                if entering_session_completion {
                    let mut refreshed = self
                        .session_lister
                        .as_ref()
                        .map(|lister| lister())
                        .unwrap_or_default();
                    // `/sessions` may have added synced-only entries. Preserve
                    // those while refreshing the machine cache so completion
                    // offers every session shown in the unified list.
                    for session in std::mem::take(&mut self.session_completion_cache) {
                        if !refreshed.iter().any(|item| item.id == session.id) {
                            refreshed.push(session);
                        }
                    }
                    self.session_completion_cache = refreshed;
                }
                // Refresh the path-completion cache when entering a Path context
                // (or when the prefix widens back to empty after narrowing), so
                // `git ls-files` runs once per menu-open, not per keystroke.
                let entering_path = matches!(ctx, CompletionContext::Path { .. })
                    && !matches!(
                        self.completion.as_ref().map(|c| &c.ctx),
                        Some(CompletionContext::Path { .. })
                    );
                if entering_path {
                    self.path_completion_cache = git_ls_files(&self.workspace_root);
                }
                if self.items_for_ctx(&ctx).is_empty() {
                    self.completion = None;
                    return;
                }
                // Reset the highlight only when the context actually changed, so
                // navigation survives unrelated redraws.
                if changed {
                    self.completion = Some(CompletionState { ctx, selected: 0 });
                }
            }
            _ => self.completion = None,
        }
    }

    /// Re-sync completion after an editing key. History recall can load a slash
    /// command like `/help` into the input; keep completion closed there so the
    /// next Up/Down continues moving through history instead of navigating the
    /// command menu.
    pub(crate) fn sync_completion_after_edit_key(
        &mut self,
        key: &KeyEvent,
        history_search_was_active: bool,
    ) {
        if history_search_was_active
            || self.history_search.is_some()
            || matches!(key.code, KeyCode::Up | KeyCode::Down)
        {
            self.completion = None;
        } else {
            self.sync_completion();
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
            let line = self.input.submit();
            if !line.trim().is_empty() {
                self.input.save_history(&self.workspace_root);
            }
            Some(line)
        } else {
            None
        }
    }
}

/// Score a path against a lowercase `prefix` for `@file` completion. Returns
/// `Some((rank, path))` when the path matches, where lower rank = better match:
/// - 0: the full path starts with the prefix (e.g. `src/ren` → `src/render.rs`)
/// - 1: the filename (last segment) starts with the prefix (e.g. `ren` → `src/render.rs`)
/// - 2: the prefix is a subsequence of the path (e.g. `srr` → `src/render.rs`)
/// `None` when no match. An empty prefix matches everything at rank 0.
fn fuzzy_path_score(path_lower: &str, prefix_lower: &str) -> Option<(u8, String)> {
    if prefix_lower.is_empty() {
        return Some((0, path_lower.to_string()));
    }
    if path_lower.starts_with(prefix_lower) {
        return Some((0, path_lower.to_string()));
    }
    // Filename = substring after the last '/'.
    let filename = path_lower.rsplit('/').next().unwrap_or(path_lower);
    if filename.starts_with(prefix_lower) {
        return Some((1, path_lower.to_string()));
    }
    // Subsequence match: every char of prefix appears in order in the path.
    if is_subsequence(path_lower, prefix_lower) {
        return Some((2, path_lower.to_string()));
    }
    None
}

/// True if `needle` is a subsequence of `haystack` (chars in order, not
/// necessarily contiguous).
fn is_subsequence(haystack: &str, needle: &str) -> bool {
    let mut hi = haystack.chars();
    needle.chars().all(|nc| hi.any(|hc| hc == nc))
}

/// List tracked + untracked-but-not-ignored files under `root` via
/// `git ls-files`, which respects `.gitignore` and is fast even in large repos.
/// Returns paths relative to `root`. Empty if `root` isn't a git repo or git
/// isn't available.
fn git_ls_files(root: &std::path::Path) -> Vec<String> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "--cached", "--others", "--exclude-standard"])
        .output();
    match out {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
            .lines()
            .map(|s| s.to_string())
            .collect(),
        _ => Vec::new(),
    }
}

/// Shallow fallback for non-git directories: list files and immediate children
/// of subdirectories matching `prefix_lower`, capped at `max`. Only descends
/// one level so it stays fast on large trees.
fn shallow_walk(root: &std::path::Path, prefix_lower: &str, max: usize) -> Vec<String> {
    let mut out = Vec::new();
    let entries = match std::fs::read_dir(root) {
        Ok(e) => e,
        Err(_) => return out,
    };
    for entry in entries.flatten() {
        if out.len() >= max {
            break;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        let rel = name.clone();
        if rel.to_lowercase().starts_with(prefix_lower) {
            out.push(rel);
        }
        // If the prefix could match children of this dir, list them too.
        if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            let sub = root.join(&name);
            if let Ok(sub_entries) = std::fs::read_dir(&sub) {
                for se in sub_entries.flatten() {
                    if out.len() >= max {
                        break;
                    }
                    let sname = se.file_name().to_string_lossy().into_owned();
                    if sname.starts_with('.') {
                        continue;
                    }
                    let child = format!("{name}/{sname}");
                    if child.to_lowercase().starts_with(prefix_lower) {
                        out.push(child);
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{fuzzy_path_score, is_subsequence};

    #[test]
    fn fuzzy_path_score_ranks_prefix_then_filename_then_subsequence() {
        // Full path prefix → rank 0.
        assert_eq!(
            fuzzy_path_score("src/render.rs", "src/ren"),
            Some((0, "src/render.rs".to_string()))
        );
        // Filename prefix → rank 1.
        assert_eq!(
            fuzzy_path_score("src/render.rs", "ren"),
            Some((1, "src/render.rs".to_string()))
        );
        // Subsequence → rank 2.
        assert_eq!(
            fuzzy_path_score("src/render.rs", "srr"),
            Some((2, "src/render.rs".to_string()))
        );
        // No match.
        assert_eq!(fuzzy_path_score("src/render.rs", "xyz"), None);
        // Empty prefix matches everything at rank 0.
        assert_eq!(
            fuzzy_path_score("src/render.rs", ""),
            Some((0, "src/render.rs".to_string()))
        );
    }

    #[test]
    fn is_subsequence_checks_char_order() {
        assert!(is_subsequence("src/render.rs", "srr"));
        assert!(is_subsequence("abc", "ac"));
        assert!(!is_subsequence("abc", "ca"));
        assert!(is_subsequence("abc", ""));
    }
}
