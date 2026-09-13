//! Cursor project-rule reminders attached after successful file reads.
//!
//! Walks ancestors of the read path for `.cursor/rules/*.{md,mdc}` and
//! `.cursorrules`. Matching rules are appended once per process so a long
//! edit loop does not reprint the same guideline on every `read`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};

use serde::Deserialize;

static INJECTED: LazyLock<Mutex<HashSet<PathBuf>>> = LazyLock::new(|| Mutex::new(HashSet::new()));

#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CursorRuleFrontmatter {
    #[serde(default)]
    always_apply: bool,
    #[serde(default)]
    globs: Option<GlobField>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum GlobField {
    String(String),
    List(Vec<String>),
}

impl GlobField {
    fn into_patterns(self) -> Vec<String> {
        match self {
            Self::String(value) => split_patterns(&value),
            Self::List(values) => values
                .into_iter()
                .flat_map(|v| split_patterns(&v))
                .collect(),
        }
    }
}

fn split_patterns(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Append matching Cursor rules to a successful `read` payload.
pub fn append_cursor_rules_for_read(workspace_root: &Path, read_path: &Path, content: &mut String) {
    let Some(reminder) = reminder_for(workspace_root, read_path) else {
        return;
    };
    if !content.is_empty() {
        content.push_str("\n\n");
    }
    content.push_str(&reminder);
}

fn reminder_for(workspace_root: &Path, read_path: &Path) -> Option<String> {
    let rel = read_path.strip_prefix(workspace_root).unwrap_or(read_path);
    let rel_str = rel.to_string_lossy();
    let mut bodies = Vec::new();
    for scope in ancestor_scopes(workspace_root, read_path) {
        for rule in scan_scope(&scope) {
            if !rule_matches(&rule, &rel_str) {
                continue;
            }
            if !mark_injected(&rule.path) {
                continue;
            }
            bodies.push(format!(
                "[Cursor rule: {}]\n{}",
                rule.path.display(),
                rule.body.trim()
            ));
        }
    }
    if bodies.is_empty() {
        None
    } else {
        Some(bodies.join("\n\n"))
    }
}

struct ParsedRule {
    path: PathBuf,
    body: String,
    always_apply: bool,
    globs: Vec<String>,
}

fn ancestor_scopes(workspace_root: &Path, read_path: &Path) -> Vec<PathBuf> {
    let mut scopes = Vec::new();
    let mut current = if read_path.is_dir() {
        read_path.to_path_buf()
    } else {
        read_path.parent().unwrap_or(read_path).to_path_buf()
    };
    loop {
        if current.starts_with(workspace_root) {
            scopes.push(current.clone());
        }
        if current == workspace_root {
            break;
        }
        match current.parent() {
            Some(parent) => current = parent.to_path_buf(),
            None => break,
        }
    }
    scopes
}

fn scan_scope(scope: &Path) -> Vec<ParsedRule> {
    let mut out = Vec::new();
    let rules_dir = scope.join(".cursor").join("rules");
    if let Ok(entries) = std::fs::read_dir(&rules_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if !matches!(ext, "md" | "mdc") {
                continue;
            }
            if let Some(rule) = parse_rule(&path) {
                out.push(rule);
            }
        }
    }
    let legacy = scope.join(".cursorrules");
    if legacy.is_file()
        && let Some(rule) = parse_rule(&legacy)
    {
        out.push(rule);
    }
    out
}

fn parse_rule(path: &Path) -> Option<ParsedRule> {
    let raw = std::fs::read_to_string(path).ok()?;
    let (front, body) = split_frontmatter(&raw);
    let meta: CursorRuleFrontmatter = front.and_then(serde_yaml_frontmatter).unwrap_or_default();
    Some(ParsedRule {
        path: path.to_path_buf(),
        body: body.trim().to_string(),
        always_apply: meta.always_apply || path.file_name().is_some_and(|n| n == ".cursorrules"),
        globs: meta.globs.map(GlobField::into_patterns).unwrap_or_default(),
    })
}

fn split_frontmatter(raw: &str) -> (Option<&str>, &str) {
    let Some(rest) = raw
        .strip_prefix("---\n")
        .or_else(|| raw.strip_prefix("---\r\n"))
    else {
        return (None, raw);
    };
    let Some(end) = rest.find("\n---").or_else(|| rest.find("\r\n---")) else {
        return (None, raw);
    };
    let front = &rest[..end];
    let after = rest[end..]
        .find('\n')
        .map(|i| &rest[end + i + 1..])
        .unwrap_or("");
    (Some(front), after)
}

fn serde_yaml_frontmatter(text: &str) -> Option<CursorRuleFrontmatter> {
    // Cursor frontmatter is YAML; accept the small subset we care about without
    // a YAML crate: `alwaysApply: true` and `globs: ...`.
    let mut meta = CursorRuleFrontmatter::default();
    for line in text.lines() {
        let line = line.trim();
        if let Some(value) = line.strip_prefix("alwaysApply:") {
            meta.always_apply = matches!(value.trim(), "true" | "True" | "yes" | "on");
        } else if let Some(value) = line.strip_prefix("always_apply:") {
            meta.always_apply = matches!(value.trim(), "true" | "True" | "yes" | "on");
        } else if let Some(value) = line.strip_prefix("globs:") {
            let value = value.trim().trim_matches('"').trim_matches('\'');
            if !value.is_empty() && value != "[]" {
                meta.globs = Some(GlobField::String(value.to_string()));
            }
        }
    }
    Some(meta)
}

fn rule_matches(rule: &ParsedRule, rel: &str) -> bool {
    if rule.always_apply || rule.globs.is_empty() {
        return true;
    }
    rule.globs.iter().any(|glob| glob_match(glob, rel))
}

fn glob_match(pattern: &str, path: &str) -> bool {
    let pat = pattern.trim().trim_start_matches("./");
    let path = path.trim_start_matches("./");
    if pat == "**" || pat == "*" {
        return true;
    }
    if let Some(suffix) = pat.strip_prefix("**/") {
        return path.ends_with(suffix)
            || path.contains(&format!("/{suffix}"))
            || glob_match(suffix, path);
    }
    if let Some(ext) = pat.strip_prefix("*.") {
        return path.rsplit('.').next() == Some(ext);
    }
    path == pat || path.ends_with(&format!("/{pat}"))
}

fn mark_injected(path: &Path) -> bool {
    INJECTED
        .lock()
        .map(|mut set| set.insert(path.to_path_buf()))
        .unwrap_or(false)
}

#[cfg(test)]
pub(crate) fn reset_injected_for_tests() {
    if let Ok(mut set) = INJECTED.lock() {
        set.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn injects_matching_cursor_rule_once() {
        reset_injected_for_tests();
        let dir = TempDir::new().unwrap();
        let rules = dir.path().join(".cursor").join("rules");
        std::fs::create_dir_all(&rules).unwrap();
        std::fs::write(
            rules.join("rust.mdc"),
            "---\nalwaysApply: false\nglobs: \"*.rs\"\n---\nUse Result.\n",
        )
        .unwrap();
        let file = dir.path().join("src").join("lib.rs");
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, "fn f() {}\n").unwrap();
        let mut content = "fn f() {}".to_string();
        append_cursor_rules_for_read(dir.path(), &file, &mut content);
        assert!(content.contains("Use Result."), "{content}");
        let mut again = "fn f() {}".to_string();
        append_cursor_rules_for_read(dir.path(), &file, &mut again);
        assert_eq!(again, "fn f() {}");
    }
}
