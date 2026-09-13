//! Explicit review-target chdir helpers (no prompt-driven auto-chdir).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};

use crate::paths;

pub(crate) fn absolutize_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    Ok(std::env::current_dir()
        .context("determining current directory")?
        .join(path))
}

pub(crate) fn resolve_runtime_roots() -> Result<(PathBuf, PathBuf)> {
    crate::paths::resolve_runtime_roots()
}

/// Change into an explicitly supplied review-target directory.
///
/// Prompt-token heuristics used to auto-chdir here; that is intentionally gone.
/// Callers must pass a path from `--review-target` (or equivalent).
pub(crate) fn chdir_to_review_target(target: &Path) -> Result<PathBuf> {
    let target = if target.is_absolute() {
        target.to_path_buf()
    } else {
        std::env::current_dir()
            .context("determining current directory")?
            .join(target)
    };
    ensure!(
        target.is_dir(),
        "review target is not a directory: {}",
        target.display()
    );
    let target = target
        .canonicalize()
        .with_context(|| format!("canonicalizing review target {}", target.display()))?;
    let current = std::env::current_dir().context("determining current directory")?;
    let current = current.canonicalize().unwrap_or(current);
    if target != current {
        std::env::set_current_dir(&target)
            .with_context(|| format!("changing to review target {}", target.display()))?;
    }
    Ok(target)
}

/// Prompt-token path parsing — retained only for unit tests. Runtime chdir is
/// exclusively via [`chdir_to_review_target`] / `--review-target`.
#[cfg(test)]
pub(crate) fn review_target_dir_from_prompt_at(
    prompt: &str,
    cwd: &Path,
    home: Option<&Path>,
) -> Option<PathBuf> {
    let prompt = prompt
        .split("\n\nstdin:\n```")
        .next()
        .unwrap_or(prompt)
        .trim();
    if !prompt_looks_like_review_request(prompt) {
        return None;
    }
    prompt
        .split_whitespace()
        .filter_map(trim_prompt_path_token)
        .filter_map(|token| expand_review_target_token(token, cwd, home))
        .next()
}

#[cfg(test)]
fn prompt_looks_like_review_request(prompt: &str) -> bool {
    let normalized = prompt
        .split_whitespace()
        .filter(|raw| match trim_prompt_path_token(raw) {
            Some(token) => !token_looks_pathish(token),
            None => true,
        })
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { ' ' })
        .collect::<String>();
    let words = normalized.split_whitespace().collect::<Vec<_>>();
    words.iter().any(|word| {
        matches!(
            *word,
            "review" | "audit" | "status" | "roadmap" | "gap" | "gaps" | "security"
        )
    })
}

#[cfg(test)]
fn trim_prompt_path_token(raw: &str) -> Option<&str> {
    let mut token = raw.trim_matches(|ch: char| {
        matches!(
            ch,
            '"' | '\'' | '`' | '<' | '>' | '(' | ')' | '[' | ']' | '{' | '}' | ','
        )
    });
    while token.len() > 1
        && token
            .chars()
            .last()
            .is_some_and(|ch| matches!(ch, '.' | ',' | ';' | ':' | '?' | '!'))
    {
        token = &token[..token.len() - 1];
    }
    (!token.is_empty()).then_some(token)
}

#[cfg(test)]
fn token_looks_pathish(token: &str) -> bool {
    token == "~"
        || token == "."
        || token == ".."
        || token.starts_with("~/")
        || token.starts_with("./")
        || token.starts_with("../")
        || token.starts_with('/')
        || token.contains('/')
}

#[cfg(test)]
fn expand_review_target_token(token: &str, cwd: &Path, home: Option<&Path>) -> Option<PathBuf> {
    if token.contains("://") {
        return None;
    }
    let expanded = if token == "~" {
        home?.to_path_buf()
    } else if let Some(rest) = token.strip_prefix("~/") {
        home?.join(rest)
    } else {
        PathBuf::from(token)
    };
    let path = if expanded.is_absolute() {
        expanded
    } else if token_looks_pathish(token) {
        cwd.join(expanded)
    } else {
        return None;
    };
    if !path.is_dir() {
        return None;
    }
    Some(path.canonicalize().unwrap_or(path))
}
