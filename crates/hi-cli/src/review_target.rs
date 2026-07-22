//! Prompt heuristics that chdir into a review target directory.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};

use crate::session;

pub(crate) fn absolutize_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    Ok(std::env::current_dir()
        .context("determining current directory")?
        .join(path))
}

pub(crate) fn resolve_runtime_roots() -> Result<(PathBuf, PathBuf)> {
    let workspace_root = std::env::current_dir()
        .context("determining workspace root")?
        .canonicalize()
        .context("canonicalizing workspace root")?;
    ensure!(
        workspace_root.is_dir(),
        "workspace root is not a directory: {}",
        workspace_root.display()
    );
    let state_root = session::data_root()
        .map(|root| {
            root.join("projects")
                .join(session::cwd_digest())
                .join("runtime")
        })
        .unwrap_or_else(|| workspace_root.join(".hi/state"));
    std::fs::create_dir_all(&state_root)
        .with_context(|| format!("creating workspace state root {}", state_root.display()))?;
    let state_root = state_root.canonicalize().with_context(|| {
        format!(
            "canonicalizing workspace state root {}",
            state_root.display()
        )
    })?;
    ensure!(
        state_root != workspace_root && !workspace_root.starts_with(&state_root),
        "workspace state root must not equal or contain the workspace root"
    );
    Ok((workspace_root, state_root))
}

pub(crate) fn maybe_chdir_to_prompt_review_target(prompt: &str) -> Result<Option<PathBuf>> {
    let Some(target) = review_target_dir_from_prompt(prompt) else {
        return Ok(None);
    };
    let current = std::env::current_dir().context("determining current directory")?;
    let current = current.canonicalize().unwrap_or(current);
    if target == current {
        return Ok(Some(target));
    }
    std::env::set_current_dir(&target)
        .with_context(|| format!("changing to review target {}", target.display()))?;
    Ok(Some(target))
}

pub(crate) fn review_target_dir_from_prompt(prompt: &str) -> Option<PathBuf> {
    let cwd = std::env::current_dir().ok()?;
    let home = std::env::var_os("HOME").map(PathBuf::from);
    review_target_dir_from_prompt_at(prompt, &cwd, home.as_deref())
}

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
