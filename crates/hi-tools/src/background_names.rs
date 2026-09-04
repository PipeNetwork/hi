//! Stable human-facing names for background shell handles.

/// Short auto-name for a shell command (UI / status lines). Never includes
/// the complete JSON arguments or command text.
pub fn shell_title(command: &str) -> String {
    let tokens: Vec<&str> = command.split_whitespace().collect();
    if tokens.is_empty() {
        return "shell".into();
    }
    let mut i = 0usize;
    while i < tokens.len() && tokens[i].contains('=') && !tokens[i].starts_with('-') {
        i += 1;
    }
    if i >= tokens.len() {
        return "shell".into();
    }
    let head = tokens[i];
    let base = std::path::Path::new(head)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(head);
    let mut parts = vec![base.to_string()];
    let mut j = i + 1;
    while j < tokens.len() && parts.len() < 3 {
        let token = tokens[j];
        if token.starts_with('-') && token != "-m" && token != "-c" {
            break;
        }
        if matches!(token, "|" | "||" | "&&" | ";" | ">" | ">>" | "<")
            || token.contains('/')
            || token.contains('\\')
        {
            break;
        }
        if token.chars().all(|character| character.is_ascii_digit()) {
            j += 1;
            continue;
        }
        parts.push(token.to_string());
        j += 1;
        if parts.len() == 2 && matches!(parts[1].as_str(), "run" | "test" | "build" | "exec") {
            continue;
        }
        if parts.len() >= 2
            && !matches!(
                parts[0].as_str(),
                "npm" | "pnpm" | "yarn" | "cargo" | "go" | "python" | "python3" | "pip" | "uv"
            )
        {
            break;
        }
    }
    let title = parts.join(" ");
    const MAX: usize = 40;
    if title.chars().count() <= MAX {
        title
    } else {
        format!("{}…", title.chars().take(MAX).collect::<String>())
    }
}

/// Command-derived slug plus a registry-local monotonic counter.
pub(crate) fn handle_id(command: &str, n: u64) -> String {
    let mut slug = String::new();
    let mut previous_dash = true;
    for character in shell_title(command).chars() {
        if slug.len() >= 24 {
            break;
        }
        if character.is_ascii_alphanumeric() {
            slug.push(character.to_ascii_lowercase());
            previous_dash = false;
        } else if !previous_dash {
            slug.push('-');
            previous_dash = true;
        }
    }
    let slug = slug.trim_end_matches('-');
    let slug = if slug.is_empty() || slug == "task" {
        "sh"
    } else {
        slug
    };
    format!("{slug}_{n}")
}
