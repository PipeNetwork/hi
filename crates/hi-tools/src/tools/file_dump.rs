//! Map shell file dumps onto `read` so they keep the numbered-page budget.

use std::path::Path;

pub(super) fn file_dump_read_arguments(command: &str) -> Option<String> {
    let dump = file_dump_command(command)?;
    parse_file_dump_command(dump).or_else(|| parse_grep_dump_pipeline(dump))
}

fn file_dump_command(command: &str) -> Option<&str> {
    let trimmed = command.trim();
    if trimmed.is_empty()
        || trimmed.matches('|').count() > 1
        || trimmed
            .chars()
            .any(|ch| matches!(ch, '\n' | '\r' | '>' | '<' | '$' | '`'))
    {
        return None;
    }
    if trimmed.contains('&') && !trimmed.contains("&&") {
        return None;
    }
    let mut segments = Vec::new();
    for chunk in trimmed.split("&&") {
        for piece in chunk.split(';') {
            let piece = piece.trim();
            if !piece.is_empty() {
                segments.push(piece);
            }
        }
    }
    let (last, prefixes) = segments.split_last()?;
    if prefixes
        .iter()
        .copied()
        .any(|prefix| !is_banner_shell(prefix))
    {
        return None;
    }
    Some(*last)
}

fn is_banner_shell(command: &str) -> bool {
    let words: Vec<&str> = command.split_whitespace().collect();
    let Some(start) = words
        .iter()
        .position(|word| !is_env_assignment(word) && *word != "env")
    else {
        return true;
    };
    matches!(basename(words[start]), "echo" | "printf" | "true" | ":")
}

fn parse_file_dump_command(trimmed: &str) -> Option<String> {
    if trimmed.contains('|') {
        return None;
    }
    let words: Vec<&str> = trimmed.split_whitespace().collect();
    let start = words
        .iter()
        .position(|word| !is_env_assignment(word) && *word != "env")?;
    let program = basename(words[start]);
    let args: Vec<&str> = words[start + 1..].iter().copied().map(unquote).collect();
    match program {
        "cat" => parse_cat_dump(&args),
        "sed" => parse_sed_print_dump(&args),
        "head" => parse_head_dump(&args),
        _ => None,
    }
}

fn unquote(token: &str) -> &str {
    let bytes = token.as_bytes();
    if bytes.len() >= 2
        && ((bytes[0] == b'\'' && *bytes.last().unwrap() == b'\'')
            || (bytes[0] == b'"' && *bytes.last().unwrap() == b'"'))
    {
        &token[1..token.len() - 1]
    } else {
        token
    }
}

fn looks_like_glob(path: &str) -> bool {
    path.contains('*') || path.contains('?') || path.contains('[')
}

fn basename(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

fn is_env_assignment(tok: &str) -> bool {
    !tok.starts_with('-')
        && tok.split_once('=').is_some_and(|(k, _)| {
            !k.is_empty() && k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        })
}

fn read_args_json(path: &str, offset: Option<usize>, limit: Option<usize>) -> Option<String> {
    if path.is_empty() || path == "-" || looks_like_glob(path) {
        return None;
    }
    let mut map = serde_json::Map::new();
    map.insert("path".into(), serde_json::Value::String(path.to_string()));
    if let Some(offset) = offset {
        map.insert("offset".into(), serde_json::json!(offset));
    }
    if let Some(limit) = limit {
        map.insert("limit".into(), serde_json::json!(limit));
    }
    Some(serde_json::Value::Object(map).to_string())
}

fn parse_cat_dump(args: &[&str]) -> Option<String> {
    let mut paths = Vec::new();
    for arg in args {
        if *arg == "--" {
            continue;
        }
        if matches!(
            *arg,
            "-n" | "-b" | "-s" | "-u" | "--number" | "--number-nonblank" | "--squeeze-blank"
        ) {
            continue;
        }
        if arg.starts_with('-') {
            return None;
        }
        paths.push(*arg);
    }
    match paths.as_slice() {
        [path] => read_args_json(path, None, None),
        paths if (1..=32).contains(&paths.len()) => {
            Some(serde_json::json!({ "paths": paths }).to_string())
        }
        _ => None,
    }
}

fn parse_sed_print_dump(args: &[&str]) -> Option<String> {
    let mut quiet = false;
    let mut script: Option<&str> = None;
    let mut path: Option<&str> = None;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i];
        if arg == "--" {
            i += 1;
            continue;
        }
        if matches!(arg, "-n" | "--quiet" | "--silent") {
            quiet = true;
            i += 1;
            continue;
        }
        if matches!(arg, "-e" | "--expression") {
            script = Some(args.get(i + 1).copied()?);
            i += 2;
            continue;
        }
        if let Some(expr) = arg.strip_prefix("-e")
            && !expr.is_empty()
        {
            script = Some(expr);
            i += 1;
            continue;
        }
        if arg == "-i" || arg == "--in-place" || arg.starts_with("-i") {
            return None;
        }
        if arg.starts_with('-') {
            return None;
        }
        if script.is_none() {
            script = Some(arg);
        } else if path.is_none() {
            path = Some(arg);
        } else {
            return None;
        }
        i += 1;
    }
    if !quiet {
        return None;
    }
    let (offset, limit) = parse_sed_line_range(script?)?;
    read_args_json(path?, Some(offset), Some(limit))
}

fn parse_sed_line_range(script: &str) -> Option<(usize, usize)> {
    let script = unquote(script).strip_suffix('p')?;
    if let Some((start, end)) = script.split_once(',') {
        let start: usize = start.parse().ok()?;
        let end: usize = end.parse().ok()?;
        if start == 0 || end < start {
            return None;
        }
        Some((start, end.saturating_sub(start).saturating_add(1)))
    } else {
        let line: usize = script.parse().ok()?;
        (line > 0).then_some((line, 1))
    }
}

fn parse_head_dump(args: &[&str]) -> Option<String> {
    let mut limit: Option<usize> = None;
    let mut path: Option<&str> = None;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i];
        if arg == "--" {
            i += 1;
            continue;
        }
        if arg == "-n" || arg == "--lines" {
            limit = Some(parse_line_count(args.get(i + 1).copied()?)?);
            i += 2;
            continue;
        }
        if let Some(rest) = arg.strip_prefix("--lines=") {
            limit = Some(parse_line_count(rest)?);
            i += 1;
            continue;
        }
        if let Some(rest) = arg.strip_prefix("-n")
            && !rest.is_empty()
            && rest.bytes().all(|b| b.is_ascii_digit())
        {
            limit = Some(parse_line_count(rest)?);
            i += 1;
            continue;
        }
        if arg.starts_with('-') && arg.len() > 1 && arg[1..].bytes().all(|b| b.is_ascii_digit()) {
            limit = Some(parse_line_count(&arg[1..])?);
            i += 1;
            continue;
        }
        if arg.starts_with('-') {
            return None;
        }
        if path.is_some() {
            return None;
        }
        path = Some(arg);
        i += 1;
    }
    read_args_json(path?, None, Some(limit.unwrap_or(10)))
}

fn parse_line_count(token: &str) -> Option<usize> {
    let n: usize = unquote(token).parse().ok()?;
    (n > 0).then_some(n)
}

fn is_grep_full_file_dump_pattern(pattern: &str) -> bool {
    matches!(unquote(pattern), "" | "^" | "." | ".*" | "^.*$")
}

fn parse_grep_dump_pipeline(command: &str) -> Option<String> {
    let (grep_cmd, range) = match command.split_once('|') {
        Some((left, right)) => {
            if right.contains('|') {
                return None;
            }
            (left.trim(), Some(parse_stdin_sed_range(right.trim())?))
        }
        None => (command.trim(), None),
    };
    let path = parse_grep_dump_path(grep_cmd)?;
    match range {
        Some((offset, limit)) => read_args_json(&path, Some(offset), Some(limit)),
        None => read_args_json(&path, None, None),
    }
}

fn parse_grep_dump_path(command: &str) -> Option<String> {
    let words: Vec<&str> = command.split_whitespace().collect();
    let start = words
        .iter()
        .position(|word| !is_env_assignment(word) && *word != "env")?;
    if !matches!(basename(words[start]), "grep" | "ggrep" | "egrep") {
        return None;
    }
    let mut pattern: Option<&str> = None;
    let mut path: Option<&str> = None;
    for arg in words[start + 1..].iter().copied() {
        let arg = unquote(arg);
        if arg == "--" {
            continue;
        }
        if matches!(arg, "-n" | "--line-number") {
            continue;
        }
        if arg.starts_with('-') {
            return None;
        }
        if pattern.is_none() {
            pattern = Some(arg);
        } else if path.is_none() {
            path = Some(arg);
        } else {
            return None;
        }
    }
    if !is_grep_full_file_dump_pattern(pattern?) {
        return None;
    }
    Some(path?.to_string())
}

fn parse_stdin_sed_range(command: &str) -> Option<(usize, usize)> {
    let words: Vec<&str> = command.split_whitespace().collect();
    let start = words
        .iter()
        .position(|word| !is_env_assignment(word) && *word != "env")?;
    if basename(words[start]) != "sed" {
        return None;
    }
    let mut quiet = false;
    let mut script: Option<&str> = None;
    for arg in words[start + 1..].iter().copied().map(unquote) {
        if arg == "--" {
            continue;
        }
        if matches!(arg, "-n" | "--quiet" | "--silent") {
            quiet = true;
            continue;
        }
        if arg.starts_with('-') || script.is_some() {
            return None;
        }
        script = Some(arg);
    }
    if !quiet {
        return None;
    }
    parse_sed_line_range(script?)
}

pub(super) fn file_dump_is_read_eligible(root: &Path, arguments: &str) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(arguments) else {
        return false;
    };
    let mut paths = Vec::new();
    if let Some(path) = value.get("path").and_then(|v| v.as_str()) {
        paths.push(path);
    }
    if let Some(list) = value.get("paths").and_then(|v| v.as_array()) {
        for path in list {
            let Some(path) = path.as_str() else {
                return false;
            };
            paths.push(path);
        }
    }
    if paths.is_empty() {
        return false;
    }
    paths
        .iter()
        .all(|path| workspace_file_fits_read(root, path))
}

fn workspace_file_fits_read(root: &Path, rel: &str) -> bool {
    if rel.is_empty() || rel.starts_with('/') || rel.split(['/', '\\']).any(|part| part == "..") {
        return false;
    }
    let path = root.join(rel);
    let Ok(meta) = std::fs::metadata(&path) else {
        return false;
    };
    meta.is_file() && meta.len() <= crate::read::MAX_READ_FILE_BYTES
}
