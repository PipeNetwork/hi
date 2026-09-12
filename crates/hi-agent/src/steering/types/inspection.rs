use super::*;

/// A stable signature for a read-only inspection call, used to detect rounds
/// that re-inspect already-seen evidence. Returns `None` for mutating or
/// unclassified tools (those always count as potentially new evidence). The
/// signature includes read pagination and grep context because those
/// arguments change the evidence returned by the tool. A malformed read-only
/// call returns `None`; callers treat that as potentially new evidence so the
/// normal tool execution path can report the argument error.
///
/// [`EvidenceTracker::round_adds_evidence`] treats every new page of a still-
/// truncated file as new evidence. The offset stays in the signature so
/// identical pages still fold without imposing an arbitrary page count.
pub(crate) fn inspection_signature(name: &str, arguments: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(arguments).ok()?;
    match name {
        "read" => {
            let mut paths = hi_tools::target_paths("read", arguments);
            if paths.is_empty() {
                return None;
            }
            paths.sort_unstable();
            paths.dedup();
            let path = paths.join("\u{1f}");
            const DEFAULT_READ_LIMIT: u64 = 2000;
            let offset = optional_u64_field(&value, "offset")?.unwrap_or(1).max(1);
            let limit = optional_u64_field(&value, "limit")?
                .map(|n| n.max(1))
                .filter(|&n| n != DEFAULT_READ_LIMIT)
                .map_or_else(|| "default".to_string(), |n| n.to_string());
            Some(format!("read:{path}:{offset}:{limit}"))
        }
        "list" => {
            let path = value.get("path").and_then(|v| v.as_str()).unwrap_or(".");
            Some(format!("list:{path}"))
        }
        "grep" => {
            let pattern = value.get("pattern")?.as_str()?;
            let glob = value.get("glob").and_then(|v| v.as_str()).unwrap_or("");
            let path = value.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let context = optional_u64_field(&value, "context")?.unwrap_or(0);
            Some(format!("grep:{pattern}:{glob}:{path}:{context}"))
        }
        "glob" => {
            let pattern = value.get("pattern")?.as_str()?;
            let path = value.get("path").and_then(|v| v.as_str()).unwrap_or("");
            Some(format!("glob:{pattern}:{path}"))
        }
        "repo_map" => {
            let task = value.get("task").and_then(|v| v.as_str()).unwrap_or("");
            let path = value.get("path").and_then(|v| v.as_str()).unwrap_or("");
            Some(format!("repo_map:{task}:{path}"))
        }
        "find_symbol" => {
            let query = value.get("query")?.as_str()?;
            let path = value.get("path").and_then(|v| v.as_str()).unwrap_or("");
            Some(format!("find_symbol:{query}:{path}"))
        }
        "obs_recall" => {
            let id = value.get("id")?.as_str()?;
            let offset = value.get("offset").and_then(|v| v.as_u64()).unwrap_or(0);
            Some(format!("obs_recall:{id}:{offset}"))
        }
        "explore" => {
            let task = value
                .get("task")
                .or_else(|| value.get("prompt"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            Some(format!("explore:{task}"))
        }
        "bash_output" | "bash_kill" => {
            let id = value.get("id")?.as_str()?;
            if id.is_empty() {
                return None;
            }
            Some(format!("{name}:{id}"))
        }
        "bash" => bash_inspection_signature(arguments)
            .map(|command| format!("bash:inspection:{command}"))
            .or_else(|| bash_no_progress_signature(arguments).map(|sig| format!("bash:{sig}"))),
        _ => None,
    }
}

fn optional_u64_field(value: &serde_json::Value, field: &str) -> Option<Option<u64>> {
    match value.get(field) {
        Some(v) if v.is_null() => Some(None),
        Some(v) => v.as_u64().map(Some),
        None => Some(None),
    }
}

/// Coarse signature for tool failures that cannot produce new evidence on
/// retry with different arguments (missing `rg` under Seatbelt, etc.).
pub(crate) fn inspection_infrastructure_error_signature(
    name: &str,
    output: &str,
) -> Option<String> {
    if !output.starts_with("Error:") && !output.contains("execvp()") {
        return None;
    }
    let lower = output.to_ascii_lowercase();
    let unavailable = lower.contains("execvp()")
        || (lower.contains("ripgrep") && lower.contains("unavailable"))
        || (name == "grep"
            && lower.contains("no such file")
            && (lower.contains("'rg'") || lower.contains("of 'rg'")));
    unavailable.then(|| format!("{name}:error:unavailable"))
}
