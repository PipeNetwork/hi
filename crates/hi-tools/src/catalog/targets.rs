/// Best-effort extraction of the primary target path from a tool call's JSON
/// arguments — the `path` field for read/write/edit/list, the `path`/`glob` for
/// grep. Returns `None` for tools without a meaningful single path (e.g.
/// `bash`, or a `grep` with only a pattern). Used by the agent to infer
/// within-batch dependencies: a read of a file a mutating call earlier in the
/// same batch targeted should observe that mutation, so it's serialized after.
/// Tolerant — a failed parse yields `None`, which the caller treats as "no
/// dependency inferred" (safe fallback to emission order).
pub fn target_path(name: &str, arguments: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(arguments).ok()?;
    match name {
        // `read` accepts compatibility paths or a typed workspace URI. Other
        // schemes have no local path until a host resolver routes them.
        "read" => crate::read::workspace_path_from_read_arguments(arguments),
        "write" | "edit" | "multi_edit" => value.get("path")?.as_str().map(str::to_string),
        // list's path is optional (defaults to ".").
        "list" => value.get("path")?.as_str().map(str::to_string),
        // Optional scope path for orientation tools (directory, not a single file).
        "repo_map" | "find_symbol" => value.get("path")?.as_str().map(str::to_string),
        // grep: prefer an explicit `path`; fall back to `glob` only as a hint
        // (a glob isn't a single file, so return None to avoid over-serializing).
        "grep" => value.get("path")?.as_str().map(str::to_string),
        // apply_patch: the patch text contains `*** Update File: <path>` (or
        // `*** Add File:`/`*** Delete File:`) directives. Return the path only
        // when the patch targets exactly one file. Multi-file patches have no
        // single target, so return None and let dependency inference treat the
        // mutation as unknown-path, serializing later reads conservatively.
        "apply_patch" => {
            let patch = value.get("patch")?.as_str()?;
            let mut paths: Vec<String> = patch
                .lines()
                .filter_map(|line| {
                    line.trim()
                        .strip_prefix("*** Update File: ")
                        .or_else(|| line.trim().strip_prefix("*** Add File: "))
                        .or_else(|| line.trim().strip_prefix("*** Delete File: "))
                        .map(str::trim)
                        .filter(|path| !path.is_empty())
                        .map(str::to_string)
                })
                .collect();
            paths.sort();
            paths.dedup();
            if paths.len() == 1 { paths.pop() } else { None }
        }
        // diff/glob/bash: no single meaningful target path for dep inference.
        _ => None,
    }
}

/// Every concrete path this call targeted. Used by the eval tape: a batched
/// `read` of several files must still count as a read of each one.
pub fn target_paths(name: &str, arguments: &str) -> Vec<String> {
    if let Some(one) = target_path(name, arguments) {
        return vec![one];
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(arguments) else {
        return Vec::new();
    };
    if name == "read" {
        return value
            .get("paths")
            .and_then(|v| v.as_array())
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.as_str().map(str::to_string))
                    .filter(|path| !path.is_empty())
                    .collect()
            })
            .unwrap_or_default();
    }
    Vec::new()
}
