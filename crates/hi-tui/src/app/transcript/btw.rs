/// Compact tool-arg detail for the BTW pane timeline (path/pattern/command).
pub(super) fn btw_tool_detail(name: &str, arguments: &str) -> String {
    let v: serde_json::Value = serde_json::from_str(arguments).unwrap_or_default();
    let pick = |keys: &[&str]| {
        keys.iter()
            .find_map(|k| v.get(*k).and_then(|x| x.as_str()))
            .unwrap_or("")
            .to_string()
    };
    match name {
        "read" | "list" | "glob" | "diff" => pick(&["path", "target", "directory"]),
        "grep" => {
            let pat = pick(&["pattern", "query"]);
            let path = pick(&["path", "glob"]);
            if path.is_empty() {
                pat
            } else {
                format!("{pat} in {path}")
            }
        }
        "repo_map" | "find_symbol" => pick(&["task", "symbol", "query", "name"]),
        "web_search" | "web_fetch" => pick(&["query", "url"]),
        _ => pick(&["path", "command", "query", "task"]),
    }
}
