//! Preflight call planning: [`read_only_preflight_initial_calls`] and
//! helpers for entrypoint detection, grep-output path extraction, and
//! implementation validation detection. Uses constants from
//! [`constants`](super::constants) and types from [`types`](super::types).

use super::constants::{
    GAP_PREFLIGHT_PATTERN, READ_ONLY_PREFLIGHT_DIFF_MAX_LINES, READ_ONLY_PREFLIGHT_GREP_MAX_LINES,
    SECURITY_PREFLIGHT_PATTERN,
};
use super::types::{PreflightCall, ReviewIntent};
pub(crate) fn read_only_preflight_initial_calls_in(
    root: &std::path::Path,
    intent: ReviewIntent,
) -> Vec<PreflightCall> {
    let mut calls = Vec::new();
    if matches!(
        intent,
        ReviewIntent::Review | ReviewIntent::Status | ReviewIntent::Roadmap | ReviewIntent::Gaps
    ) {
        calls.push(PreflightCall::new("diff", serde_json::json!({})));
    }
    push_preflight_read_if_exists_in(&mut calls, root, "Cargo.toml", 100);
    if !matches!(intent, ReviewIntent::Security) {
        push_preflight_read_if_exists_in(&mut calls, root, "README.md", 100);
    }

    match intent {
        ReviewIntent::Security => {
            calls.push(PreflightCall::new(
                "grep",
                serde_json::json!({
                    "pattern": SECURITY_PREFLIGHT_PATTERN,
                    "context": 0,
                    "glob": "*.rs",
                }),
            ));
        }
        ReviewIntent::Roadmap | ReviewIntent::Gaps => {
            calls.extend(
                preflight_entrypoint_candidates_in(root)
                    .into_iter()
                    .take(3)
                    .map(|path| PreflightCall::read(path, 180)),
            );
            calls.push(PreflightCall::new(
                "grep",
                serde_json::json!({
                    "pattern": GAP_PREFLIGHT_PATTERN,
                    "context": 0,
                }),
            ));
        }
        ReviewIntent::Review | ReviewIntent::Status => {
            calls.extend(
                preflight_entrypoint_candidates_in(root)
                    .into_iter()
                    .take(3)
                    .map(|path| PreflightCall::read(path, 180)),
            );
        }
    }
    calls
}

fn push_preflight_read_if_exists_in(
    calls: &mut Vec<PreflightCall>,
    root: &std::path::Path,
    path: &str,
    limit: u32,
) {
    if root.join(path).is_file() {
        calls.push(PreflightCall::read(path, limit));
    }
}

pub(crate) fn preflight_entrypoint_candidates_in(root: &std::path::Path) -> Vec<String> {
    let mut candidates = Vec::new();
    for path in ["src/lib.rs", "src/main.rs"] {
        if root.join(path).is_file() {
            candidates.push(path.to_string());
        }
    }

    let Ok(entries) = std::fs::read_dir(root.join("crates")) else {
        return candidates;
    };
    let mut crate_dirs = entries
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.file_type().is_ok_and(|file_type| file_type.is_dir()))
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    crate_dirs.sort();
    for crate_dir in crate_dirs {
        for file in ["src/lib.rs", "src/main.rs"] {
            let path = crate_dir.join(file);
            if path.is_file() {
                candidates.push(path.to_string_lossy().to_string());
            }
        }
    }
    candidates
}

pub(crate) fn paths_from_grep_output_in(root: &std::path::Path, output: &str) -> Vec<String> {
    let mut paths = Vec::new();
    for line in output.lines() {
        let Some((path, _)) = line.split_once(':') else {
            continue;
        };
        let path = path.trim();
        if path.is_empty()
            || path.starts_with("no matches")
            || path.starts_with("Error:")
            || !root.join(path).is_file()
            || paths.iter().any(|existing| existing == path)
        {
            continue;
        }
        paths.push(path.to_string());
        if paths.len() >= 4 {
            break;
        }
    }
    paths
}

pub(crate) fn preflight_path_relevant_for_intent(intent: ReviewIntent, path: &str) -> bool {
    if !matches!(intent, ReviewIntent::Security) {
        return true;
    }
    let lower = path.to_ascii_lowercase();
    matches!(
        lower.rsplit('.').next(),
        Some(
            "rs" | "toml"
                | "lock"
                | "sh"
                | "bash"
                | "zsh"
                | "py"
                | "js"
                | "jsx"
                | "ts"
                | "tsx"
                | "go"
                | "java"
                | "kt"
                | "kts"
                | "rb"
                | "php"
                | "c"
                | "cc"
                | "cpp"
                | "h"
                | "hpp"
        )
    )
}

pub(crate) fn compact_preflight_tool_output(name: &str, output: &str) -> String {
    let max_lines = match name {
        "grep" => READ_ONLY_PREFLIGHT_GREP_MAX_LINES,
        "diff" => READ_ONLY_PREFLIGHT_DIFF_MAX_LINES,
        _ => return output.to_string(),
    };
    let mut lines = output.lines().collect::<Vec<_>>();
    if lines.len() <= max_lines {
        return output.to_string();
    }
    let omitted = lines.len().saturating_sub(max_lines);
    lines.truncate(max_lines);
    format!(
        "{}\n[preflight {name} output truncated: {omitted} additional line(s) omitted]",
        lines.join("\n")
    )
}

pub(crate) fn implementation_preflight_command() -> &'static str {
    r#"set +e
printf '[git_status]\n'
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git status --short 2>&1 | head -80
else
  printf 'not a git repository\n'
fi
printf '\n[git_diff_stat]\n'
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git diff --stat 2>&1 | head -80
else
  printf 'not a git repository\n'
fi
printf '\n[workspace_manifests]\n'
find . \( -path './.git' -o -path './target' -o -path './node_modules' -o -path './.venv' -o -path './venv' -o -path './vendor' -o -path './models' -o -path './.cache' -o -path './dist' -o -path './build' -o -path './.next' -o -path './.turbo' -o -path './coverage' \) -prune -o \( -name Cargo.toml -o -name package.json -o -name pyproject.toml -o -name go.mod -o -name Makefile -o -name justfile \) -print | head -80
printf '\n[readme_docs]\n'
find . -maxdepth 3 \( -path './.git' -o -path './target' -o -path './node_modules' -o -path './.venv' -o -path './venv' -o -path './vendor' -o -path './models' -o -path './.cache' -o -path './dist' -o -path './build' -o -path './.next' -o -path './.turbo' -o -path './coverage' \) -prune -o \( -iname 'README*' -o -iname 'DESIGN*' -o -path './docs/*' \) -print | head -80
printf '\n[likely_entrypoints]\n'
find . \( -path './.git' -o -path './target' -o -path './node_modules' -o -path './.venv' -o -path './venv' -o -path './vendor' -o -path './models' -o -path './.cache' -o -path './dist' -o -path './build' -o -path './.next' -o -path './.turbo' -o -path './coverage' \) -prune -o \( -path './src/main.rs' -o -path './src/lib.rs' -o -path './src/bin/*.rs' -o -path './main.py' -o -path './app.py' -o -path './src/index.*' -o -path './src/main.*' -o -path './app/page.*' \) -print | head -80
printf '\n[detected_verification]\n'
if find . -maxdepth 3 \( -path './.git' -o -path './target' -o -path './node_modules' -o -path './.venv' -o -path './venv' -o -path './vendor' -o -path './models' -o -path './.cache' -o -path './dist' -o -path './build' -o -path './.next' -o -path './.turbo' -o -path './coverage' \) -prune -o -name Cargo.toml -print -quit | grep -q .; then
  printf 'primary=cargo test\n'
  printf 'alternates=cargo check; cargo build\n'
elif [ -f package.json ]; then
  if grep -q '"test"' package.json; then printf 'primary=npm test\n'; else printf 'primary=npm run build\n'; fi
  printf 'alternates=npm run check; npm run lint\n'
elif [ -f pyproject.toml ]; then
  printf 'primary=python -m pytest\n'
elif [ -f go.mod ]; then
  printf 'primary=go test ./...\n'
elif [ -f Makefile ]; then
  printf 'primary=make test\n'
else
  printf 'primary=inspect manifests first\n'
fi
"#
}

pub(crate) fn preferred_validation_from_preflight(output: &str) -> Option<String> {
    for line in output.lines() {
        let Some(command) = line.trim().strip_prefix("primary=") else {
            continue;
        };
        let command = command.trim();
        if !command.is_empty() && command != "inspect manifests first" {
            return Some(command.to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{ReviewIntent, paths_from_grep_output_in, read_only_preflight_initial_calls_in};

    #[test]
    fn preflight_planning_uses_selected_workspace_root() {
        let root = std::env::temp_dir().join(format!(
            "hi-preflight-root-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(root.join("crates/demo/src")).unwrap();
        std::fs::write(root.join("Cargo.toml"), "[workspace]\nmembers=[]\n").unwrap();
        std::fs::write(root.join("README.md"), "# demo\n").unwrap();
        std::fs::write(root.join("crates/demo/src/lib.rs"), "pub fn demo() {}\n").unwrap();

        let calls = read_only_preflight_initial_calls_in(&root, ReviewIntent::Review);
        let paths = calls
            .iter()
            .filter(|call| call.name == "read")
            .map(|call| call.arguments.clone())
            .collect::<Vec<_>>();
        assert!(paths.iter().any(|args| args.contains("Cargo.toml")));
        assert!(paths.iter().any(|args| args.contains("README.md")));
        assert!(
            paths
                .iter()
                .any(|args| args.contains("crates/demo/src/lib.rs"))
        );

        let grep = "crates/demo/src/lib.rs:1:pub fn demo() {}\n";
        assert_eq!(
            paths_from_grep_output_in(&root, grep),
            vec!["crates/demo/src/lib.rs"]
        );

        let _ = std::fs::remove_dir_all(root);
    }
}
