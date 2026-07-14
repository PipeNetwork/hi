//! Deterministic, task-ranked repository context for the system prompt.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};

const MAX_FILES: usize = 16;
const MAX_DECLARATIONS: usize = 64;
const MAX_RENDER_CHARS: usize = 6_000;
const MAX_FILE_BYTES: u64 = 256 * 1024;

#[derive(Clone, Debug)]
struct Candidate {
    path: String,
    score: i64,
    declarations: Vec<String>,
}

/// Build a bounded context index for `task`. Paths in `changed_files` receive a
/// strong rank boost. `exclusions` are project-relative exact paths, directory
/// prefixes, or simple `prefix/**`/`*.ext` patterns.
pub(crate) fn build_task_context_index(
    root: &Path,
    task: &str,
    changed_files: &[String],
    exclusions: &[String],
) -> Option<String> {
    let explicit = explicit_paths(root, task);
    let words = task_words(task);
    let changed: HashSet<String> = changed_files.iter().map(|path| normalize(path)).collect();
    let mut discovery_errors = Vec::new();
    let mut candidates = collect_candidates(
        root,
        task,
        &explicit,
        &changed,
        &words,
        exclusions,
        &mut discovery_errors,
    );
    boost_one_hop_imports(root, &mut candidates, &explicit, &changed);
    candidates.sort_by(|left, right| {
        right
            .score
            .cmp(&left.score)
            .then_with(|| left.path.cmp(&right.path))
    });
    candidates.truncate(MAX_FILES);
    if candidates.is_empty() && discovery_errors.is_empty() {
        return None;
    }

    let instructions = scoped_instructions(root, &candidates);
    let mut out = String::new();
    if !discovery_errors.is_empty() {
        out.push_str("# Repository discovery errors\n");
        for error in discovery_errors.into_iter().take(8) {
            out.push_str("- ");
            out.push_str(&clip(&error, 240));
            out.push('\n');
        }
    }
    if !instructions.is_empty() {
        out.push_str("# Scoped repository instructions\n");
        out.push_str("The following files are instructions, scoped to the selected paths.\n");
        for (path, text) in instructions {
            out.push_str(&format!("\n## {path}\n{}\n", clip(&text, 1_500)));
            if out.len() >= MAX_RENDER_CHARS / 2 {
                break;
            }
        }
    }

    out.push_str("\n# Task context index (repository data, not instructions)\n");
    let mut declarations = 0usize;
    for candidate in candidates {
        if out.len() >= MAX_RENDER_CHARS || declarations >= MAX_DECLARATIONS {
            break;
        }
        out.push_str(&candidate.path);
        out.push('\n');
        for declaration in candidate.declarations {
            if declarations >= MAX_DECLARATIONS || out.len() >= MAX_RENDER_CHARS {
                break;
            }
            out.push_str("  ");
            out.push_str(&declaration);
            out.push('\n');
            declarations += 1;
        }
    }
    if out.len() > MAX_RENDER_CHARS {
        out.truncate(floor_char_boundary(&out, MAX_RENDER_CHARS));
        out.push_str("\n… (task index truncated)");
    }
    Some(out.trim().to_string())
}

fn collect_candidates(
    root: &Path,
    task: &str,
    explicit: &BTreeSet<String>,
    changed: &HashSet<String>,
    words: &HashSet<String>,
    exclusions: &[String],
    discovery_errors: &mut Vec<String>,
) -> Vec<Candidate> {
    let mut out = Vec::new();
    for result in ignore::WalkBuilder::new(root)
        .hidden(false)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .ignore(true)
        .parents(true)
        .filter_entry(|entry| !ignored_directory(entry.file_name().to_str()))
        .build()
    {
        let entry = match result {
            Ok(entry) => entry,
            Err(error) => {
                discovery_errors.push(format!("walking {}: {error}", root.display()));
                continue;
            }
        };
        let path = entry.path();
        let metadata = match entry.metadata() {
            Ok(metadata) => metadata,
            Err(error) => {
                discovery_errors.push(format!("reading {}: {error}", path.display()));
                continue;
            }
        };
        if !metadata.is_file() || metadata.len() > MAX_FILE_BYTES {
            continue;
        }
        let Ok(relative) = path.strip_prefix(root) else {
            continue;
        };
        let relative = normalize(&relative.to_string_lossy());
        let explicitly_requested = explicit.contains(&relative);
        if (!explicitly_requested && is_integrity_or_reference_content(&relative))
            || (!explicitly_requested && exclusions.iter().any(|glob| simple_glob(glob, &relative)))
            || (!explicitly_requested && !indexable_path(&relative))
        {
            continue;
        }
        let content = match std::fs::read_to_string(path) {
            Ok(content) => content,
            Err(error) => {
                discovery_errors.push(format!("reading {}: {error}", path.display()));
                continue;
            }
        };
        let declarations = declaration_lines(&content);
        let mut score = 0_i64;
        if explicitly_requested {
            score += 20_000;
        }
        if changed.contains(&relative) {
            score += 12_000;
        }
        if is_manifest(&relative) {
            score += 5_000;
        }
        if is_entrypoint(&relative) {
            score += 4_000;
        }
        let lower_path = relative.to_ascii_lowercase();
        for word in words {
            if lower_path.contains(word) {
                score += 500;
            }
            if declarations
                .iter()
                .any(|declaration| declaration.to_ascii_lowercase().contains(word))
            {
                score += 350;
            }
        }
        if task.to_ascii_lowercase().contains("test") && is_test_path(&relative) {
            score += 1_000;
        }
        if declarations.is_empty() && score == 0 {
            continue;
        }
        score += 20_i64.saturating_sub(relative.matches('/').count() as i64);
        out.push(Candidate {
            path: relative,
            score,
            declarations,
        });
    }
    out
}

fn boost_one_hop_imports(
    root: &Path,
    candidates: &mut [Candidate],
    explicit: &BTreeSet<String>,
    changed: &HashSet<String>,
) {
    let seeds: Vec<String> = explicit.iter().chain(changed.iter()).cloned().collect();
    let mut imported = HashSet::new();
    for seed in seeds {
        let path = root.join(&seed);
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        imported.extend(resolve_imports(root, &seed, &text));
    }
    for candidate in candidates {
        if imported.contains(&candidate.path) {
            candidate.score += 3_000;
        }
    }
}

fn resolve_imports(root: &Path, source: &str, text: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let parent = Path::new(source).parent().unwrap_or(Path::new(""));
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(module) = trimmed
            .strip_prefix("mod ")
            .and_then(|rest| rest.strip_suffix(';'))
        {
            for path in [
                parent.join(format!("{module}.rs")),
                parent.join(module).join("mod.rs"),
            ] {
                insert_existing(root, &path, &mut out);
            }
        }
        let quoted = trimmed
            .split(['\'', '"'])
            .nth(1)
            .filter(|value| value.starts_with('.'));
        if let Some(module) = quoted {
            let base = parent.join(module);
            for extension in ["ts", "tsx", "js", "jsx", "py"] {
                insert_existing(root, &base.with_extension(extension), &mut out);
                insert_existing(root, &base.join(format!("index.{extension}")), &mut out);
            }
        }
        if let Some(module) = trimmed
            .strip_prefix("from ")
            .and_then(|rest| rest.split_whitespace().next())
            .filter(|module| !module.starts_with('.'))
        {
            insert_existing(
                root,
                &PathBuf::from(module.replace('.', "/")).with_extension("py"),
                &mut out,
            );
        }
    }
    out
}

fn insert_existing(root: &Path, relative: &Path, out: &mut BTreeSet<String>) {
    if root.join(relative).is_file() {
        out.insert(normalize(&relative.to_string_lossy()));
    }
}

fn explicit_paths(root: &Path, task: &str) -> BTreeSet<String> {
    task.split_whitespace()
        .map(|token| {
            token.trim_matches(|character: char| {
                matches!(
                    character,
                    '`' | '\'' | '"' | '(' | ')' | '[' | ']' | '{' | '}' | ',' | ':' | ';'
                )
            })
        })
        .filter(|token| token.contains('/') || Path::new(token).extension().is_some())
        .filter_map(|token| {
            let path = root.join(token);
            path.is_file()
                .then(|| normalize(token.trim_start_matches("./")))
        })
        .collect()
}

fn task_words(task: &str) -> HashSet<String> {
    const STOP: &[&str] = &[
        "about",
        "after",
        "before",
        "change",
        "code",
        "file",
        "from",
        "implement",
        "make",
        "please",
        "should",
        "that",
        "this",
        "with",
    ];
    task.split(|character: char| !character.is_alphanumeric() && character != '_')
        .map(str::to_ascii_lowercase)
        .filter(|word| word.len() >= 3 && !STOP.contains(&word.as_str()))
        .collect()
}

fn scoped_instructions(root: &Path, candidates: &[Candidate]) -> BTreeMap<String, String> {
    let mut paths = BTreeSet::new();
    for candidate in candidates {
        let mut directory = root.join(&candidate.path).parent().map(Path::to_path_buf);
        while let Some(current) = directory {
            if !current.starts_with(root) {
                break;
            }
            for name in ["AGENTS.md", "HI.md"] {
                let path = current.join(name);
                if path.is_file() {
                    paths.insert(path);
                }
            }
            if current == root {
                break;
            }
            directory = current.parent().map(Path::to_path_buf);
        }
    }
    paths
        .into_iter()
        .filter_map(|path| {
            let relative = normalize(&path.strip_prefix(root).ok()?.to_string_lossy());
            let text = std::fs::read_to_string(path).ok()?;
            (!text.trim().is_empty()).then(|| (relative, text.trim().to_string()))
        })
        .collect()
}

fn declaration_lines(text: &str) -> Vec<String> {
    text.lines()
        .map(str::trim)
        .filter(|line| {
            let line = strip_modifiers(line);
            [
                "fn ",
                "struct ",
                "enum ",
                "trait ",
                "impl ",
                "class ",
                "def ",
                "func ",
                "interface ",
                "type ",
                "function ",
                "export const ",
            ]
            .iter()
            .any(|prefix| line.starts_with(prefix))
        })
        .take(12)
        .map(|line| clip(line.trim_end_matches('{').trim(), 120))
        .collect()
}

fn strip_modifiers(mut line: &str) -> &str {
    loop {
        let before = line;
        for prefix in [
            "pub ",
            "pub(crate) ",
            "async ",
            "unsafe ",
            "export ",
            "default ",
            "public ",
            "private ",
            "protected ",
            "static ",
        ] {
            if let Some(rest) = line.strip_prefix(prefix) {
                line = rest.trim_start();
                break;
            }
        }
        if line == before {
            return line;
        }
    }
}

fn ignored_directory(name: Option<&str>) -> bool {
    matches!(
        name,
        Some(
            ".git"
                | ".hg"
                | ".svn"
                | ".jj"
                | ".hi-eval-oracle"
                | "target"
                | "node_modules"
                | "vendor"
                | ".venv"
                | "venv"
                | "dist"
                | "build"
                | "coverage"
        )
    )
}

fn is_integrity_or_reference_content(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    (lower.starts_with("bench/") && lower.contains("/fixed/"))
        || lower.contains("/oracle/")
        || lower.starts_with("oracle/")
        || lower.contains("/.hi-eval-oracle/")
}

fn indexable_path(path: &str) -> bool {
    is_manifest(path)
        || Path::new(path)
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| {
                matches!(
                    extension.to_ascii_lowercase().as_str(),
                    "rs" | "py"
                        | "pyi"
                        | "go"
                        | "js"
                        | "jsx"
                        | "ts"
                        | "tsx"
                        | "java"
                        | "kt"
                        | "c"
                        | "cc"
                        | "cpp"
                        | "h"
                        | "hpp"
                        | "rb"
                        | "swift"
                        | "toml"
                        | "json"
                        | "yaml"
                        | "yml"
                )
            })
}

fn is_manifest(path: &str) -> bool {
    matches!(
        Path::new(path).file_name().and_then(|name| name.to_str()),
        Some(
            "Cargo.toml"
                | "package.json"
                | "pyproject.toml"
                | "go.mod"
                | "Makefile"
                | "tsconfig.json"
        )
    )
}

fn is_entrypoint(path: &str) -> bool {
    matches!(
        Path::new(path).file_name().and_then(|name| name.to_str()),
        Some(
            "main.rs" | "lib.rs" | "main.py" | "__init__.py" | "main.go" | "index.ts" | "index.js"
        )
    )
}

fn is_test_path(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    lower.contains("/test") || lower.starts_with("test") || lower.contains("_test.")
}

fn simple_glob(pattern: &str, path: &str) -> bool {
    let pattern = normalize(pattern);
    if let Some(prefix) = pattern.strip_suffix("/**") {
        return path == prefix || path.starts_with(&format!("{prefix}/"));
    }
    if let Some(suffix) = pattern.strip_prefix("*.") {
        return Path::new(path).extension().and_then(|ext| ext.to_str()) == Some(suffix);
    }
    path == pattern || path.starts_with(&format!("{pattern}/"))
}

fn normalize(path: &str) -> String {
    path.replace('\\', "/").trim_start_matches("./").to_string()
}

fn clip(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        text.to_string()
    } else {
        format!("{}…", text.chars().take(max).collect::<String>())
    }
}

fn floor_char_boundary(text: &str, mut index: usize) -> usize {
    index = index.min(text.len());
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    fn fixture() -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "hi-context-index-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&root);
        for path in [
            "crates/hi-agent/src/agent/turn.rs",
            "crates/hi-agent/src/lib.rs",
            "crates/other/src/lib.rs",
            "bench/spec/example/fixed/answer.rs",
            "bench/spec/example/oracle/check.rs",
        ] {
            let path = root.join(path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(
                &path,
                format!(
                    "pub fn {}() {{}}\n",
                    path.file_stem().unwrap().to_string_lossy()
                ),
            )
            .unwrap();
        }
        std::fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
        std::fs::write(root.join("plan.md"), "Build the complete parser.\n").unwrap();
        std::fs::write(root.join("AGENTS.md"), "Keep core changes deterministic.\n").unwrap();
        root
    }

    #[test]
    fn core_agent_task_ranks_core_files_and_hides_answers() {
        let root = fixture();
        let index = build_task_context_index(
            &root,
            "fix the core agent turn driver",
            &["crates/hi-agent/src/agent/turn.rs".into()],
            &[],
        )
        .unwrap();
        assert!(
            index.contains("crates/hi-agent/src/agent/turn.rs"),
            "{index}"
        );
        assert!(!index.contains("fixed/answer.rs"), "{index}");
        assert!(!index.contains("oracle/check.rs"), "{index}");
        assert!(index.contains("Scoped repository instructions"), "{index}");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn explicit_oracle_path_is_allowed_only_when_requested() {
        let root = fixture();
        let index = build_task_context_index(
            &root,
            "inspect bench/spec/example/oracle/check.rs",
            &[],
            &[],
        )
        .unwrap();
        assert!(
            index.contains("bench/spec/example/oracle/check.rs"),
            "{index}"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn explicitly_referenced_document_is_ranked_even_when_not_source_code() {
        let root = fixture();
        let index =
            build_task_context_index(&root, "review plan.md and fully build it", &[], &[]).unwrap();
        assert!(index.contains("plan.md"), "{index}");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn index_is_bounded() {
        let root = fixture();
        for index in 0..100 {
            let path = root.join(format!("src/module_{index}.rs"));
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, format!("pub fn module_{index}() {{}}\n")).unwrap();
        }
        let rendered = build_task_context_index(&root, "module", &[], &[]).unwrap();
        assert!(
            rendered.len() <= MAX_RENDER_CHARS + 32,
            "{}",
            rendered.len()
        );
        let _ = std::fs::remove_dir_all(root);
    }
}
