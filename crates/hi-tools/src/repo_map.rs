//! In-memory polyglot repository map and symbol index.
//!
//! Built on a cheap walk + line heuristics (not a full language server). The
//! agent uses this for first-turn orientation (`repo_map` / `find_symbol` tools
//! and the turn-setup seed) so the first reads hit the right files.
//!
//! Phase M: beyond bare symbol hits, ranking expands **companion paths**
//! (same-directory siblings, tests, package roots) and cheap **import edges**
//! so orientation seeds point at the local neighborhood, not only the one
//! matching symbol line.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::ToolOutcome;
use crate::condense::truncate;

const MAX_FILE_BYTES: u64 = 256 * 1024;
const MAX_SYMBOLS_PER_FILE: usize = 48;
const DEFAULT_MAP_FILES: usize = 40;
const DEFAULT_SYMBOL_HITS: usize = 24;
const MAX_MAP_RENDER_CHARS: usize = 8_000;
const MAX_ORIENTATION_CHARS: usize = 2_400;

#[derive(Clone, Debug)]
struct Symbol {
    path: String,
    line: u32,
    name: String,
    kind: &'static str,
}

#[derive(Clone, Debug)]
struct FileSummary {
    path: String,
    score_base: i64,
    declarations: Vec<String>,
    is_manifest: bool,
    is_entrypoint: bool,
    /// Unresolved import/mod tokens extracted from this file (local graph edges).
    import_hints: Vec<String>,
}

#[derive(Clone, Debug, Default)]
struct BuiltIndex {
    root: PathBuf,
    /// Cheap invalidation key: count + total size + max mtime nanos.
    fingerprint: u128,
    files: Vec<FileSummary>,
    symbols: Vec<Symbol>,
    /// path → paths that likely import or sit next to it (resolved after walk).
    neighbors: BTreeMap<String, Vec<String>>,
}

/// Process-local cache of the last built index for a workspace root.
#[derive(Debug, Default)]
pub struct RepoMapCache {
    index: Option<BuiltIndex>,
}

impl RepoMapCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.index = None;
    }

    fn get_or_build(&mut self, root: &Path) -> Result<&BuiltIndex> {
        let root = root
            .canonicalize()
            .with_context(|| format!("canonicalizing {}", root.display()))?;
        let fingerprint = fingerprint_workspace(&root)?;
        let needs_rebuild = self
            .index
            .as_ref()
            .is_none_or(|index| index.root != root || index.fingerprint != fingerprint);
        if needs_rebuild {
            self.index = Some(build_index(&root, fingerprint)?);
        }
        Ok(self.index.as_ref().expect("index just built or retained"))
    }
}

/// Ranked file/declaration map for orientation. Optional `task` boosts path and
/// symbol word hits; optional `path` scopes under a subdirectory.
pub(crate) async fn run_repo_map(
    root: &Path,
    cache: &std::sync::Mutex<RepoMapCache>,
    arguments: &str,
) -> Result<ToolOutcome> {
    #[derive(Deserialize)]
    struct Args {
        #[serde(default)]
        task: Option<String>,
        #[serde(default)]
        path: Option<String>,
        #[serde(default)]
        limit: Option<usize>,
    }
    let args: Args = crate::tools::parse(arguments)?;
    let limit = args.limit.unwrap_or(DEFAULT_MAP_FILES).clamp(1, 100);
    let scope = args
        .path
        .as_deref()
        .map(normalize)
        .filter(|path| !path.is_empty());
    let task = args.task.as_deref().unwrap_or("").trim();
    let rendered = {
        let mut guard = lock_cache(cache);
        let index = guard.get_or_build(root)?;
        render_repo_map(index, task, scope.as_deref(), limit)
    };
    Ok(ToolOutcome::plain(truncate(&rendered)))
}

/// Look up definitions by symbol name (case-insensitive substring).
pub(crate) async fn run_find_symbol(
    root: &Path,
    cache: &std::sync::Mutex<RepoMapCache>,
    arguments: &str,
) -> Result<ToolOutcome> {
    #[derive(Deserialize)]
    struct Args {
        query: String,
        #[serde(default)]
        limit: Option<usize>,
        #[serde(default)]
        path: Option<String>,
    }
    let args: Args = crate::tools::parse(arguments)?;
    let query = args.query.trim();
    if query.is_empty() {
        bail!("`find_symbol` requires a non-empty `query`");
    }
    if query.len() < 2 {
        bail!("`find_symbol` query must be at least 2 characters");
    }
    let limit = args.limit.unwrap_or(DEFAULT_SYMBOL_HITS).clamp(1, 100);
    let scope = args
        .path
        .as_deref()
        .map(normalize)
        .filter(|path| !path.is_empty());
    let rendered = {
        let mut guard = lock_cache(cache);
        let index = guard.get_or_build(root)?;
        render_symbol_hits(index, query, scope.as_deref(), limit)
    };
    Ok(ToolOutcome::plain(truncate(&rendered)))
}

/// Compact symbol-oriented seed for the system prompt on the first model call.
/// Returns `None` when the task has no usable identifier tokens or the workspace
/// has no matching symbols.
pub fn orientation_for_task(
    root: &Path,
    task: &str,
    cache: &std::sync::Mutex<RepoMapCache>,
) -> Option<String> {
    let words = task_words(task);
    if words.is_empty() {
        return None;
    }
    let mut guard = lock_cache(cache);
    let index = guard.get_or_build(root).ok()?;
    let mut hits = collect_symbol_hits(index, &words, None, 16);
    if hits.is_empty() {
        // Fall back to path-ranked map when tokens only match paths/manifests.
        let map = render_repo_map(index, task, None, 12);
        if map.contains("No indexable") || map.contains("(no files") {
            return None;
        }
        let mut out = String::from(
            "# Repo map seed (repository data, not instructions)\n\
             Prefer `find_symbol` / `repo_map` over blind `list` for orientation.\n",
        );
        out.push_str(&map);
        if out.len() > MAX_ORIENTATION_CHARS {
            out.truncate(floor_char_boundary(&out, MAX_ORIENTATION_CHARS));
            out.push_str("\n…");
        }
        return Some(out);
    }
    hits.sort_by(|left, right| {
        right
            .score
            .cmp(&left.score)
            .then_with(|| left.path.cmp(&right.path))
            .then_with(|| left.line.cmp(&right.line))
    });
    hits.truncate(12);
    let seed_paths: Vec<String> = hits.iter().map(|h| h.path.clone()).collect();
    let companions = companion_paths_for(index, &seed_paths, 8);

    let mut out = String::from(
        "# Symbol hits for this task (repository data, not instructions)\n\
         Prefer reading these before broad `list`/`grep`. Use `find_symbol` for more.\n",
    );
    for hit in &hits {
        out.push_str(&format!(
            "- `{}` {}:{} ({})\n",
            hit.name, hit.path, hit.line, hit.kind
        ));
        if out.len() >= MAX_ORIENTATION_CHARS {
            break;
        }
    }
    if !companions.is_empty() && out.len() < MAX_ORIENTATION_CHARS {
        out.push_str("\n# Nearby / imported (read if the hit alone is incomplete)\n");
        for path in &companions {
            out.push_str(&format!("- {path}\n"));
            if out.len() >= MAX_ORIENTATION_CHARS {
                break;
            }
        }
    }
    if out.len() > MAX_ORIENTATION_CHARS {
        out.truncate(floor_char_boundary(&out, MAX_ORIENTATION_CHARS));
        out.push_str("\n…");
    }
    Some(out)
}

/// Paths strongly suggested by symbol/path hits for `task` — used to boost the
/// existing task context index ranking. Includes companion/import neighbors
/// (Phase M) so multi-file neighborhoods surface, not only the matched symbol.
pub fn ranked_paths_for_task(
    root: &Path,
    task: &str,
    cache: &std::sync::Mutex<RepoMapCache>,
    limit: usize,
) -> Vec<String> {
    let words = task_words(task);
    if words.is_empty() {
        return Vec::new();
    }
    let mut guard = lock_cache(cache);
    let Ok(index) = guard.get_or_build(root) else {
        return Vec::new();
    };
    let mut paths = BTreeMap::<String, i64>::new();
    let primary_hits = collect_symbol_hits(index, &words, None, limit.saturating_mul(3));
    for hit in &primary_hits {
        *paths.entry(hit.path.clone()).or_default() += hit.score;
    }
    for file in &index.files {
        let mut score = 0_i64;
        let path_lower = file.path.to_ascii_lowercase();
        for word in &words {
            if path_lower.contains(word) {
                score += 50;
            }
        }
        if score > 0 {
            *paths.entry(file.path.clone()).or_default() += score + file.score_base;
        }
    }
    // Boost neighbors of the strongest hits (directory siblings, imports, package roots).
    let seed: Vec<String> = {
        let mut ranked: Vec<_> = paths.iter().map(|(p, s)| (p.clone(), *s)).collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.into_iter().take(6).map(|(p, _)| p).collect()
    };
    for path in companion_paths_for(index, &seed, limit.saturating_mul(2)) {
        // Weaker than a direct symbol hit, stronger than an unrelated file.
        *paths.entry(path).or_default() += 1_500;
    }
    let mut ranked = paths.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    ranked.truncate(limit.max(1));
    ranked.into_iter().map(|(path, _)| path).collect()
}

/// Paths near `seeds`: neighbors graph + same-stem tests. Excludes the seeds
/// themselves. Ordered by how many seeds point at them, then path.
fn companion_paths_for(index: &BuiltIndex, seeds: &[String], limit: usize) -> Vec<String> {
    if seeds.is_empty() || limit == 0 {
        return Vec::new();
    }
    let seed_set: HashSet<&str> = seeds.iter().map(String::as_str).collect();
    let mut scores: BTreeMap<String, i64> = BTreeMap::new();
    for seed in seeds {
        if let Some(list) = index.neighbors.get(seed) {
            for n in list {
                if !seed_set.contains(n.as_str()) {
                    *scores.entry(n.clone()).or_default() += 3;
                }
            }
        }
        // Same-stem test companion (foo.rs ↔ foo_test.rs / test_foo.py / foo.test.ts).
        for file in &index.files {
            if seed_set.contains(file.path.as_str()) {
                continue;
            }
            if is_test_companion(seed, &file.path) {
                *scores.entry(file.path.clone()).or_default() += 4;
            }
        }
    }
    let mut ranked: Vec<_> = scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.truncate(limit);
    ranked.into_iter().map(|(p, _)| p).collect()
}

fn is_test_companion(source: &str, candidate: &str) -> bool {
    let src_stem = file_stem(source);
    let cand_stem = file_stem(candidate);
    if src_stem.is_empty() || cand_stem.is_empty() {
        return false;
    }
    // Same directory preferred.
    if parent_dir(source) != parent_dir(candidate) {
        // Allow tests/ mirroring src/ loosely via stem only.
        if !is_test_path(candidate) && !is_test_path(source) {
            return false;
        }
    }
    let src = src_stem.to_ascii_lowercase();
    let cand = cand_stem.to_ascii_lowercase();
    cand == format!("{src}_test")
        || cand == format!("test_{src}")
        || cand == format!("{src}.test")
        || cand == format!("{src}.spec")
        || cand == format!("{src}_spec")
        || (is_test_path(candidate) && cand.contains(&src))
        || (is_test_path(source) && src.contains(&cand))
}

fn file_stem(path: &str) -> String {
    let name = path.rsplit('/').next().unwrap_or(path);
    match name.rfind('.') {
        Some(idx) if idx > 0 => name[..idx].to_string(),
        _ => name.to_string(),
    }
}

fn lock_cache(cache: &std::sync::Mutex<RepoMapCache>) -> std::sync::MutexGuard<'_, RepoMapCache> {
    cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn fingerprint_workspace(root: &Path) -> Result<u128> {
    let mut count = 0_u128;
    let mut bytes = 0_u128;
    let mut max_mtime = 0_u128;
    for entry in walk_files(root) {
        let path = entry?;
        let meta =
            std::fs::metadata(&path).with_context(|| format!("statting {}", path.display()))?;
        if !meta.is_file() || meta.len() > MAX_FILE_BYTES {
            continue;
        }
        let Ok(relative) = path.strip_prefix(root) else {
            continue;
        };
        let relative = normalize(&relative.to_string_lossy());
        if !indexable_path(&relative) {
            continue;
        }
        count += 1;
        bytes = bytes.saturating_add(meta.len() as u128);
        let mtime = meta
            .modified()
            .ok()
            .and_then(|time| time.duration_since(SystemTime::UNIX_EPOCH).ok())
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        max_mtime = max_mtime.max(mtime);
    }
    // Pack into one key; collisions are acceptable (worst case: extra rebuild).
    Ok(count
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(bytes)
        .wrapping_add(max_mtime ^ max_mtime.rotate_left(17)))
}

fn build_index(root: &Path, fingerprint: u128) -> Result<BuiltIndex> {
    let mut files = Vec::new();
    let mut symbols = Vec::new();
    let mut discovery_errors = 0_u32;
    for entry in walk_files(root) {
        let path = match entry {
            Ok(path) => path,
            Err(_) => {
                discovery_errors = discovery_errors.saturating_add(1);
                continue;
            }
        };
        let meta = match std::fs::metadata(&path) {
            Ok(meta) => meta,
            Err(_) => {
                discovery_errors = discovery_errors.saturating_add(1);
                continue;
            }
        };
        if !meta.is_file() || meta.len() > MAX_FILE_BYTES {
            continue;
        }
        let Ok(relative) = path.strip_prefix(root) else {
            continue;
        };
        let relative = normalize(&relative.to_string_lossy());
        if !indexable_path(&relative) {
            continue;
        }
        let content = match std::fs::read_to_string(&path) {
            Ok(content) => content,
            Err(_) => {
                discovery_errors = discovery_errors.saturating_add(1);
                continue;
            }
        };
        let file_symbols = extract_symbols(&relative, &content);
        let declarations = file_symbols
            .iter()
            .take(12)
            .map(|symbol| format!("{} {}", symbol.kind, symbol.name))
            .collect::<Vec<_>>();
        for symbol in file_symbols {
            symbols.push(symbol);
        }
        let import_hints = extract_import_hints(&relative, &content);
        let is_manifest = is_manifest(&relative);
        let is_entrypoint = is_entrypoint(&relative);
        let mut score_base = 0_i64;
        if is_manifest {
            score_base += 5_000;
        }
        if is_entrypoint {
            score_base += 4_000;
        }
        if is_test_path(&relative) {
            score_base += 500;
        }
        // Prefer shallow paths slightly.
        let depth = relative.bytes().filter(|byte| *byte == b'/').count() as i64;
        score_base += (8 - depth).clamp(0, 8) * 50;
        files.push(FileSummary {
            path: relative,
            score_base,
            declarations,
            is_manifest,
            is_entrypoint,
            import_hints,
        });
    }
    let _ = discovery_errors;
    files.sort_by(|left, right| left.path.cmp(&right.path));
    symbols.sort_by(|left, right| {
        left.name
            .to_ascii_lowercase()
            .cmp(&right.name.to_ascii_lowercase())
            .then_with(|| left.path.cmp(&right.path))
            .then_with(|| left.line.cmp(&right.line))
    });
    let neighbors = build_neighbor_graph(&files);
    Ok(BuiltIndex {
        root: root.to_path_buf(),
        fingerprint,
        files,
        symbols,
        neighbors,
    })
}

/// Same-directory companions + resolved relative import edges.
fn build_neighbor_graph(files: &[FileSummary]) -> BTreeMap<String, Vec<String>> {
    let path_set: HashSet<&str> = files.iter().map(|f| f.path.as_str()).collect();
    let mut by_dir: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for file in files {
        let dir = parent_dir(&file.path);
        by_dir.entry(dir).or_default().push(file.path.clone());
    }
    let mut neighbors: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for file in files {
        let mut set = HashSet::new();
        // Directory siblings (capped) — tests, mod.rs, nearby modules.
        if let Some(siblings) = by_dir.get(&parent_dir(&file.path)) {
            for sib in siblings.iter().take(24) {
                if sib != &file.path {
                    set.insert(sib.clone());
                }
            }
        }
        // Package-root companions: Cargo.toml / package.json / go.mod / pyproject.
        for ancestor in package_root_paths(&file.path) {
            if path_set.contains(ancestor.as_str()) {
                set.insert(ancestor);
            }
        }
        // Relative import hints → concrete files when they resolve.
        for hint in &file.import_hints {
            for candidate in resolve_import_candidates(&file.path, hint) {
                if path_set.contains(candidate.as_str()) {
                    set.insert(candidate);
                }
            }
        }
        if !set.is_empty() {
            let mut list: Vec<String> = set.into_iter().collect();
            list.sort();
            list.truncate(16);
            neighbors.insert(file.path.clone(), list);
        }
    }
    neighbors
}

fn parent_dir(path: &str) -> String {
    match path.rfind('/') {
        Some(idx) => path[..idx].to_string(),
        None => String::new(),
    }
}

fn package_root_paths(path: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut dir = parent_dir(path);
    loop {
        for name in [
            "Cargo.toml",
            "package.json",
            "go.mod",
            "pyproject.toml",
            "setup.py",
        ] {
            if dir.is_empty() {
                out.push(name.to_string());
            } else {
                out.push(format!("{dir}/{name}"));
            }
        }
        if dir.is_empty() {
            break;
        }
        dir = parent_dir(&dir);
    }
    out
}

fn resolve_import_candidates(from_path: &str, hint: &str) -> Vec<String> {
    let hint = hint
        .trim()
        .trim_matches(|c| c == '"' || c == '\'' || c == ';');
    if hint.is_empty() || hint.starts_with("http") {
        return Vec::new();
    }
    // Only relative / crate-local style paths.
    let relative = if hint.starts_with("./") || hint.starts_with("../") {
        let base = parent_dir(from_path);
        normalize_join(&base, hint)
    } else if hint.starts_with("crate::")
        || hint.starts_with("super::")
        || hint.starts_with("self::")
    {
        // Rust path — map last segment to a likely module file under same tree.
        let seg = hint.rsplit("::").next().unwrap_or(hint).trim().to_string();
        if seg.is_empty() {
            return Vec::new();
        }
        let dir = parent_dir(from_path);
        return [
            format!("{dir}/{seg}.rs"),
            format!("{dir}/{seg}/mod.rs"),
            format!("{seg}.rs"),
            format!("{seg}/mod.rs"),
        ]
        .into_iter()
        .map(|p| normalize(&p))
        .filter(|p| !p.is_empty())
        .collect();
    } else if hint.contains('/') || hint.starts_with('.') {
        normalize_join(&parent_dir(from_path), hint)
    } else {
        // Bare module token — try sibling file.
        let dir = parent_dir(from_path);
        let mut cands = vec![
            if dir.is_empty() {
                format!("{hint}.rs")
            } else {
                format!("{dir}/{hint}.rs")
            },
            if dir.is_empty() {
                format!("{hint}.py")
            } else {
                format!("{dir}/{hint}.py")
            },
            if dir.is_empty() {
                format!("{hint}.ts")
            } else {
                format!("{dir}/{hint}.ts")
            },
            if dir.is_empty() {
                format!("{hint}.js")
            } else {
                format!("{dir}/{hint}.js")
            },
            if dir.is_empty() {
                format!("{hint}.go")
            } else {
                format!("{dir}/{hint}.go")
            },
        ];
        if !dir.is_empty() {
            cands.push(format!("{dir}/{hint}/mod.rs"));
            cands.push(format!("{dir}/{hint}/index.ts"));
            cands.push(format!("{dir}/{hint}/index.js"));
            cands.push(format!("{dir}/{hint}/__init__.py"));
        }
        return cands.into_iter().map(|p| normalize(&p)).collect();
    };
    let mut out = Vec::new();
    let base = normalize(&relative);
    if base.is_empty() {
        return out;
    }
    // As-is and with common extensions / index files.
    out.push(base.clone());
    for ext in [".rs", ".py", ".ts", ".tsx", ".js", ".jsx", ".go"] {
        if !base.ends_with(ext) {
            out.push(format!("{base}{ext}"));
        }
    }
    out.push(format!("{base}/mod.rs"));
    out.push(format!("{base}/index.ts"));
    out.push(format!("{base}/index.js"));
    out.push(format!("{base}/__init__.py"));
    out
}

fn normalize_join(base: &str, rel: &str) -> String {
    let mut parts: Vec<String> = if base.is_empty() {
        Vec::new()
    } else {
        base.split('/')
            .filter(|p| !p.is_empty())
            .map(|p| p.to_string())
            .collect()
    };
    let rel_norm = rel.replace('\\', "/");
    for part in rel_norm.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other.to_string()),
        }
    }
    parts.join("/")
}

/// Cheap import/mod tokens from source (not a full parser).
fn extract_import_hints(path: &str, content: &str) -> Vec<String> {
    let mut hints = Vec::new();
    let lower_path = path.to_ascii_lowercase();
    for (idx, line) in content.lines().enumerate() {
        if idx > 120 {
            // Imports are almost always at the top.
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with('#') {
            continue;
        }
        if lower_path.ends_with(".rs") {
            if let Some(rest) = trimmed
                .strip_prefix("mod ")
                .or_else(|| trimmed.strip_prefix("pub mod "))
            {
                let name = rest
                    .trim()
                    .trim_end_matches(';')
                    .trim_end_matches('{')
                    .trim();
                if !name.is_empty() && !name.contains(' ') {
                    hints.push(name.to_string());
                }
            }
            if let Some(rest) = trimmed
                .strip_prefix("use ")
                .or_else(|| trimmed.strip_prefix("pub use "))
            {
                let pathish = rest.split_whitespace().next().unwrap_or("");
                let pathish = pathish.trim_end_matches(';').trim_end_matches('{');
                if pathish.starts_with("crate::")
                    || pathish.starts_with("super::")
                    || pathish.starts_with("self::")
                {
                    hints.push(pathish.to_string());
                }
            }
        } else if lower_path.ends_with(".py") {
            if let Some(rest) = trimmed.strip_prefix("from ") {
                let module = rest.split_whitespace().next().unwrap_or("");
                if module.starts_with('.') {
                    hints.push(module.to_string());
                }
            } else if let Some(rest) = trimmed.strip_prefix("import ") {
                let module = rest
                    .split_whitespace()
                    .next()
                    .unwrap_or("")
                    .trim_end_matches(',');
                if module.starts_with('.') {
                    hints.push(module.to_string());
                }
            }
        } else if lower_path.ends_with(".ts")
            || lower_path.ends_with(".tsx")
            || lower_path.ends_with(".js")
            || lower_path.ends_with(".jsx")
        {
            // import … from '…'  / require('…')
            if let Some(idx) = trimmed.find("from ") {
                let after = &trimmed[idx + 5..];
                if let Some(q) = after.find(['\'', '"']) {
                    let quote = after.as_bytes()[q] as char;
                    if let Some(end) = after[q + 1..].find(quote) {
                        let spec = &after[q + 1..q + 1 + end];
                        if spec.starts_with('.') {
                            hints.push(spec.to_string());
                        }
                    }
                }
            } else if let Some(idx) = trimmed.find("require(") {
                let after = &trimmed[idx + 8..];
                if let Some(q) = after.find(['\'', '"']) {
                    let quote = after.as_bytes()[q] as char;
                    if let Some(end) = after[q + 1..].find(quote) {
                        let spec = &after[q + 1..q + 1 + end];
                        if spec.starts_with('.') {
                            hints.push(spec.to_string());
                        }
                    }
                }
            }
        } else if lower_path.ends_with(".go") {
            if trimmed.starts_with("import ") {
                if let Some(q) = trimmed.find('"') {
                    if let Some(end) = trimmed[q + 1..].find('"') {
                        let spec = &trimmed[q + 1..q + 1 + end];
                        // Local replace-style paths sometimes relative.
                        if spec.starts_with("./") || spec.starts_with("../") {
                            hints.push(spec.to_string());
                        }
                    }
                }
            }
        }
        if hints.len() >= 24 {
            break;
        }
    }
    hints.sort();
    hints.dedup();
    hints
}

fn walk_files(root: &Path) -> impl Iterator<Item = Result<PathBuf>> {
    ignore::WalkBuilder::new(root)
        .hidden(false)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .ignore(true)
        .parents(true)
        .filter_entry(|entry| {
            let name = entry.file_name().to_str().unwrap_or("");
            !ignored_directory(name)
        })
        .build()
        .filter_map(|result| match result {
            Ok(entry) if entry.file_type().is_some_and(|kind| kind.is_file()) => {
                Some(Ok(entry.into_path()))
            }
            Ok(_) => None,
            Err(error) => Some(Err(anyhow::anyhow!(error))),
        })
}

fn render_repo_map(index: &BuiltIndex, task: &str, scope: Option<&str>, limit: usize) -> String {
    let words = task_words(task);
    let mut ranked = index
        .files
        .iter()
        .filter(|file| in_scope(&file.path, scope))
        .map(|file| {
            let mut score = file.score_base;
            let path_lower = file.path.to_ascii_lowercase();
            for word in &words {
                if path_lower.contains(word) {
                    score += 2_000;
                }
                for declaration in &file.declarations {
                    if declaration.to_ascii_lowercase().contains(word) {
                        score += 1_200;
                    }
                }
            }
            (score, file)
        })
        .filter(|(score, file)| {
            // Without a task, keep manifests/entrypoints and shallow source.
            !words.is_empty() || *score > 0 || file.is_manifest || file.is_entrypoint
        })
        .collect::<Vec<_>>();
    if ranked.is_empty() {
        // Task filters were too tight — fall back to base ranking.
        ranked = index
            .files
            .iter()
            .filter(|file| in_scope(&file.path, scope))
            .map(|file| (file.score_base, file))
            .collect();
    }
    ranked.sort_by(|left, right| {
        right
            .0
            .cmp(&left.0)
            .then_with(|| left.1.path.cmp(&right.1.path))
    });
    ranked.truncate(limit);

    if ranked.is_empty() {
        return match scope {
            Some(path) => format!("No indexable source files under `{path}`."),
            None => "No indexable source files found in the workspace.".into(),
        };
    }

    let mut out = String::from("# Repo map\n");
    if !task.is_empty() {
        out.push_str(&format!("task filter: {task}\n"));
    }
    if let Some(scope) = scope {
        out.push_str(&format!("scope: {scope}\n"));
    }
    out.push_str(&format!(
        "showing {} of {} indexed files · {} symbols\n",
        ranked.len(),
        index.files.len(),
        index.symbols.len()
    ));
    for (_, file) in ranked {
        out.push_str(&file.path);
        out.push('\n');
        for declaration in file.declarations.iter().take(8) {
            out.push_str("  ");
            out.push_str(declaration);
            out.push('\n');
        }
        if out.len() >= MAX_MAP_RENDER_CHARS {
            out.push_str("… (repo map truncated)\n");
            break;
        }
    }
    if out.len() > MAX_MAP_RENDER_CHARS {
        out.truncate(floor_char_boundary(&out, MAX_MAP_RENDER_CHARS));
        out.push_str("\n… (repo map truncated)");
    }
    out
}

#[derive(Clone, Debug)]
struct Hit {
    path: String,
    line: u32,
    name: String,
    kind: &'static str,
    score: i64,
}

fn collect_symbol_hits(
    index: &BuiltIndex,
    words: &HashSet<String>,
    scope: Option<&str>,
    limit: usize,
) -> Vec<Hit> {
    if words.is_empty() {
        return Vec::new();
    }
    let mut hits = Vec::new();
    for symbol in &index.symbols {
        if !in_scope(&symbol.path, scope) {
            continue;
        }
        let name_lower = symbol.name.to_ascii_lowercase();
        let mut score = 0_i64;
        for word in words {
            if name_lower == *word {
                score += 10_000;
            } else if name_lower.starts_with(word) {
                score += 6_000;
            } else if name_lower.contains(word) {
                score += 3_000;
            }
        }
        if score == 0 {
            continue;
        }
        // Prefer non-test definitions slightly when scores tie elsewhere.
        if is_test_path(&symbol.path) {
            score -= 200;
        }
        hits.push(Hit {
            path: symbol.path.clone(),
            line: symbol.line,
            name: symbol.name.clone(),
            kind: symbol.kind,
            score,
        });
    }
    hits.sort_by(|left, right| {
        right
            .score
            .cmp(&left.score)
            .then_with(|| left.path.cmp(&right.path))
            .then_with(|| left.line.cmp(&right.line))
    });
    hits.truncate(limit.max(1));
    hits
}

fn render_symbol_hits(
    index: &BuiltIndex,
    query: &str,
    scope: Option<&str>,
    limit: usize,
) -> String {
    let mut words = HashSet::new();
    let normalized = query.trim().to_ascii_lowercase();
    if !normalized.is_empty() {
        words.insert(normalized);
    }
    // Also split CamelCase / snake_case style queries into tokens.
    for word in task_words(query) {
        words.insert(word);
    }
    let hits = collect_symbol_hits(index, &words, scope, limit);
    if hits.is_empty() {
        return format!("No symbols matching `{query}`.");
    }
    let mut out = format!("# Symbol search `{query}` · {} hit(s)\n", hits.len());
    let mut by_path: BTreeMap<String, Vec<Hit>> = BTreeMap::new();
    for hit in hits {
        by_path.entry(hit.path.clone()).or_default().push(hit);
    }
    for (path, path_hits) in by_path {
        out.push_str(&path);
        out.push('\n');
        for hit in path_hits {
            out.push_str(&format!("  {}:{} {}\n", hit.line, hit.kind, hit.name));
        }
        if out.len() >= MAX_MAP_RENDER_CHARS {
            out.push_str("… (truncated)\n");
            break;
        }
    }
    out
}

fn extract_symbols(path: &str, content: &str) -> Vec<Symbol> {
    let mut out = Vec::new();
    for (idx, raw) in content.lines().enumerate() {
        if out.len() >= MAX_SYMBOLS_PER_FILE {
            break;
        }
        let line_no = (idx + 1) as u32;
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with('#') {
            continue;
        }
        let stripped = strip_modifiers(trimmed);
        if let Some((kind, name)) = match_declaration(stripped) {
            out.push(Symbol {
                path: path.to_string(),
                line: line_no,
                name: name.to_string(),
                kind,
            });
        }
    }
    out
}

fn match_declaration(line: &str) -> Option<(&'static str, &str)> {
    const RULES: &[(&str, &str)] = &[
        ("fn ", "fn"),
        ("struct ", "struct"),
        ("enum ", "enum"),
        ("trait ", "trait"),
        ("impl ", "impl"),
        ("class ", "class"),
        ("def ", "def"),
        ("func ", "func"),
        ("interface ", "interface"),
        ("type ", "type"),
        ("function ", "function"),
        ("export const ", "const"),
        ("const ", "const"),
        ("let ", "let"),
        ("var ", "var"),
        ("mod ", "mod"),
        ("module ", "module"),
    ];
    for (prefix, kind) in RULES {
        if let Some(rest) = line.strip_prefix(prefix) {
            let name = first_ident(rest)?;
            if *kind == "impl" {
                // `impl Foo` / `impl Trait for Foo` — keep the last type-ish token.
                let cleaned = rest
                    .split('{')
                    .next()
                    .unwrap_or(rest)
                    .split(" for ")
                    .last()
                    .unwrap_or(rest)
                    .trim();
                let name = first_ident(cleaned).unwrap_or(name);
                return Some((*kind, name));
            }
            if *kind == "const" || *kind == "let" || *kind == "var" {
                // Only keep exported / top-level-looking bindings with Upper or snake names
                // that look like API surface (skip trivial locals if indented — already trimmed).
                if name.len() < 2 {
                    return None;
                }
            }
            return Some((*kind, name));
        }
    }
    None
}

fn first_ident(input: &str) -> Option<&str> {
    let start = input
        .char_indices()
        .find(|(_, ch)| ch.is_ascii_alphabetic() || *ch == '_')
        .map(|(idx, _)| idx)?;
    let rest = &input[start..];
    let end = rest
        .char_indices()
        .find(|(_, ch)| !(ch.is_ascii_alphanumeric() || *ch == '_'))
        .map(|(idx, _)| idx)
        .unwrap_or(rest.len());
    let ident = &rest[..end];
    if ident.is_empty() { None } else { Some(ident) }
}

fn strip_modifiers(mut line: &str) -> &str {
    loop {
        let before = line;
        for prefix in [
            "pub(crate) ",
            "pub(super) ",
            "pub ",
            "async ",
            "unsafe ",
            "export default ",
            "export ",
            "default ",
            "public ",
            "private ",
            "protected ",
            "static ",
            "abstract ",
            "final ",
            "readonly ",
            "declare ",
        ] {
            if let Some(rest) = line.strip_prefix(prefix) {
                line = rest.trim_start();
                break;
            }
        }
        // Rust visibility with paths already handled; strip attributes lightly.
        if let Some(rest) = line.strip_prefix("#[") {
            if let Some(idx) = rest.find(']') {
                line = rest[idx + 1..].trim_start();
                continue;
            }
        }
        if line == before {
            return line;
        }
    }
}

fn task_words(task: &str) -> HashSet<String> {
    const STOP: &[&str] = &[
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "into",
        "about",
        "have",
        "has",
        "was",
        "are",
        "were",
        "been",
        "being",
        "your",
        "you",
        "our",
        "out",
        "use",
        "using",
        "please",
        "just",
        "like",
        "make",
        "need",
        "want",
        "should",
        "could",
        "would",
        "can",
        "will",
        "all",
        "any",
        "add",
        "fix",
        "update",
        "change",
        "create",
        "implement",
        "read",
        "write",
        "file",
        "code",
        "function",
        "class",
        "module",
        "project",
        "repo",
    ];
    let mut words = HashSet::new();
    for raw in task.split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_') {
        if raw.len() < 3 {
            continue;
        }
        let lower = raw.to_ascii_lowercase();
        if STOP.contains(&lower.as_str()) {
            continue;
        }
        words.insert(lower);
        // Split snake_case / camelCase pieces.
        for piece in split_ident_pieces(raw) {
            if piece.len() >= 3 {
                let piece_lower = piece.to_ascii_lowercase();
                if !STOP.contains(&piece_lower.as_str()) {
                    words.insert(piece_lower);
                }
            }
        }
    }
    words
}

fn split_ident_pieces(input: &str) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = input.chars().collect();
    for (idx, ch) in chars.iter().enumerate() {
        if *ch == '_' || *ch == '-' {
            if !current.is_empty() {
                pieces.push(std::mem::take(&mut current));
            }
            continue;
        }
        if ch.is_ascii_uppercase()
            && !current.is_empty()
            && chars
                .get(idx.wrapping_sub(1))
                .is_some_and(|prev| prev.is_ascii_lowercase())
        {
            pieces.push(std::mem::take(&mut current));
        }
        current.push(*ch);
    }
    if !current.is_empty() {
        pieces.push(current);
    }
    pieces
}

fn in_scope(path: &str, scope: Option<&str>) -> bool {
    match scope {
        None => true,
        Some(scope) if scope.is_empty() => true,
        Some(scope) => path == scope || path.starts_with(&format!("{scope}/")),
    }
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
            "main.rs"
                | "lib.rs"
                | "mod.rs"
                | "main.py"
                | "__init__.py"
                | "main.go"
                | "index.ts"
                | "index.js"
                | "app.ts"
                | "app.js"
        )
    )
}

fn is_test_path(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    lower.contains("/test")
        || lower.starts_with("test")
        || lower.contains("_test.")
        || lower.contains("/tests/")
}

fn ignored_directory(name: &str) -> bool {
    matches!(
        name,
        "target"
            | "node_modules"
            | ".git"
            | ".hg"
            | ".svn"
            | ".hi"
            | "dist"
            | "build"
            | "vendor"
            | ".venv"
            | "venv"
            | "__pycache__"
            | ".mypy_cache"
            | ".pytest_cache"
            | "coverage"
            | ".idea"
            | ".vscode"
    )
}

fn normalize(path: &str) -> String {
    path.replace('\\', "/")
        .trim_start_matches("./")
        .trim_matches('/')
        .to_string()
}

fn floor_char_boundary(text: &str, max: usize) -> usize {
    if max >= text.len() {
        return text.len();
    }
    let mut idx = max;
    while idx > 0 && !text.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn temp_repo(label: &str) -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "hi-repo-map-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(
            root.join("src/lib.rs"),
            "pub struct WorkspaceRuntime;\n\
             pub fn build_task_context_index() {}\n\
             impl WorkspaceRuntime {\n\
                 pub fn reconcile(&self) {}\n\
             }\n",
        )
        .unwrap();
        std::fs::write(
            root.join("src/auth.py"),
            "class AuthService:\n    pass\n\ndef verify_password(pw):\n    return True\n",
        )
        .unwrap();
        std::fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"demo\"\nversion = \"0.1.0\"\n",
        )
        .unwrap();
        root
    }

    #[tokio::test]
    async fn find_symbol_locates_rust_and_python() {
        let root = temp_repo("find");
        let cache = Mutex::new(RepoMapCache::new());
        let out = run_find_symbol(&root, &cache, r#"{"query":"WorkspaceRuntime"}"#)
            .await
            .unwrap();
        assert!(out.content.contains("WorkspaceRuntime"), "{}", out.content);
        assert!(out.content.contains("src/lib.rs"), "{}", out.content);

        let out = run_find_symbol(&root, &cache, r#"{"query":"verify_password"}"#)
            .await
            .unwrap();
        assert!(out.content.contains("verify_password"), "{}", out.content);
        assert!(out.content.contains("src/auth.py"), "{}", out.content);
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn repo_map_ranks_task_words() {
        let root = temp_repo("map");
        let cache = Mutex::new(RepoMapCache::new());
        let out = run_repo_map(
            &root,
            &cache,
            r#"{"task":"fix WorkspaceRuntime reconcile","limit":10}"#,
        )
        .await
        .unwrap();
        assert!(out.content.contains("src/lib.rs"), "{}", out.content);
        assert!(
            out.content.contains("WorkspaceRuntime") || out.content.contains("struct"),
            "{}",
            out.content
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn orientation_seed_mentions_symbol_hits() {
        let root = temp_repo("orient");
        let cache = Mutex::new(RepoMapCache::new());
        let seed = orientation_for_task(
            &root,
            "investigate WorkspaceRuntime and verify_password",
            &cache,
        )
        .expect("seed");
        assert!(seed.contains("WorkspaceRuntime"), "{seed}");
        assert!(seed.contains("verify_password"), "{seed}");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn orientation_includes_companions_and_imports() {
        let root = temp_repo("companions");
        // Sibling module + test + relative import neighborhood.
        std::fs::write(
            root.join("src/billing.rs"),
            "mod invoice;\npub fn charge() {}\nuse crate::invoice::LineItem;\n",
        )
        .unwrap();
        std::fs::write(
            root.join("src/invoice.rs"),
            "pub struct LineItem;\npub fn total() -> i32 { 0 }\n",
        )
        .unwrap();
        std::fs::write(
            root.join("src/billing_test.rs"),
            "#[test]\nfn charge_works() {}\n",
        )
        .unwrap();
        let cache = Mutex::new(RepoMapCache::new());
        let seed = orientation_for_task(&root, "fix charge in billing", &cache).expect("seed");
        assert!(
            seed.contains("charge") || seed.contains("billing"),
            "{seed}"
        );
        // Companions section should surface neighbors of the hit file.
        assert!(
            seed.contains("invoice.rs")
                || seed.contains("billing_test.rs")
                || seed.contains("Cargo.toml")
                || seed.contains("Nearby"),
            "expected companion neighborhood in seed: {seed}"
        );
        let paths = ranked_paths_for_task(&root, "fix charge in billing", &cache, 12);
        assert!(
            paths.iter().any(|p| p.contains("billing")),
            "billing should rank: {paths:?}"
        );
        assert!(
            paths.iter().any(|p| p.contains("invoice")
                || p.contains("billing_test")
                || p.ends_with("Cargo.toml")),
            "companions should rank alongside hit: {paths:?}"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn extract_import_hints_from_polyglot_sources() {
        let rs = extract_import_hints(
            "src/lib.rs",
            "mod foo;\nuse crate::bar::Baz;\nuse std::io;\n",
        );
        assert!(rs.iter().any(|h| h == "foo"), "{rs:?}");
        assert!(rs.iter().any(|h| h.starts_with("crate::")), "{rs:?}");
        assert!(!rs.iter().any(|h| h.starts_with("std::")), "{rs:?}");

        let ts = extract_import_hints(
            "web/app.ts",
            "import { x } from './util';\nconst y = require(\"../lib/helper\");\n",
        );
        assert!(ts.iter().any(|h| h == "./util"), "{ts:?}");
        assert!(ts.iter().any(|h| h == "../lib/helper"), "{ts:?}");

        let py = extract_import_hints("pkg/a.py", "from .b import c\nimport os\n");
        assert!(py.iter().any(|h| h == ".b"), "{py:?}");
    }

    #[test]
    fn cache_reuses_fingerprint_until_edit() {
        let root = temp_repo("cache");
        let cache = Mutex::new(RepoMapCache::new());
        {
            let mut guard = cache.lock().unwrap();
            let first = guard.get_or_build(&root).unwrap().fingerprint;
            let second = guard.get_or_build(&root).unwrap().fingerprint;
            assert_eq!(first, second);
        }
        // Ensure mtime advances on coarse filesystems.
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(root.join("src/new.rs"), "pub fn brand_new() {}").unwrap();
        let paths = ranked_paths_for_task(&root, "brand_new helper", &cache, 8);
        assert!(
            paths.iter().any(|path| path.contains("new.rs")),
            "{paths:?}"
        );
        let _ = std::fs::remove_dir_all(root);
    }
}
