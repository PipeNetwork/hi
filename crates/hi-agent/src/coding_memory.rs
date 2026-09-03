//! Deterministic coding-fact extraction after green verified turns.
//!
//! No extra model call: facts are derived from the turn's ledger, verification
//! executions, and task contract, then recorded into the session decision log
//! (compaction-safe system prompt) and optionally merged into project
//! `.hi/memory.md` so the next session starts oriented.

use std::collections::BTreeSet;
use std::path::Path;

use crate::decision::Decision;
use crate::memory::{self, MEMORY_MAX_CHARS};
use crate::verify::VerificationExecution;

/// Inputs from one green, file-changing turn.
pub(crate) struct CodingFactInput<'a> {
    pub changed_files: &'a [String],
    pub verify_executions: &'a [VerificationExecution],
    pub wants_tests: bool,
    pub workspace_root: &'a Path,
}

/// Extract durable coding facts from a successful turn. Pure — no I/O except
/// existence checks for package markers under `workspace_root`.
pub(crate) fn extract_coding_facts(input: &CodingFactInput<'_>) -> Vec<Decision> {
    let mut out = Vec::new();

    if let Some(d) = verify_command_fact(input.verify_executions) {
        out.push(d);
    }
    if let Some(d) = package_ownership_fact(input.workspace_root, input.changed_files) {
        out.push(d);
    }
    if input.wants_tests
        && let Some(d) = test_gate_fact(input.verify_executions, input.changed_files)
    {
        out.push(d);
    }
    if let Some(d) = stack_fact(input.workspace_root, input.changed_files) {
        out.push(d);
    }

    out
}

fn verify_command_fact(executions: &[VerificationExecution]) -> Option<Decision> {
    // Prefer the last succeeded shell stage (most expensive / most definitive).
    let stage = executions.iter().rev().find(|e| {
        e.status == hi_tools::ToolStatus::Succeeded
            && e.name != "lsp"
            && !e.command.is_empty()
            && e.command != "diagnostics"
    })?;
    Some(Decision {
        summary: format!("verify: {}", stage.name),
        rationale: format!(
            "This turn passed deterministic verification with `{}`. Prefer the same \
             command (or package-local equivalent) before claiming done on related work.",
            stage.command
        ),
        files: Vec::new(),
    })
}

fn package_ownership_fact(root: &Path, changed: &[String]) -> Option<Decision> {
    let packages = hi_tools::affected_any_package_dirs(root, changed);
    if packages.is_empty() {
        // Fall back to top-level dirs of changed paths.
        let mut tops = BTreeSet::new();
        for path in changed {
            let top = path
                .replace('\\', "/")
                .split('/')
                .next()
                .unwrap_or(path)
                .to_string();
            if !top.is_empty() && top != "." && !top.starts_with('.') {
                tops.insert(top);
            }
        }
        if tops.is_empty() {
            return None;
        }
        let list = tops.into_iter().take(6).collect::<Vec<_>>();
        return Some(Decision {
            summary: format!("touched: {}", list.join(", ")),
            rationale: format!(
                "Successful edits landed under {}. Keep related changes in these \
                 areas unless the task explicitly expands scope.",
                list.join(", ")
            ),
            files: changed.iter().take(8).cloned().collect(),
        });
    }
    let list = packages.into_iter().take(6).collect::<Vec<_>>();
    Some(Decision {
        summary: format!("packages: {}", list.join(", ")),
        rationale: format!(
            "Verified changes belonged to package(s) {}. Prefer package-local \
             check/test (`cargo check --manifest-path`, pytest -q, npm test, go test) \
             for follow-ups in the same packages.",
            list.join(", ")
        ),
        files: changed.iter().take(8).cloned().collect(),
    })
}

fn test_gate_fact(executions: &[VerificationExecution], changed: &[String]) -> Option<Decision> {
    let test_stage = executions.iter().rev().find(|e| {
        e.status == hi_tools::ToolStatus::Succeeded
            && (e.name.contains("test")
                || e.command.contains("test")
                || e.command.contains("pytest"))
    })?;
    Some(Decision {
        summary: format!("tests: {}", test_stage.name),
        rationale: format!(
            "Task was test-gated and `{}` passed. Keep tests green when touching \
             the same surface; run package-local tests before finishing.",
            test_stage.command
        ),
        files: changed.iter().take(6).cloned().collect(),
    })
}

fn stack_fact(root: &Path, changed: &[String]) -> Option<Decision> {
    let mut stacks = BTreeSet::new();
    for path in changed {
        let lower = path.to_ascii_lowercase();
        if lower.ends_with(".rs") || lower.contains("cargo.toml") {
            stacks.insert("Rust/Cargo");
        } else if lower.ends_with(".py") || lower.contains("pyproject.toml") {
            stacks.insert("Python");
        } else if lower.ends_with(".go") || lower.ends_with("go.mod") {
            stacks.insert("Go");
        } else if lower.ends_with(".ts")
            || lower.ends_with(".tsx")
            || lower.ends_with(".js")
            || lower.contains("package.json")
        {
            stacks.insert("JS/TS");
        }
    }
    // Confirm against workspace markers for a stronger claim.
    if root.join("Cargo.toml").is_file() {
        stacks.insert("Rust/Cargo");
    }
    if stacks.is_empty() {
        return None;
    }
    let list = stacks.into_iter().collect::<Vec<_>>();
    Some(Decision {
        summary: format!("stack: {}", list.join(" + ")),
        rationale: format!(
            "This workspace area is {}. Prefer stack-native tools \
             (repo_map/find_symbol, package-local verify) over generic full-tree builds.",
            list.join(" + ")
        ),
        files: Vec::new(),
    })
}

/// Merge fact bullets into project memory.md without a model call.
/// Dedupes by case-insensitive summary prefix; caps total body size.
pub(crate) fn merge_facts_into_memory(path: &Path, facts: &[Decision]) -> Result<usize, String> {
    if facts.is_empty() {
        return Ok(0);
    }
    let body = memory::read_layer(path);
    let mut bullets: Vec<(Option<u32>, String)> = body
        .lines()
        .filter_map(memory::parse_bullet_line)
        .filter(|(_, text)| !text.is_empty())
        .collect();

    let mut added = 0usize;
    for fact in facts {
        let line = format!("{} — {}", fact.summary, one_line(&fact.rationale));
        let key = fact.summary.to_ascii_lowercase();
        if let Some((_, slot)) = bullets
            .iter_mut()
            .find(|(_, text)| text.to_ascii_lowercase().starts_with(&key))
        {
            *slot = line;
            continue;
        }
        bullets.push((None, line));
        added += 1;
    }

    // Soft cap: keep newest-ish facts by dropping from the front when over char budget.
    let render = |bullets: &[(Option<u32>, String)]| {
        bullets
            .iter()
            .map(|(id, text)| match id {
                Some(id) => format!("- [#{id}] {text}"),
                None => format!("- {text}"),
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    let mut rendered = render(&bullets);
    while rendered.chars().count() > MEMORY_MAX_CHARS && !bullets.is_empty() {
        bullets.remove(0);
        rendered = render(&bullets);
    }

    let mut next = memory::max_bullet_id(&rendered).saturating_add(1);
    let rendered = memory::ensure_bullet_ids(&rendered, &mut next);
    memory::write_memory(path, &rendered)?;
    Ok(added)
}

fn one_line(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(180)
        .collect()
}

// memory::strip_header and MEMORY_MAX_CHARS need to be accessible.
// MEMORY_MAX_CHARS is private — use a local cap or pub(crate) it.

#[cfg(test)]
mod tests {
    use super::*;
    use hi_tools::ToolStatus;

    fn exec(name: &str, command: &str, ok: bool) -> VerificationExecution {
        VerificationExecution {
            round: 1,
            name: name.into(),
            command: command.into(),
            status: if ok {
                ToolStatus::Succeeded
            } else {
                ToolStatus::Failed
            },
            process: None,
            truncation: None,
        }
    }

    #[test]
    fn extracts_verify_and_package_facts() {
        let dir = std::env::temp_dir().join(format!(
            "hi-coding-mem-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("crates/demo/src")).unwrap();
        std::fs::write(
            dir.join("Cargo.toml"),
            "[workspace]\nmembers=[\"crates/demo\"]\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("crates/demo/Cargo.toml"),
            "[package]\nname=\"demo\"\nversion=\"0.1.0\"\n",
        )
        .unwrap();
        std::fs::write(dir.join("crates/demo/src/lib.rs"), "pub fn x() {}\n").unwrap();

        let executions = vec![
            exec("lsp", "diagnostics", true),
            exec(
                "affected-check:crates/demo",
                "cargo check --quiet --manifest-path 'crates/demo/Cargo.toml'",
                true,
            ),
            exec(
                "affected-test:crates/demo",
                "cargo test --quiet --manifest-path 'crates/demo/Cargo.toml'",
                true,
            ),
        ];
        let changed = vec!["crates/demo/src/lib.rs".into()];
        let facts = extract_coding_facts(&CodingFactInput {
            changed_files: &changed,
            verify_executions: &executions,
            wants_tests: true,
            workspace_root: &dir,
        });
        assert!(
            facts.iter().any(|d| d.summary.starts_with("verify:")),
            "{facts:?}"
        );
        assert!(
            facts.iter().any(|d| d.summary.contains("packages:")),
            "{facts:?}"
        );
        assert!(
            facts.iter().any(|d| d.summary.starts_with("tests:")),
            "{facts:?}"
        );
        assert!(
            facts.iter().any(|d| d.summary.contains("Rust")),
            "{facts:?}"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn merge_into_memory_dedupes_by_summary() {
        let dir = std::env::temp_dir().join(format!(
            "hi-coding-mem-merge-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("memory.md");
        let facts = vec![Decision {
            summary: "verify: check".into(),
            rationale: "passed cargo check".into(),
            files: vec![],
        }];
        let n = merge_facts_into_memory(&path, &facts).unwrap();
        assert_eq!(n, 1);
        let n2 = merge_facts_into_memory(&path, &facts).unwrap();
        assert_eq!(n2, 0, "second merge should refresh not double");
        let body = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            body.matches("verify: check").count(),
            1,
            "deduped body: {body}"
        );
        assert!(body.contains("[#"), "coding facts get stable ids: {body}");
        let _ = std::fs::remove_dir_all(dir);
    }
}
