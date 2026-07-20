//! Mid-turn fast feedback after mutations: package targeting helpers and
//! affected-package checks/tests (Rust, Python, JS/TS, Go). LSP diagnostics
//! stay in the agent (needs `LspManager`); this module owns the shell-side runs.

use std::collections::BTreeSet;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::process::ProcessRunner;
use crate::condense::truncate;

const CARGO_CHECK_TIMEOUT_SECS: u64 = 180;
const PACKAGE_TEST_TIMEOUT_SECS: u64 = 300;
const MAX_FEEDBACK_CHARS: usize = 4_000;

/// Relative package directories (from workspace root) that own any of
/// `changed_files` and contain a `[package]` Cargo.toml. The workspace root
/// itself is never returned — callers should fall back to a root `cargo check`
/// when this set is empty but Rust sources still changed.
pub fn affected_cargo_package_dirs(root: &Path, changed_files: &[String]) -> BTreeSet<String> {
    affected_package_dirs(root, changed_files, |directory| {
        let manifest = directory.join("Cargo.toml");
        manifest.is_file()
            && std::fs::read_to_string(manifest)
                .ok()
                .is_some_and(|text| text.lines().any(|line| line.trim() == "[package]"))
    })
}

/// Generic nearest-package walk used by cargo (and available for other
/// ecosystems). Skips the workspace root so root pipelines stay singular.
pub fn affected_package_dirs(
    root: &Path,
    changed_files: &[String],
    is_package_root: impl Fn(&Path) -> bool,
) -> BTreeSet<String> {
    let mut packages = BTreeSet::new();
    for changed in changed_files {
        let relative = Path::new(changed);
        if relative.is_absolute()
            || relative
                .components()
                .any(|component| !matches!(component, std::path::Component::Normal(_)))
        {
            continue;
        }
        let mut directory = root.join(relative);
        if !directory.is_dir() {
            directory.pop();
        }
        while directory.starts_with(root) && directory != root {
            if is_package_root(&directory) {
                if let Ok(relative_package) = directory.strip_prefix(root) {
                    packages.insert(relative_package.to_string_lossy().replace('\\', "/"));
                }
                break;
            }
            if !directory.pop() {
                break;
            }
        }
    }
    packages
}

/// Paths that look like Rust sources (for LSP + cargo fast feedback).
pub fn rust_source_paths(paths: impl IntoIterator<Item = impl AsRef<str>>) -> Vec<String> {
    filter_source_paths(paths, &["rs"])
}

/// Python sources.
pub fn python_source_paths(paths: impl IntoIterator<Item = impl AsRef<str>>) -> Vec<String> {
    filter_source_paths(paths, &["py", "pyi"])
}

/// JavaScript / TypeScript sources.
pub fn javascript_source_paths(paths: impl IntoIterator<Item = impl AsRef<str>>) -> Vec<String> {
    filter_source_paths(paths, &["js", "jsx", "ts", "tsx", "mjs", "cjs"])
}

/// Go sources.
pub fn go_source_paths(paths: impl IntoIterator<Item = impl AsRef<str>>) -> Vec<String> {
    filter_source_paths(paths, &["go"])
}

/// Sources that language servers commonly cover (for mid-turn diagnostics).
pub fn lsp_source_paths(paths: impl IntoIterator<Item = impl AsRef<str>>) -> Vec<String> {
    filter_source_paths(
        paths,
        &["rs", "py", "pyi", "go", "js", "jsx", "ts", "tsx"],
    )
}

fn filter_source_paths(
    paths: impl IntoIterator<Item = impl AsRef<str>>,
    exts: &[&str],
) -> Vec<String> {
    let mut out = BTreeSet::new();
    for path in paths {
        let path = path.as_ref().replace('\\', "/");
        if let Some(ext) = Path::new(&path).extension().and_then(|e| e.to_str()) {
            if exts.iter().any(|want| ext.eq_ignore_ascii_case(want)) {
                out.insert(path);
            }
        }
    }
    out.into_iter().collect()
}

/// Nested dirs with a Python package marker (pyproject/setup/pytest.ini/…).
pub fn affected_python_package_dirs(root: &Path, changed_files: &[String]) -> BTreeSet<String> {
    affected_package_dirs(root, changed_files, is_python_package_root)
}

/// Nested dirs with `package.json`.
pub fn affected_javascript_package_dirs(root: &Path, changed_files: &[String]) -> BTreeSet<String> {
    affected_package_dirs(root, changed_files, |directory| {
        directory.join("package.json").is_file()
    })
}

/// Nested dirs with `go.mod`.
pub fn affected_go_package_dirs(root: &Path, changed_files: &[String]) -> BTreeSet<String> {
    affected_package_dirs(root, changed_files, |directory| {
        directory.join("go.mod").is_file()
    })
}

pub fn is_python_package_root(directory: &Path) -> bool {
    [
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "pytest.ini",
        "tox.ini",
    ]
    .iter()
    .any(|marker| directory.join(marker).is_file())
}

/// Union of all nested package labels touched by `changed_files` (any language).
pub fn affected_any_package_dirs(root: &Path, changed_files: &[String]) -> BTreeSet<String> {
    let mut out = affected_cargo_package_dirs(root, changed_files);
    out.extend(affected_python_package_dirs(root, changed_files));
    out.extend(affected_javascript_package_dirs(root, changed_files));
    out.extend(affected_go_package_dirs(root, changed_files));
    out
}

/// Run `cargo check` for each affected package (or a single root check when the
/// change set only hits the workspace root package).
pub async fn run_affected_cargo_checks(
    root: &Path,
    changed_files: &[String],
    already_checked: &mut BTreeSet<String>,
) -> CargoCommandOutcome {
    run_affected_cargo_command(
        root,
        changed_files,
        already_checked,
        CargoSubcommand::Check,
    )
    .await
}

/// Run `cargo test --quiet` for each affected package after a clean check.
/// Uses a separate dedupe set from checks so a green check does not skip tests.
pub async fn run_affected_cargo_tests(
    root: &Path,
    changed_files: &[String],
    already_tested: &mut BTreeSet<String>,
) -> CargoCommandOutcome {
    run_affected_cargo_command(
        root,
        changed_files,
        already_tested,
        CargoSubcommand::Test,
    )
    .await
}

#[derive(Clone, Copy, Debug)]
enum CargoSubcommand {
    Check,
    Test,
}

impl CargoSubcommand {
    fn name(self) -> &'static str {
        match self {
            Self::Check => "check",
            Self::Test => "test",
        }
    }

    fn timeout(self) -> Duration {
        match self {
            Self::Check => Duration::from_secs(CARGO_CHECK_TIMEOUT_SECS),
            Self::Test => Duration::from_secs(PACKAGE_TEST_TIMEOUT_SECS),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Check => "cargo check",
            Self::Test => "cargo test",
        }
    }
}

async fn run_affected_cargo_command(
    root: &Path,
    changed_files: &[String],
    already_ran: &mut BTreeSet<String>,
    command: CargoSubcommand,
) -> CargoCommandOutcome {
    let rust_paths = rust_source_paths(changed_files.iter());
    if rust_paths.is_empty() {
        return CargoCommandOutcome::Skipped;
    }
    let packages = affected_cargo_package_dirs(root, &rust_paths);
    let mut targets: Vec<(String, PathBuf)> = packages
        .into_iter()
        .map(|label| {
            let manifest = root.join(&label).join("Cargo.toml");
            (label, manifest)
        })
        .collect();
    if targets.is_empty() {
        // Root-package or workspace-root-only edits: one quiet command at root.
        let root_manifest = root.join("Cargo.toml");
        if root_manifest.is_file() {
            targets.push((".".into(), root_manifest));
        } else {
            return CargoCommandOutcome::Skipped;
        }
    }

    let runner = match ProcessRunner::new(root) {
        Ok(runner) => runner,
        Err(error) => {
            return CargoCommandOutcome::Unavailable {
                detail: format!("{} runner failed: {error:#}", command.label()),
            };
        }
    };

    let mut ran = Vec::new();
    for (label, manifest) in targets {
        if !already_ran.insert(label.clone()) {
            continue;
        }
        if !manifest.is_file() {
            continue;
        }
        let manifest_arg = manifest.to_string_lossy().into_owned();
        let args = vec![
            std::ffi::OsString::from(command.name()),
            std::ffi::OsString::from("--quiet"),
            std::ffi::OsString::from("--manifest-path"),
            std::ffi::OsString::from(&manifest_arg),
        ];
        let execution = match runner
            .run_program("cargo", &args, command.timeout())
            .await
        {
            Ok(execution) => execution,
            Err(error) => {
                return CargoCommandOutcome::Unavailable {
                    detail: format!(
                        "{} failed to start for {label}: {error:#}",
                        command.label()
                    ),
                };
            }
        };
        ran.push(label.clone());
        if execution.status != crate::ToolStatus::Succeeded {
            let body = bound_feedback(&execution.model_content());
            return CargoCommandOutcome::Failed {
                command: command.label(),
                package: label,
                output: body,
            };
        }
    }
    if ran.is_empty() {
        CargoCommandOutcome::Skipped
    } else {
        CargoCommandOutcome::Passed {
            command: command.label(),
            packages: ran,
        }
    }
}

/// Run package-local tests for non-Rust ecosystems touched by `changed_files`
/// (pytest / `npm test` / `go test`). First failure stops. Uses the same outcome
/// type and dedupe set as cargo tests so seals share one namespace.
pub async fn run_affected_polyglot_tests(
    root: &Path,
    changed_files: &[String],
    already_tested: &mut BTreeSet<String>,
) -> CargoCommandOutcome {
    let mut jobs: Vec<PackageTestJob> = Vec::new();

    let py_paths = python_source_paths(changed_files.iter());
    if !py_paths.is_empty() {
        let mut packages = affected_python_package_dirs(root, &py_paths);
        // Root-only Python tree (markers at workspace root).
        if packages.is_empty() && is_python_package_root(root) {
            packages.insert(".".into());
        }
        for label in packages {
            jobs.push(PackageTestJob {
                label: label.clone(),
                program: "pytest",
                args: vec![OsString::from("-q"), OsString::from(&label)],
            });
        }
    }

    let js_paths = javascript_source_paths(changed_files.iter());
    if !js_paths.is_empty() {
        let mut packages = affected_javascript_package_dirs(root, &js_paths);
        if packages.is_empty() && root.join("package.json").is_file() {
            packages.insert(".".into());
        }
        for label in packages {
            let prefix = if label == "." {
                ".".to_string()
            } else {
                label.clone()
            };
            jobs.push(PackageTestJob {
                label: label.clone(),
                program: "npm",
                args: vec![
                    OsString::from("--prefix"),
                    OsString::from(&prefix),
                    OsString::from("test"),
                    OsString::from("--silent"),
                ],
            });
        }
    }

    let go_paths = go_source_paths(changed_files.iter());
    if !go_paths.is_empty() {
        let mut packages = affected_go_package_dirs(root, &go_paths);
        if packages.is_empty() && root.join("go.mod").is_file() {
            packages.insert(".".into());
        }
        for label in packages {
            let dir = if label == "." {
                ".".to_string()
            } else {
                label.clone()
            };
            jobs.push(PackageTestJob {
                label: label.clone(),
                program: "go",
                args: vec![
                    OsString::from("-C"),
                    OsString::from(&dir),
                    OsString::from("test"),
                    OsString::from("./..."),
                ],
            });
        }
    }

    if jobs.is_empty() {
        return CargoCommandOutcome::Skipped;
    }

    let runner = match ProcessRunner::new(root) {
        Ok(runner) => runner,
        Err(error) => {
            return CargoCommandOutcome::Unavailable {
                detail: format!("polyglot test runner failed: {error:#}"),
            };
        }
    };

    let mut ran = Vec::new();
    for job in jobs {
        if !already_tested.insert(job.label.clone()) {
            continue;
        }
        let execution = match runner
            .run_program(
                job.program,
                &job.args,
                Duration::from_secs(PACKAGE_TEST_TIMEOUT_SECS),
            )
            .await
        {
            Ok(execution) => execution,
            Err(_error) => {
                // Missing toolchain (no pytest/npm/go) — skip that ecosystem.
                already_tested.remove(&job.label);
                continue;
            }
        };
        ran.push(job.label.clone());
        if execution.status != crate::ToolStatus::Succeeded {
            let body = bound_feedback(&execution.model_content());
            return CargoCommandOutcome::Failed {
                command: polyglot_command_label(job.program),
                package: job.label,
                output: body,
            };
        }
    }
    if ran.is_empty() {
        CargoCommandOutcome::Skipped
    } else {
        CargoCommandOutcome::Passed {
            command: "package test",
            packages: ran,
        }
    }
}

fn polyglot_command_label(program: &str) -> &'static str {
    match program {
        "pytest" => "pytest",
        "npm" => "npm test",
        "go" => "go test",
        _ => "package test",
    }
}

struct PackageTestJob {
    label: String,
    program: &'static str,
    args: Vec<OsString>,
}

/// Outcome of a mid-turn package check/test over affected packages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CargoCommandOutcome {
    Skipped,
    Passed {
        command: &'static str,
        packages: Vec<String>,
    },
    Failed {
        command: &'static str,
        package: String,
        output: String,
    },
    Unavailable {
        detail: String,
    },
}

/// Backward-compatible alias used by existing call sites / tests.
pub type CargoCheckOutcome = CargoCommandOutcome;

impl CargoCommandOutcome {
    /// Model-facing failure only (code errors). Infrastructure skips return None.
    pub fn failure_message(&self) -> Option<String> {
        match self {
            Self::Failed {
                command,
                package,
                output,
            } => {
                let structured = crate::format_structured_failure(
                    &format!("⚠ fast check · {command} ({package}) failed"),
                    output,
                    Some(
                        "Read the error above and fix its root cause before continuing — \
                         turn-end verify will re-run the same check.",
                    ),
                );
                Some(structured.body)
            }
            _ => None,
        }
    }

    pub fn ui_status(&self) -> Option<String> {
        match self {
            Self::Failed {
                command,
                package,
                output,
            } => {
                let structured = crate::format_structured_failure(
                    &format!("fast check · {command} ({package}) failed"),
                    output,
                    None,
                );
                Some(structured.summary)
            }
            Self::Unavailable { detail } => Some(format!("fast check · cargo skipped: {detail}")),
            // Passes are silent in the UI (avoid noise on every clean edit).
            Self::Passed { .. } | Self::Skipped => None,
        }
    }

    pub fn is_passed(&self) -> bool {
        matches!(self, Self::Passed { .. })
    }

    pub fn is_failed(&self) -> bool {
        matches!(self, Self::Failed { .. })
    }
}

/// Format LSP error diagnostics into a short model-facing block.
pub fn format_lsp_error_feedback(errors: &[(String, u32, u32, String)]) -> String {
    if errors.is_empty() {
        return String::new();
    }
    let mut lines = Vec::with_capacity(errors.len().min(24) + 1);
    lines.push(format!(
        "⚠ fast check · LSP diagnostics ({} error(s)):",
        errors.len()
    ));
    for (path, line, col, message) in errors.iter().take(24) {
        lines.push(format!("{path}:{line}:{col}: {message}"));
    }
    if errors.len() > 24 {
        lines.push(format!("… {} more error(s) omitted", errors.len() - 24));
    }
    bound_feedback(&lines.join("\n"))
}

fn bound_feedback(text: &str) -> String {
    let truncated = truncate(text);
    if truncated.chars().count() <= MAX_FEEDBACK_CHARS {
        return truncated;
    }
    let mut out = truncated
        .chars()
        .take(MAX_FEEDBACK_CHARS)
        .collect::<String>();
    out.push_str("\n… [fast check output truncated]");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn temp_workspace(label: &str) -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "hi-fast-fb-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("crates/demo/src")).unwrap();
        std::fs::write(
            root.join("Cargo.toml"),
            "[workspace]\nmembers = [\"crates/demo\"]\n",
        )
        .unwrap();
        std::fs::write(
            root.join("crates/demo/Cargo.toml"),
            "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        std::fs::write(root.join("crates/demo/src/lib.rs"), "pub fn ok() {}\n").unwrap();
        root
    }

    #[test]
    fn finds_nested_cargo_package_for_rust_paths() {
        let root = temp_workspace("pkg");
        let dirs = affected_cargo_package_dirs(
            &root,
            &["crates/demo/src/lib.rs".into(), "README.md".into()],
        );
        assert_eq!(
            dirs.into_iter().collect::<Vec<_>>(),
            vec!["crates/demo".to_string()]
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn rust_source_paths_filters_extensions() {
        let paths = rust_source_paths(["src/a.rs", "src/b.py", "crates/x/src/m.rs"]);
        assert_eq!(paths, vec!["crates/x/src/m.rs", "src/a.rs"]);
    }

    #[test]
    fn format_lsp_errors_names_locations() {
        let text = format_lsp_error_feedback(&[(
            "src/lib.rs".into(),
            4,
            1,
            "missing semicolon".into(),
        )]);
        assert!(text.contains("src/lib.rs:4:1"));
        assert!(text.contains("missing semicolon"));
        assert!(text.contains("LSP diagnostics"));
    }

    #[tokio::test]
    async fn cargo_check_passes_on_clean_package() {
        if std::process::Command::new("cargo")
            .arg("--version")
            .output()
            .map(|o| !o.status.success())
            .unwrap_or(true)
        {
            eprintln!("skipping: cargo not on PATH");
            return;
        }
        let root = temp_workspace("check-ok");
        let mut seen = BTreeSet::new();
        let outcome = run_affected_cargo_checks(
            &root,
            &["crates/demo/src/lib.rs".into()],
            &mut seen,
        )
        .await;
        match outcome {
            CargoCheckOutcome::Passed { packages, command } => {
                assert_eq!(command, "cargo check");
                assert_eq!(packages, vec!["crates/demo".to_string()]);
            }
            other => panic!("expected pass, got {other:?}"),
        }
        // Second call dedupes.
        let again =
            run_affected_cargo_checks(&root, &["crates/demo/src/lib.rs".into()], &mut seen).await;
        assert_eq!(again, CargoCheckOutcome::Skipped);
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn cargo_check_fails_on_broken_package() {
        if std::process::Command::new("cargo")
            .arg("--version")
            .output()
            .map(|o| !o.status.success())
            .unwrap_or(true)
        {
            eprintln!("skipping: cargo not on PATH");
            return;
        }
        let root = temp_workspace("check-bad");
        std::fs::write(
            root.join("crates/demo/src/lib.rs"),
            "pub fn broken( -> {}\n",
        )
        .unwrap();
        let mut seen = BTreeSet::new();
        let outcome = run_affected_cargo_checks(
            &root,
            &["crates/demo/src/lib.rs".into()],
            &mut seen,
        )
        .await;
        match outcome {
            CargoCheckOutcome::Failed {
                package,
                output,
                command,
            } => {
                assert_eq!(command, "cargo check");
                assert_eq!(package, "crates/demo");
                assert!(
                    output.contains("error") || !output.is_empty(),
                    "expected rustc diagnostics, got: {output}"
                );
            }
            other => panic!("expected fail, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn cargo_test_passes_on_clean_package() {
        if std::process::Command::new("cargo")
            .arg("--version")
            .output()
            .map(|o| !o.status.success())
            .unwrap_or(true)
        {
            eprintln!("skipping: cargo not on PATH");
            return;
        }
        let root = temp_workspace("test-ok");
        std::fs::write(
            root.join("crates/demo/src/lib.rs"),
            "pub fn ok() -> i32 { 1 }\n\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn it_works() { assert_eq!(super::ok(), 1); }\n}\n",
        )
        .unwrap();
        let mut seen = BTreeSet::new();
        let outcome =
            run_affected_cargo_tests(&root, &["crates/demo/src/lib.rs".into()], &mut seen).await;
        match outcome {
            CargoCommandOutcome::Passed { packages, command } => {
                assert_eq!(command, "cargo test");
                assert_eq!(packages, vec!["crates/demo".to_string()]);
            }
            other => panic!("expected test pass, got {other:?}"),
        }
        let again =
            run_affected_cargo_tests(&root, &["crates/demo/src/lib.rs".into()], &mut seen).await;
        assert_eq!(again, CargoCommandOutcome::Skipped);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn polyglot_source_and_package_discovery() {
        let root = std::env::temp_dir().join(format!(
            "hi-poly-disc-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("pkg/sub")).unwrap();
        std::fs::write(root.join("pkg/pyproject.toml"), "[project]\nname='pkg'\n").unwrap();
        std::fs::write(root.join("pkg/sub/mod.py"), "x=1\n").unwrap();
        std::fs::create_dir_all(root.join("web/src")).unwrap();
        std::fs::write(root.join("web/package.json"), "{\"name\":\"web\"}\n").unwrap();
        std::fs::write(root.join("web/src/app.ts"), "export {}\n").unwrap();
        std::fs::create_dir_all(root.join("svc")).unwrap();
        std::fs::write(root.join("svc/go.mod"), "module svc\n\ngo 1.22\n").unwrap();
        std::fs::write(root.join("svc/main.go"), "package main\n").unwrap();

        assert_eq!(
            python_source_paths(["pkg/sub/mod.py", "web/src/app.ts"]),
            vec!["pkg/sub/mod.py"]
        );
        assert_eq!(
            affected_python_package_dirs(&root, &["pkg/sub/mod.py".into()])
                .into_iter()
                .collect::<Vec<_>>(),
            vec!["pkg".to_string()]
        );
        assert_eq!(
            affected_javascript_package_dirs(&root, &["web/src/app.ts".into()])
                .into_iter()
                .collect::<Vec<_>>(),
            vec!["web".to_string()]
        );
        assert_eq!(
            affected_go_package_dirs(&root, &["svc/main.go".into()])
                .into_iter()
                .collect::<Vec<_>>(),
            vec!["svc".to_string()]
        );
        let any = affected_any_package_dirs(
            &root,
            &[
                "pkg/sub/mod.py".into(),
                "web/src/app.ts".into(),
                "svc/main.go".into(),
            ],
        );
        assert!(any.contains("pkg") && any.contains("web") && any.contains("svc"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn polyglot_pytest_fails_on_broken_test() {
        if std::process::Command::new("pytest")
            .arg("--version")
            .output()
            .map(|o| !o.status.success())
            .unwrap_or(true)
        {
            eprintln!("skipping: pytest not on PATH");
            return;
        }
        let root = std::env::temp_dir().join(format!(
            "hi-poly-py-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("demo")).unwrap();
        std::fs::write(root.join("demo/pyproject.toml"), "[project]\nname='demo'\n").unwrap();
        std::fs::write(
            root.join("demo/test_demo.py"),
            "def test_ok():\n    assert 1 == 2\n",
        )
        .unwrap();
        let mut seen = BTreeSet::new();
        let outcome =
            run_affected_polyglot_tests(&root, &["demo/test_demo.py".into()], &mut seen).await;
        match outcome {
            CargoCommandOutcome::Failed {
                package, command, ..
            } => {
                assert_eq!(package, "demo");
                assert_eq!(command, "pytest");
            }
            other => panic!("expected pytest fail, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn cargo_test_fails_on_broken_test() {
        if std::process::Command::new("cargo")
            .arg("--version")
            .output()
            .map(|o| !o.status.success())
            .unwrap_or(true)
        {
            eprintln!("skipping: cargo not on PATH");
            return;
        }
        let root = temp_workspace("test-bad");
        std::fs::write(
            root.join("crates/demo/src/lib.rs"),
            "pub fn ok() -> i32 { 1 }\n\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn it_works() { assert_eq!(super::ok(), 2); }\n}\n",
        )
        .unwrap();
        let mut seen = BTreeSet::new();
        let outcome =
            run_affected_cargo_tests(&root, &["crates/demo/src/lib.rs".into()], &mut seen).await;
        match outcome {
            CargoCommandOutcome::Failed {
                package,
                command,
                output,
            } => {
                assert_eq!(command, "cargo test");
                assert_eq!(package, "crates/demo");
                assert!(
                    !output.is_empty() || output.contains("assert"),
                    "expected test failure output, got: {output}"
                );
            }
            other => panic!("expected test fail, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(root);
    }
}
