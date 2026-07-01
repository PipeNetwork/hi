//! Implementation heuristics: GPU training estimator bootstrap file
//! generation, tool-call mutation/edit/validation classification, and shell
//! command analysis. Uses [`contains_any`] from [`intent`](super::intent) and
//! [`ImplementationIntent`] from [`types`](super::types).

use super::intent::contains_any;
use super::types::ImplementationIntent;
pub(crate) fn should_bootstrap_gpu_training_estimator(intent: ImplementationIntent) -> bool {
    intent.gpu_training_estimator && implementation_workspace_can_accept_rust_bootstrap()
}

pub(crate) fn implementation_workspace_can_accept_rust_bootstrap() -> bool {
    implementation_workspace_can_accept_rust_bootstrap_at(std::path::Path::new("."))
}

pub(crate) fn implementation_workspace_can_accept_rust_bootstrap_at(
    root: &std::path::Path,
) -> bool {
    let manifest_paths = [
        "Cargo.toml",
        "package.json",
        "pyproject.toml",
        "go.mod",
        "Makefile",
        "justfile",
    ];
    if manifest_paths.iter().any(|path| root.join(path).exists()) {
        return false;
    }
    if ["src/main.rs", "src/lib.rs"]
        .iter()
        .any(|path| root.join(path).exists())
    {
        return false;
    }
    if let Ok(mut entries) = std::fs::read_dir(root.join("src"))
        && entries.next().is_some()
    {
        return false;
    }
    true
}

pub(crate) fn gpu_training_estimator_bootstrap_files(
    intent: ImplementationIntent,
) -> Vec<(&'static str, String)> {
    vec![
        (
            "Cargo.toml",
            gpu_training_estimator_bootstrap_cargo_toml(intent),
        ),
        ("src/lib.rs", gpu_training_estimator_bootstrap_lib_rs()),
        (
            "src/main.rs",
            gpu_training_estimator_bootstrap_main_rs(intent),
        ),
    ]
}

pub(crate) fn gpu_training_estimator_bootstrap_cargo_toml(intent: ImplementationIntent) -> String {
    if intent.tui {
        include_str!("../../templates/gpu_estimator/Cargo.toml.tui").to_string()
    } else {
        include_str!("../../templates/gpu_estimator/Cargo.toml.cli").to_string()
    }
}

pub(crate) fn gpu_training_estimator_bootstrap_lib_rs() -> String {
    include_str!("../../templates/gpu_estimator/lib.rs").to_string()
}

pub(crate) fn gpu_training_estimator_bootstrap_main_rs(intent: ImplementationIntent) -> String {
    if intent.tui {
        include_str!("../../templates/gpu_estimator/main.rs.tui").to_string()
    } else {
        include_str!("../../templates/gpu_estimator/main.rs.cli").to_string()
    }
}
pub(crate) fn bash_command(arguments: &str) -> Option<String> {
    let value = serde_json::from_str::<serde_json::Value>(arguments).ok()?;
    value
        .get("command")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}

pub(crate) fn implementation_tool_call_mutates(name: &str, arguments: &str) -> bool {
    if hi_tools::is_filesystem_mutating(name) {
        return true;
    }
    if name != "bash" {
        return false;
    }
    let Some(command) = bash_command(arguments) else {
        return false;
    };
    shell_command_likely_mutates_workspace(&command)
}

pub(crate) fn implementation_tool_call_substantively_edits(name: &str, arguments: &str) -> bool {
    if matches!(name, "write" | "edit" | "multi_edit" | "apply_patch") {
        return true;
    }
    if name != "bash" {
        return false;
    }
    let Some(command) = bash_command(arguments) else {
        return false;
    };
    shell_command_likely_edits_files(&command)
}

pub(crate) fn implementation_tool_call_validates(name: &str, arguments: &str) -> bool {
    if name != "bash" {
        return false;
    }
    let Some(command) = bash_command(arguments) else {
        return false;
    };
    shell_command_likely_validates(&command)
}

pub(crate) fn shell_command_likely_mutates_workspace(command: &str) -> bool {
    let compact = command
        .to_ascii_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    contains_any(
        &compact,
        &[
            "cargo init",
            "npm init",
            "pnpm init",
            "yarn init",
            "bun init",
            "cargo add",
            "npm install",
            "pnpm add",
            "yarn add",
            "bun add",
            "mkdir ",
            "touch ",
            "cat >",
            "tee ",
            "sed -i",
            "apply_patch",
            "patch -p",
        ],
    )
}

pub(crate) fn shell_command_likely_edits_files(command: &str) -> bool {
    let compact = command
        .to_ascii_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    contains_any(
        &compact,
        &[
            "cat >",
            "cat <<",
            "tee ",
            "sed -i",
            "perl -i",
            "apply_patch",
            "patch -p",
            "python - <<",
            "python3 - <<",
        ],
    )
}

pub(crate) fn shell_command_likely_validates(command: &str) -> bool {
    let compact = command
        .to_ascii_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    contains_any(
        &compact,
        &[
            "cargo test",
            "cargo check",
            "cargo build",
            "cargo clippy",
            "npm test",
            "npm run test",
            "npm run build",
            "npm run check",
            "npm run lint",
            "pnpm test",
            "pnpm build",
            "pnpm check",
            "pnpm lint",
            "yarn test",
            "yarn build",
            "bun test",
            "bun run build",
            "pytest",
            "python -m pytest",
            "go test",
            "make test",
            "make check",
            "make build",
            "just test",
            "just check",
            "just build",
            "timeout 5s cargo run",
            "cargo run --",
        ],
    )
}
