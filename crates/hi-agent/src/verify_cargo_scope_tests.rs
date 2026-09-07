use super::*;
use crate::recovery::{TaskRecoveryState, ValidationResult};

fn observation(root: &Path, command: &str, status: ValidationResult) -> ValidationObservation {
    ValidationObservation::command(
        uuid::Uuid::new_v4().to_string(),
        command,
        "current".into(),
        status,
        "error[E0308]: mismatched types",
        root,
        false,
    )
}

fn workspace() -> tempfile::TempDir {
    let directory = tempfile::tempdir().unwrap();
    for package in [".", "crates/core", "crates/consumer", "quoted'name"] {
        let path = directory.path().join(package);
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(
            path.join("Cargo.toml"),
            "[package]\nname='test'\nversion='0.1.0'\n",
        )
        .unwrap();
    }
    directory
}

#[test]
fn native_default_checks_share_exact_manifest_identity() {
    let directory = workspace();
    let root = directory.path();
    for package in [".", "crates/core", "quoted'name"] {
        let mut fast = observation(root, "legacy fast label", ValidationResult::Failed);
        canonicalize_package(&mut fast, root, "cargo check", package);
        let manifest = root.join(package).join("Cargo.toml");
        let command = format!(
            "cargo check --quiet --manifest-path {}",
            super::super::shell_quote(manifest.to_str().unwrap())
        );
        let mut final_check = observation(root, &command, ValidationResult::Passed);
        canonicalize_command(&mut final_check, root, &command);
        assert_eq!(fast.scope, final_check.scope);
        let mut state = TaskRecoveryState::default();
        state.observe(&fast);
        assert!(state.unresolved_validation_status("current").is_some());
        state.observe(&final_check);
        assert!(state.unresolved_validation_status("current").is_none());
    }
    let mut plain = observation(root, "cargo check --quiet", ValidationResult::Passed);
    canonicalize_command(&mut plain, root, "cargo check --quiet");
    assert_eq!(plain.scope, "cargo check [package:.]");
}

#[test]
fn native_pass_clears_legacy_literal_scope_without_rewriting_saved_wire() {
    let directory = workspace();
    let root = directory.path();
    let mut state = TaskRecoveryState::default();
    for command in ["cargo check [package:.]", "cargo check --quiet"] {
        state.observe(&observation(root, command, ValidationResult::Failed));
    }
    let wire = serde_json::to_string(&state).unwrap();
    let mut restored: TaskRecoveryState = serde_json::from_str(&wire).unwrap();
    assert_eq!(serde_json::to_string(&restored).unwrap(), wire);
    let mut pass = observation(root, "cargo check --quiet", ValidationResult::Passed);
    canonicalize_command(&mut pass, root, "cargo check --quiet");
    restored.observe(&pass);
    assert!(restored.unresolved_validation_status("current").is_none());
    let replayed: ValidationObservation =
        serde_json::from_value(serde_json::to_value(pass).unwrap()).unwrap();
    assert!(
        replayed.equivalent_scopes.is_empty(),
        "native aliases are execution-local proof"
    );
}

#[test]
fn semantic_flags_shell_programs_and_other_packages_cannot_clear_failure() {
    let directory = workspace();
    let root = directory.path();
    let commands = [
        "cargo check --features extra",
        "cargo check --no-default-features",
        "cargo check --workspace",
        "cargo check -p core",
        "cargo check --all-targets",
        "cargo check --target wasm32-unknown-unknown",
        "cargo check --bin app",
        "cargo check --quiet && true",
        "CARGO_FEATURES=extra cargo check",
        "cargo +nightly check",
        "cargo check --manifest-path \"$MANIFEST\"",
        "cargo check --manifest-path 'Cargo.toml' --features extra",
    ];
    for command in commands {
        let mut pass = observation(root, command, ValidationResult::Passed);
        canonicalize_command(&mut pass, root, command);
        assert_eq!(pass.scope, command);
        assert!(pass.equivalent_scopes.is_empty());
        let mut failed = observation(root, "cargo check [package:.]", ValidationResult::Failed);
        canonicalize_package(&mut failed, root, "cargo check", ".");
        let mut state = TaskRecoveryState::default();
        state.observe(&failed);
        state.observe(&pass);
        assert!(
            state.unresolved_validation_status("current").is_some(),
            "{command}"
        );
    }
    // A root/default or another package pass also cannot erase the narrower
    // manifest's failure: workspace feature unification is not equivalent.
    for package in [".", "crates/consumer"] {
        let mut state = TaskRecoveryState::default();
        state.observe(&observation(
            root,
            "cargo check [package:crates/core]",
            ValidationResult::Failed,
        ));
        state.observe(&observation(
            root,
            "cargo check --workspace",
            ValidationResult::Failed,
        ));
        let mut pass = observation(root, "cargo check", ValidationResult::Passed);
        canonicalize_package(&mut pass, root, "cargo check", package);
        state.observe(&pass);
        assert!(state.unresolved_validation_status("current").is_some());
    }
}

#[cfg(unix)]
#[test]
fn manifest_outside_workspace_is_not_native_coverage() {
    let directory = workspace();
    let external = workspace();
    std::os::unix::fs::symlink(external.path(), directory.path().join("outside")).unwrap();
    let command = "cargo check --quiet --manifest-path 'outside/Cargo.toml'";
    let mut result = observation(directory.path(), command, ValidationResult::Passed);
    canonicalize_command(&mut result, directory.path(), command);
    assert_eq!(result.scope, command);
    assert!(result.equivalent_scopes.is_empty());
}
