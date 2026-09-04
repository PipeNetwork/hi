use std::collections::BTreeMap;

use super::*;
use hi_workspace::{ControllerId, WorkspaceId};

fn git(root: &Path, args: &[&str]) {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "git failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn init_git(source: &Path) {
    fs::create_dir(source).unwrap();
    git(source, &["init", "--quiet"]);
    fs::write(source.join("tracked.txt"), "base\n").unwrap();
    git(source, &["add", "tracked.txt"]);
    git(
        source,
        &[
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@invalid",
            "commit",
            "--quiet",
            "-m",
            "base",
        ],
    );
}

fn tree_bytes(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    fn visit(root: &Path, at: &Path, files: &mut BTreeMap<PathBuf, Vec<u8>>) {
        for entry in fs::read_dir(at).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if entry.file_type().unwrap().is_dir() {
                visit(root, &path, files);
            } else {
                files.insert(
                    path.strip_prefix(root).unwrap().to_path_buf(),
                    fs::read(path).unwrap(),
                );
            }
        }
    }
    let mut files = BTreeMap::new();
    visit(root, root, &mut files);
    files
}

fn binding(source: &Path, state: &Path) -> WorkspaceBinding {
    WorkspaceBinding::new_local(
        ControllerId::new("controller"),
        WorkspaceId::new("workspace"),
        source.to_path_buf(),
        state.to_path_buf(),
    )
}

fn context(binding: WorkspaceBinding) -> CandidateSealContext {
    CandidateSealContext {
        job_id: JobId::new("candidate-job"),
        binding,
        route: CandidateRoute {
            provider: "test".into(),
            model: "test-model".into(),
            actual_model_revision: None,
            capability_digest: "blake3:test-capabilities".into(),
        },
        verification: vec![CandidateVerification {
            name: "test".into(),
            passed: true,
            verifier_digest: "blake3:test-verification".into(),
            detail: None,
            artifacts: Vec::new(),
        }],
        destination_verification: vec![hi_workspace::CandidateDestinationVerifier {
            name: "test".into(),
            command: "true".into(),
            timeout_ms: 5_000,
        }],
        destination_verification_budget_ms: 5_000,
    }
}

#[test]
fn git_candidate_is_exact_private_and_does_not_mutate_source_git() {
    let temporary = tempfile::tempdir().unwrap();
    let source = temporary.path().join("source");
    let state = temporary.path().join("state");
    fs::create_dir(&state).unwrap();
    init_git(&source);
    fs::write(source.join("tracked.txt"), "dirty\n").unwrap();
    fs::write(source.join("untracked.txt"), "new\n").unwrap();
    fs::write(source.join(".gitignore"), "ignored.txt\n").unwrap();
    fs::write(source.join("ignored.txt"), "included exact state\n").unwrap();
    let git_before = tree_bytes(&source.join(".git"));

    let candidate_path = temporary.path().join("candidate");
    let candidate = CandidateWorkspace::create(&source, &state, &candidate_path).unwrap();
    assert_eq!(
        candidate.materialization(),
        CandidateMaterialization::PrivateClone
    );
    assert_eq!(
        fs::read_to_string(candidate.root().join("tracked.txt")).unwrap(),
        "dirty\n"
    );
    assert_eq!(
        fs::read_to_string(candidate.root().join("ignored.txt")).unwrap(),
        "included exact state\n"
    );
    assert!(
        !candidate
            .root()
            .join(".git/objects/info/alternates")
            .exists()
    );
    let remotes = Command::new("git")
        .arg("-C")
        .arg(candidate.root())
        .arg("remote")
        .output()
        .unwrap();
    assert!(remotes.status.success());
    assert!(
        remotes.stdout.is_empty(),
        "candidate must not retain source transport"
    );
    fs::write(candidate.root().join("tracked.txt"), "candidate\n").unwrap();
    assert_eq!(
        fs::read_to_string(source.join("tracked.txt")).unwrap(),
        "dirty\n"
    );
    assert_eq!(tree_bytes(&source.join(".git")), git_before);
    drop(candidate);
    assert!(!candidate_path.exists());
}

#[test]
fn nested_git_workspace_is_private_and_does_not_mutate_source_git() {
    let temporary = tempfile::tempdir().unwrap();
    let repository = temporary.path().join("source");
    let source = repository.join("nested");
    let state = temporary.path().join("state");
    fs::create_dir(&state).unwrap();
    init_git(&repository);
    fs::create_dir(&source).unwrap();
    fs::write(source.join("tracked.txt"), "nested base\n").unwrap();
    git(&repository, &["add", "nested/tracked.txt"]);
    git(
        &repository,
        &[
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@invalid",
            "commit",
            "--quiet",
            "-m",
            "nested base",
        ],
    );
    fs::write(source.join("tracked.txt"), "nested dirty\n").unwrap();
    fs::write(source.join("untracked.txt"), "nested new\n").unwrap();
    let git_before = tree_bytes(&repository.join(".git"));

    let candidate =
        CandidateWorkspace::create(&source, &state, &temporary.path().join("nested-candidate"))
            .unwrap();
    assert_eq!(
        candidate.materialization(),
        CandidateMaterialization::PrivateClone
    );
    assert_eq!(
        fs::read_to_string(candidate.root().join("tracked.txt")).unwrap(),
        "nested dirty\n"
    );
    assert_eq!(
        fs::read_to_string(candidate.root().join("untracked.txt")).unwrap(),
        "nested new\n"
    );
    assert!(candidate.root().join(".git").is_dir());
    assert_eq!(tree_bytes(&repository.join(".git")), git_before);
}

#[test]
fn sealed_candidate_applies_only_through_parent_transaction() {
    let temporary = tempfile::tempdir().unwrap();
    let source = temporary.path().join("source");
    let state = temporary.path().join("state");
    fs::create_dir(&state).unwrap();
    init_git(&source);
    let git_before = tree_bytes(&source.join(".git"));
    let candidate =
        CandidateWorkspace::create(&source, &state, &temporary.path().join("candidate")).unwrap();
    fs::write(candidate.root().join("tracked.txt"), "candidate\n").unwrap();
    fs::write(candidate.root().join("added.txt"), "added\n").unwrap();
    let binding = binding(&source, &state);
    let sealed = candidate.seal_verified(context(binding.clone())).unwrap();

    assert_eq!(
        fs::read_to_string(source.join("tracked.txt")).unwrap(),
        "base\n"
    );
    assert!(!source.join("added.txt").exists());
    assert_eq!(tree_bytes(&source.join(".git")), git_before);
    let changes = apply_verified_candidate(&sealed, &binding, &state).unwrap();
    assert_eq!(changes.len(), 2);
    assert_eq!(
        fs::read_to_string(source.join("tracked.txt")).unwrap(),
        "candidate\n"
    );
    assert_eq!(
        fs::read_to_string(source.join("added.txt")).unwrap(),
        "added\n"
    );
}

#[test]
fn complete_base_change_makes_candidate_stale_before_apply() {
    let temporary = tempfile::tempdir().unwrap();
    let source = temporary.path().join("source");
    let state = temporary.path().join("state");
    fs::create_dir(&state).unwrap();
    init_git(&source);
    let candidate =
        CandidateWorkspace::create(&source, &state, &temporary.path().join("candidate")).unwrap();
    fs::write(candidate.root().join("tracked.txt"), "candidate\n").unwrap();
    let binding = binding(&source, &state);
    let sealed = candidate.seal_verified(context(binding.clone())).unwrap();
    fs::write(source.join("unrelated.txt"), "concurrent\n").unwrap();

    let error = apply_verified_candidate(&sealed, &binding, &state).unwrap_err();
    assert!(error.to_string().contains("stale"), "{error:#}");
    assert_eq!(
        fs::read_to_string(source.join("tracked.txt")).unwrap(),
        "base\n"
    );
}

#[test]
fn non_git_candidate_receives_a_private_repository() {
    let temporary = tempfile::tempdir().unwrap();
    let source = temporary.path().join("source");
    let state = temporary.path().join("state");
    fs::create_dir(&source).unwrap();
    fs::create_dir(&state).unwrap();
    fs::write(source.join("hello.txt"), "hello\n").unwrap();
    let candidate =
        CandidateWorkspace::create(&source, &state, &temporary.path().join("candidate")).unwrap();
    assert_eq!(
        candidate.materialization(),
        CandidateMaterialization::PrivateInit
    );
    assert!(candidate.root().join(".git").is_dir());
    assert!(!candidate.baseline_commit().is_empty());
}

#[test]
fn nested_git_metadata_is_never_materialized() {
    let temporary = tempfile::tempdir().unwrap();
    let source = temporary.path().join("source");
    let state = temporary.path().join("state");
    fs::create_dir_all(source.join("nested")).unwrap();
    fs::create_dir(&state).unwrap();
    fs::write(source.join("nested/.git"), "gitdir: /unsafe/source\n").unwrap();
    let candidate =
        CandidateWorkspace::create(&source, &state, &temporary.path().join("candidate")).unwrap();
    assert!(!candidate.root().join("nested/.git").exists());
}

#[test]
fn sealed_candidate_artifact_survives_restart_until_terminal_cleanup() {
    let temporary = tempfile::tempdir().unwrap();
    let source = temporary.path().join("source");
    let state = temporary.path().join("state");
    fs::create_dir(&state).unwrap();
    init_git(&source);
    let candidate =
        CandidateWorkspace::create(&source, &state, &temporary.path().join("candidate")).unwrap();
    fs::write(candidate.root().join("tracked.txt"), "candidate\n").unwrap();
    let sealed = candidate
        .seal_verified(context(binding(&source, &state)))
        .unwrap();
    let persisted = PersistedDetachedCandidate::persist(sealed.clone(), &state).unwrap();
    assert!(persisted.path().is_file());
    assert!(persisted.artifact.uri.starts_with("artifact://candidate/"));
    let artifact_uri = hi_workspace::ResourceUri::parse(&persisted.artifact.uri).unwrap();
    let mut cache = crate::paths::ReadCache::new();
    cache.set_resource_state_root(state.clone());
    let resource_body = cache
        .resource(&artifact_uri)
        .expect("persisted artifact resolves through the production read cache");
    assert!(resource_body.contains(sealed.candidate.candidate_id.as_str()));

    let mut discovered = PersistedDetachedCandidate::discover(&state).unwrap();
    assert_eq!(discovered.len(), 1);
    assert_eq!(discovered[0].detached, sealed);
    let path = discovered[0].path().to_path_buf();
    discovered.pop().unwrap().remove_after_terminal().unwrap();
    assert!(!path.exists());
}
