use super::*;

#[test]
fn test_file_fixture_is_outside_repository_and_removed_with_its_owner() {
    let repository = std::env::current_dir().unwrap();
    let fixture = temp_file("fixture-cleanup");
    let path = fixture.as_ref().to_path_buf();

    assert!(path.starts_with(std::env::temp_dir()));
    assert!(!path.starts_with(&repository));
    std::fs::write(&path, "temporary").unwrap();
    assert!(path.exists());

    drop(fixture);
    assert!(!path.exists(), "temporary test fixture was not removed");
    assert!(
        !repository.join("hi-test-scratch").exists(),
        "tests must not recreate repository-relative scratch state"
    );
}
