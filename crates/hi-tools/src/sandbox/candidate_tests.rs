use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use super::*;

#[test]
fn pipe_wrap_candidates_never_use_path_or_writable_siblings() {
    let writable = PathBuf::from("/work/project");
    let candidates = pipe_wrap_candidates(
        Some(OsStr::new("pipe-wrap")),
        Some(Path::new("/work/project/target/debug/hi")),
        std::slice::from_ref(&writable),
    );
    assert!(candidates.is_empty());

    let candidates = pipe_wrap_candidates(
        Some(OsStr::new("/operator/pipe-wrap")),
        Some(Path::new("/opt/hi/bin/hi")),
        std::slice::from_ref(&writable),
    );
    assert_eq!(
        candidates,
        [
            PathBuf::from("/operator/pipe-wrap"),
            PathBuf::from("/opt/hi/bin/pipe-wrap")
        ]
    );
    assert!(!candidates.contains(&PathBuf::from("/untrusted/path/pipe-wrap")));
    assert!(pipe_wrap_candidates(None, Some(Path::new("/tmp/hi")), &[]).is_empty());
}

#[test]
fn hermetic_profile_omits_broad_host_temp_write_rules() {
    let root = tempfile::tempdir().unwrap();
    let config = SandboxConfig {
        deny_host_temp: true,
        ..SandboxConfig::default()
    };
    let profile = seatbelt_profile_with_protected_paths(
        SandboxPolicy::Workspace,
        &[root.path()],
        &config,
        &[],
    );
    let canonical_root = root.path().canonicalize().unwrap();
    assert!(profile.contains(canonical_root.to_str().unwrap()));
    for temp in temp_roots() {
        if Path::new(&temp) != canonical_root {
            assert!(
                !profile.contains(&format!("(allow file-write* (subpath {}))", quote(&temp))),
                "hermetic profile exposed host temp {temp}: {profile}"
            );
        }
    }
}

#[test]
fn strict_candidate_profile_exposes_only_candidate_and_private_temp() {
    let owner = tempfile::tempdir().unwrap();
    let candidate = owner.path().join("candidate");
    let private_temp = owner.path().join("state/private-tmp");
    let source = owner.path().join("source");
    std::fs::create_dir_all(&candidate).unwrap();
    std::fs::create_dir_all(&private_temp).unwrap();
    std::fs::create_dir_all(&source).unwrap();
    let config = SandboxConfig {
        deny_read: vec![source.clone()],
        deny_host_temp: true,
        private_temp: Some(private_temp.clone()),
        ..SandboxConfig::default()
    };
    let profile =
        SandboxProfile::with_config(SandboxPolicy::Strict, &[candidate.as_path()], config);
    let candidate = candidate.canonicalize().unwrap();
    let private_temp = private_temp.canonicalize().unwrap();
    let source = source.canonicalize().unwrap();

    #[cfg(target_os = "macos")]
    {
        let candidate_allow = format!(
            "(allow file-write* (subpath {}))",
            quote(candidate.to_str().unwrap())
        );
        let private_temp_allow = format!(
            "(allow file-write* (subpath {}))",
            quote(private_temp.to_str().unwrap())
        );
        let source_deny = format!(
            "(deny file-read* (subpath {}))",
            quote(source.to_str().unwrap())
        );
        assert!(profile.profile.contains(&candidate_allow));
        assert!(profile.profile.contains(&private_temp_allow));
        assert!(profile.profile.contains(&source_deny));
        for temp in temp_roots() {
            for access in ["file-read*", "file-write*"] {
                assert!(
                    !profile
                        .profile
                        .contains(&format!("(allow {access} (subpath {}))", quote(&temp))),
                    "candidate profile exposed broad host temp {temp} for {access}: {}",
                    profile.profile
                );
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        assert!(profile.writable_roots.contains(&candidate));
        assert!(profile.writable_roots.contains(&private_temp));
        assert!(!profile.writable_roots.contains(&source));
        assert!(profile.config.deny_host_temp);
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    let _ = (profile, candidate, private_temp, source);
}

#[test]
fn linux_candidate_mount_plan_keeps_host_temp_private() {
    let owner = tempfile::tempdir().unwrap();
    let candidate = owner.path().join("candidate");
    let private_temp = owner.path().join("state/private-tmp");
    let source = owner.path().join("source");
    std::fs::create_dir_all(&candidate).unwrap();
    std::fs::create_dir_all(private_temp.join(PRIVATE_DENY_READ_MASK_DIR)).unwrap();
    std::fs::create_dir_all(&source).unwrap();
    let config = SandboxConfig {
        deny_read: vec![source.clone()],
        deny_host_temp: true,
        private_temp: Some(private_temp.clone()),
        ..SandboxConfig::default()
    };
    let args = pipe_wrap_arguments_with_protected_roots(
        SandboxPolicy::Strict,
        &config,
        &[candidate.clone(), private_temp.clone()],
        &[],
        OsStr::new("true"),
        &[],
    );
    let bind = |path: &Path| {
        args.windows(3).any(|window| {
            window[0] == OsStr::new("--bind")
                && window[1] == path.as_os_str()
                && window[2] == path.as_os_str()
        })
    };
    assert!(bind(&candidate), "candidate root missing: {args:?}");
    assert!(bind(&private_temp), "private temp missing: {args:?}");
    assert!(
        args.windows(2).any(|window| {
            window[0] == OsStr::new("--tmpfs") && window[1] == OsStr::new("/tmp")
        }),
        "host temp is not namespace-private: {args:?}"
    );
    let deny_mask = private_temp.join(PRIVATE_DENY_READ_MASK_DIR);
    assert!(
        args.windows(3).any(|window| {
            window[0] == OsStr::new("--ro-bind")
                && window[1] == deny_mask.as_os_str()
                && window[2] == source.as_os_str()
        }),
        "source workspace is not hidden behind an empty overlay: {args:?}"
    );
    assert!(!bind(&source), "source workspace became writable: {args:?}");
}
