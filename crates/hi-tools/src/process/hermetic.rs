//! Construction of process runners with narrow private temporary storage.

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};

use super::{ForegroundProcessRegistry, ProcessRunner, workspace_cargo_home};

pub(super) fn build_process_runner(
    root: &Path,
    policy: crate::sandbox::SandboxPolicy,
    mut config: crate::sandbox::SandboxConfig,
) -> Result<ProcessRunner> {
    let metadata = std::fs::metadata(root)
        .with_context(|| format!("reading workspace root {}", root.display()))?;
    anyhow::ensure!(
        metadata.is_dir(),
        "workspace root is not a directory: {}",
        root.display()
    );
    let root = root
        .canonicalize()
        .with_context(|| format!("canonicalizing workspace root {}", root.display()))?;
    let private_temp = config
        .private_temp
        .as_deref()
        .map(prepare_private_temp)
        .transpose()?;
    config.private_temp = private_temp.clone();
    let cargo_home = workspace_cargo_home(&root, policy)
        .filter(|cargo_home| std::fs::create_dir_all(cargo_home).is_ok());
    let mut writable = vec![root.as_path()];
    if let Some(cargo_home) = cargo_home.as_deref()
        && !cargo_home.starts_with(&root)
    {
        writable.push(cargo_home);
    }
    let sandbox = crate::sandbox::SandboxProfile::with_config(policy, &writable, config);
    warn_if_unenforced(&sandbox);
    Ok(ProcessRunner {
        root,
        foreground: ForegroundProcessRegistry::default(),
        sandbox,
        cargo_home,
        private_temp,
    })
}

fn warn_if_unenforced(sandbox: &crate::sandbox::SandboxProfile) {
    if sandbox.requested_but_unenforced() {
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            eprintln!(
                "warning: {}",
                crate::sandbox::SandboxProfile::unenforced_warning()
            );
        });
    }
}

fn prepare_private_temp(path: &Path) -> Result<PathBuf> {
    anyhow::ensure!(
        path.is_absolute(),
        "sandbox private temp must be absolute: {}",
        path.display()
    );
    std::fs::create_dir_all(path)
        .with_context(|| format!("creating sandbox private temp {}", path.display()))?;
    let path = path
        .canonicalize()
        .with_context(|| format!("canonicalizing sandbox private temp {}", path.display()))?;
    anyhow::ensure!(
        path.is_dir(),
        "sandbox private temp is not a directory: {}",
        path.display()
    );
    let broad_temp_roots = [
        PathBuf::from("/tmp"),
        PathBuf::from("/private/tmp"),
        PathBuf::from("/var/tmp"),
        PathBuf::from("/private/var/tmp"),
        PathBuf::from("/var/folders"),
        PathBuf::from("/private/var/folders"),
        std::env::temp_dir(),
    ];
    anyhow::ensure!(
        !broad_temp_roots
            .iter()
            .filter_map(|root| root.canonicalize().ok())
            .any(|root| root == path),
        "sandbox private temp must not expose a broad host temp root: {}",
        path.display()
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700))
            .with_context(|| format!("securing sandbox private temp {}", path.display()))?;
    }
    let deny_read_mask = path.join(crate::sandbox::PRIVATE_DENY_READ_MASK_DIR);
    std::fs::create_dir_all(&deny_read_mask)
        .with_context(|| format!("creating deny-read mask {}", deny_read_mask.display()))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runner_uses_one_owner_only_private_temp() {
        let owner = tempfile::tempdir().unwrap();
        let candidate = owner.path().join("candidate");
        let source = owner.path().join("source");
        let private_temp = owner.path().join("state/private-tmp");
        std::fs::create_dir_all(&candidate).unwrap();
        std::fs::create_dir_all(&source).unwrap();
        let runner = ProcessRunner::new_with_policy_and_config(
            &candidate,
            crate::sandbox::SandboxPolicy::Strict,
            crate::sandbox::SandboxConfig {
                deny_read: vec![source],
                deny_host_temp: true,
                private_temp: Some(private_temp.clone()),
                ..crate::sandbox::SandboxConfig::default()
            },
        )
        .unwrap();
        let private_temp = private_temp.canonicalize().unwrap();
        assert_eq!(runner.private_temp.as_deref(), Some(private_temp.as_path()));
        assert!(
            private_temp
                .join(crate::sandbox::PRIVATE_DENY_READ_MASK_DIR)
                .is_dir()
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mode = std::fs::metadata(&private_temp)
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o700);
        }
    }

    #[test]
    fn runner_rejects_a_broad_host_temp_root() {
        let candidate = tempfile::tempdir().unwrap();
        let error = ProcessRunner::new_with_policy_and_config(
            candidate.path(),
            crate::sandbox::SandboxPolicy::Strict,
            crate::sandbox::SandboxConfig {
                deny_host_temp: true,
                private_temp: Some(std::env::temp_dir()),
                ..crate::sandbox::SandboxConfig::default()
            },
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("must not expose a broad host temp root")
        );
    }
}
