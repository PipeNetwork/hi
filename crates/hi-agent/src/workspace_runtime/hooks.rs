use std::path::Path;
use std::sync::Arc;

/// Load global hooks and, when explicitly permitted, the local hook directory
/// only after folder trust has been resolved for this machine. Keeping this in
/// one helper makes initial and deferred runtimes use identical trust rules.
pub(super) fn discover_hooks(
    root: &Path,
    allow_project_hooks: bool,
) -> Option<Arc<hi_hooks::HookRegistry>> {
    let home = std::env::var("HOME")
        .ok()
        .map(|h| Path::new(&h).join(".hi/hooks"));
    let project_hooks = root.join(".hi/hooks");
    let project_hooks_dir = if allow_project_hooks {
        match hi_tools::folder_trust::resolve_trust(root) {
            hi_tools::folder_trust::TrustOutcome::Trusted => Some(project_hooks.as_path()),
            hi_tools::folder_trust::TrustOutcome::Untrusted
            | hi_tools::folder_trust::TrustOutcome::Prompt => None,
        }
    } else {
        None
    };
    let (hooks, errors) = hi_hooks::discover_hooks(home.as_deref(), project_hooks_dir);
    for error in &errors {
        eprintln!("hook load warning: {error}");
    }
    (!hooks.is_empty()).then(|| Arc::new(hooks))
}
