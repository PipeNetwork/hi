//! Detect the language of a file or project and resolve the LSP server command.

use std::path::{Path, PathBuf};

/// A language `hi` can talk to an LSP server for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Language {
    Rust,
    Python,
    Go,
    TypeScript,
}

impl Language {
    /// The LSP `languageId` string for this language.
    pub fn language_id(self) -> &'static str {
        match self {
            Language::Rust => "rust",
            Language::Python => "python",
            Language::Go => "go",
            Language::TypeScript => "typescript",
        }
    }
}

/// Detect the language for a single file by extension.
pub fn detect_language(path: &Path) -> Option<Language> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "rs" => Some(Language::Rust),
        "py" => Some(Language::Python),
        "go" => Some(Language::Go),
        "ts" | "tsx" => Some(Language::TypeScript),
        _ => None,
    }
}

/// Detect the primary language of a project by looking for marker files.
pub fn detect_project_language(root: &Path) -> Option<Language> {
    if root.join("Cargo.toml").exists() {
        return Some(Language::Rust);
    }
    if root.join("pyproject.toml").exists() || root.join("setup.py").exists() {
        return Some(Language::Python);
    }
    if root.join("go.mod").exists() {
        return Some(Language::Go);
    }
    if root.join("tsconfig.json").exists() {
        return Some(Language::TypeScript);
    }
    None
}

/// The server binary name and default args for a language.
///
/// Returns `(command, args)`. The command is expected on `$PATH`; if missing,
/// the caller should surface a clear install hint rather than failing silently.
pub fn server_command(lang: Language) -> (&'static str, Vec<&'static str>) {
    match lang {
        Language::Rust => ("rust-analyzer", vec![]),
        Language::Python => ("pyright-langserver", vec!["--stdio"]),
        Language::Go => ("gopls", vec!["serve"]),
        Language::TypeScript => ("typescript-language-server", vec!["--stdio"]),
    }
}

/// A human-readable install hint for when the server binary is missing.
pub fn install_hint(lang: Language) -> &'static str {
    match lang {
        Language::Rust => "install rust-analyzer: `cargo install rust-analyzer`",
        Language::Python => "install pyright: `npm install -g pyright`",
        Language::Go => "install gopls: `go install golang.org/x/tools/gopls@latest`",
        Language::TypeScript => {
            "install typescript-language-server: `npm install -g typescript-language-server typescript`"
        }
    }
}

/// Check whether the server binary for `lang` is on `$PATH`.
pub fn server_available(lang: Language) -> bool {
    let (cmd, _) = server_command(lang);
    which(cmd).is_some()
}

/// Minimal `which` — search `$PATH` for an executable. Avoids pulling in a
/// dependency for a one-off check. On Unix the executable bit is required so
/// a non-executable file on `PATH` doesn't get reported as available (which
/// would lead to a confusing spawn failure later).
fn which(cmd: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(cmd);
        if candidate.is_file() && is_executable(&candidate) {
            return Some(candidate);
        }
    }
    None
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    path.metadata()
        .map(|m| m.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn is_executable(_path: &Path) -> bool {
    // Windows doesn't have a Unix-style executable bit; `is_file()` plus the
    // PATHEXT convention is the real check, but this crate is Unix-only in
    // practice (process groups, etc.), so a permissive default is fine here.
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_language_by_extension() {
        assert_eq!(
            detect_language(Path::new("src/main.rs")),
            Some(Language::Rust)
        );
        assert_eq!(detect_language(Path::new("app.py")), Some(Language::Python));
        assert_eq!(detect_language(Path::new("main.go")), Some(Language::Go));
        assert_eq!(
            detect_language(Path::new("app.ts")),
            Some(Language::TypeScript)
        );
        assert_eq!(detect_language(Path::new("README.md")), None);
    }

    #[test]
    fn server_command_returns_binary_name() {
        let (cmd, _) = server_command(Language::Rust);
        assert_eq!(cmd, "rust-analyzer");
        let (cmd, _) = server_command(Language::Python);
        assert_eq!(cmd, "pyright-langserver");
    }

    #[test]
    fn install_hint_mentions_install() {
        assert!(install_hint(Language::Rust).contains("install"));
        assert!(install_hint(Language::Python).contains("install"));
    }
}
