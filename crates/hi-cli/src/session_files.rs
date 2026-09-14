//! Session JSONL path resolution for the Pipe frontend.

use std::path::PathBuf;

use anyhow::Result;

use crate::config::Cli;
use crate::paths;

pub fn resolve_session_path(cli: &Cli) -> Result<Option<PathBuf>> {
    if let Some(path) = &cli.session_file {
        return Ok(Some(path.clone()));
    }
    if cli.no_save {
        return Ok(None);
    }
    if let Some(id) = &cli.resume {
        return Ok(Some(paths::session_path(id)?));
    }
    if cli.cont {
        if let Some(path) = paths::latest_session() {
            return Ok(Some(path));
        }
        eprintln!("\x1b[33mno previous session; starting a new one\x1b[0m");
    }
    Ok(Some(paths::new_session_path()?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Cli;
    use clap::Parser;

    #[test]
    fn session_path_prefers_explicit_file() {
        let cli = Cli::try_parse_from(["hi", "--session-file", "/tmp/explicit.jsonl"]).unwrap();
        let path = resolve_session_path(&cli).unwrap();
        assert_eq!(
            path.as_deref(),
            Some(std::path::Path::new("/tmp/explicit.jsonl"))
        );
    }

    #[test]
    fn session_path_no_save_skips_persistence() {
        let cli = Cli::try_parse_from(["hi", "--no-save"]).unwrap();
        assert_eq!(resolve_session_path(&cli).unwrap(), None);
    }

    #[test]
    fn session_path_resume_id_is_used() {
        let cli = Cli::try_parse_from(["hi", "--resume", "abc-123"]).unwrap();
        let path = resolve_session_path(&cli).unwrap().expect("path");
        assert!(
            path.ends_with("abc-123.jsonl"),
            "unexpected resume path {}",
            path.display()
        );
    }
}
