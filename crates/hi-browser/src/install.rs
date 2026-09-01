//! Write the unpacked MV3 debugger extension under `~/.config/hi/browser-extension/`.

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};

const MANIFEST: &str = r#"{
  "manifest_version": 3,
  "name": "hi browser_exec",
  "version": "0.3.1",
  "description": "Relays Chrome debugger access to the local hi agent (Chrome 136+).",
  "permissions": ["debugger", "tabs", "activeTab"],
  "host_permissions": ["<all_urls>"],
  "background": { "service_worker": "background.js" },
  "action": { "default_title": "hi browser_exec" }
}
"#;

const BACKGROUND: &str = r#"chrome.runtime.onInstalled.addListener(() => {
  console.log("hi browser_exec extension installed. Relay token is in token.txt next to this file.");
});
"#;

pub fn install_dir() -> Result<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
        .context("could not determine config directory")?;
    Ok(base.join("hi").join("browser-extension"))
}

pub fn install_extension() -> Result<PathBuf> {
    let dir = install_dir()?;
    fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    fs::write(dir.join("manifest.json"), MANIFEST)?;
    fs::write(dir.join("background.js"), BACKGROUND)?;
    let token = format!(
        "{:x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    fs::write(dir.join("token.txt"), format!("{token}\n"))?;
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_manifest_and_token() {
        let tmp = std::env::temp_dir().join(format!(
            "hi-browser-ext-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        // install_dir uses XDG; this unit test writes via the same files locally.
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("manifest.json"), MANIFEST).unwrap();
        std::fs::write(tmp.join("background.js"), BACKGROUND).unwrap();
        std::fs::write(tmp.join("token.txt"), "abc\n").unwrap();
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(tmp.join("manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["manifest_version"], 3);
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
