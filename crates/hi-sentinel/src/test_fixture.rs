//! Isolated Hi-shaped checkout used by repair tests. Never the live user tree.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::fsutil;

const HARNESS_BUG: &str = r#"pub fn invariant_ok() -> bool {
    false
}

#[cfg(test)]
mod tests {
    #[test]
    fn invariant_holds() {
        assert!(crate::invariant_ok(), "injected invariant");
    }
}
"#;

const CLI_MAIN: &str = r#"fn main() {
    if hi_harness::invariant_ok() {
        println!("ok");
        return;
    }
    eprintln!("invariant violated");
    std::process::exit(1);
}
"#;

const PING_LIB: &str = r#"pub fn ping() -> u8 {
    1
}

#[cfg(test)]
mod tests {
    #[test]
    fn ping() {
        assert_eq!(crate::ping(), 1);
    }
}
"#;

/// Git repo that passes `checkout::validate` without a Cargo workspace.
pub fn minimal_checkout(parent: &Path) -> PathBuf {
    let root = parent.join("hi-checkout");
    write_hi_layout(&root, false);
    git_init_commit(&root);
    root
}

/// Git + Cargo workspace with an injected failing invariant in `hi-harness`.
pub fn cargo_checkout(parent: &Path) -> PathBuf {
    let root = parent.join("hi-checkout");
    write_hi_layout(&root, true);
    git_init_commit(&root);
    root
}

fn write_hi_layout(root: &Path, cargo_workspace: bool) {
    fs::create_dir_all(root.join("crates/hi-cli/src")).unwrap();
    fs::create_dir_all(root.join("crates/hi-harness/src")).unwrap();
    fs::create_dir_all(root.join("crates/hi-liveness/src")).unwrap();
    fs::create_dir_all(root.join("crates/hi-sentinel/src")).unwrap();
    fs::write(root.join(".gitignore"), "/target\n/.hi/\n**/.hi/\n").unwrap();
    if cargo_workspace {
        fs::create_dir_all(root.join(".cargo")).unwrap();
        fs::write(root.join(".cargo/config.toml"), "[net]\noffline = true\n").unwrap();
        fs::write(
            root.join("Cargo.toml"),
            r#"[workspace]
resolver = "2"
members = [
    "crates/hi-cli",
    "crates/hi-harness",
    "crates/hi-liveness",
    "crates/hi-sentinel",
]
"#,
        )
        .unwrap();
        write_pkg(
            &root.join("crates/hi-cli"),
            true,
            r#"[package]
name = "hi"
version = "0.0.0"
edition = "2021"

[dependencies]
hi-harness = { path = "../hi-harness" }
"#,
            CLI_MAIN,
        );
        write_pkg(
            &root.join("crates/hi-harness"),
            false,
            &pkg_toml("hi-harness"),
            HARNESS_BUG,
        );
        write_pkg(
            &root.join("crates/hi-liveness"),
            false,
            &pkg_toml("hi-liveness"),
            PING_LIB,
        );
        write_pkg(
            &root.join("crates/hi-sentinel"),
            false,
            &pkg_toml("hi-sentinel"),
            PING_LIB,
        );
        fs::create_dir_all(root.join("skills/autoharnessfix")).unwrap();
        fs::write(
            root.join("skills/autoharnessfix/SKILL.md"),
            "Reproduce first. Do not patch during diagnose. Do not write outside the worktree.\n",
        )
        .unwrap();
    } else {
        fs::write(
            root.join("crates/hi-cli/Cargo.toml"),
            "name = \"hi\"\nversion = \"0.0.0\"\n",
        )
        .unwrap();
        fs::write(
            root.join("crates/hi-harness/Cargo.toml"),
            "name = \"hi-harness\"\n",
        )
        .unwrap();
        fs::write(root.join("crates/hi-harness/src/lib.rs"), "pub fn x() {}\n").unwrap();
        fs::write(root.join("crates/hi-cli/src/main.rs"), "fn main() {}\n").unwrap();
    }
}

fn pkg_toml(name: &str) -> String {
    format!(
        r#"[package]
name = "{name}"
version = "0.0.0"
edition = "2021"
"#
    )
}

fn write_pkg(dir: &Path, bin: bool, toml: &str, src: &str) {
    fs::create_dir_all(dir.join("src")).unwrap();
    fs::write(dir.join("Cargo.toml"), toml).unwrap();
    if bin {
        fs::write(dir.join("src/main.rs"), src).unwrap();
    } else {
        fs::write(dir.join("src/lib.rs"), src).unwrap();
    }
}

fn git_init_commit(root: &Path) {
    let _ = fsutil::mkdir_0700(root);
    run_git(root, &["init", "-b", "main"]);
    run_git(root, &["config", "user.email", "test@test.com"]);
    run_git(root, &["config", "user.name", "Test"]);
    run_git(root, &["config", "commit.gpgsign", "false"]);
    run_git(root, &["add", "-A"]);
    run_git(root, &["commit", "-m", "fixture"]);
}

fn run_git(dir: &Path, args: &[&str]) {
    let output = Command::new("git")
        .args(args)
        .current_dir(dir)
        .env("GIT_AUTHOR_NAME", "Test")
        .env("GIT_AUTHOR_EMAIL", "test@test.com")
        .env("GIT_COMMITTER_NAME", "Test")
        .env("GIT_COMMITTER_EMAIL", "test@test.com")
        .env("GIT_TERMINAL_PROMPT", "0")
        .output()
        .unwrap_or_else(|e| panic!("git {args:?}: {e}"));
    assert!(
        output.status.success(),
        "git {args:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

pub struct StubSpec {
    pub log: PathBuf,
    pub outside: Option<PathBuf>,
}

pub fn write_stub_hi(path: &Path, spec: &StubSpec) {
    let log = spec.log.display();
    let outside = spec
        .outside
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    let script = format!(
        r#"#!/bin/sh
set -eu
log='{log}'
outside='{outside}'
mkdir -p "$log"
n=0
if [ -f "$log/count" ]; then
  n=$(cat "$log/count")
fi
n=$((n+1))
echo "$n" > "$log/count"
{{
  echo "PWD=$(pwd)"
  echo "HI_SANDBOX=${{HI_SANDBOX-}}"
  echo "HI_SENTINEL_ROLE=${{HI_SENTINEL_ROLE-}}"
  echo "HI_SENTINEL_SUPERVISED=${{HI_SENTINEL_SUPERVISED-}}"
  echo "CARGO_HOME=${{CARGO_HOME-}}"
  echo "CARGO_TARGET_DIR=${{CARGO_TARGET_DIR-}}"
  echo "XDG_STATE_HOME=${{XDG_STATE_HOME-}}"
  echo "SSH_AUTH_SOCK=${{SSH_AUTH_SOCK-}}"
  echo "GIT_ASKPASS=${{GIT_ASKPASS-}}"
  printf '%s\n' "$*"
}} > "$log/spawn-$n.env"
printf '%s\n' "$@" > "$log/spawn-$n.argv"
model=""
session=""
review=""
plain=0
nosave=0
confirm=0
prev=""
for arg in "$@"; do
  if [ "$prev" = "--model" ]; then model=$arg; prev=""; continue; fi
  if [ "$prev" = "--session-file" ]; then session=$arg; prev=""; continue; fi
  if [ "$prev" = "--review-target" ]; then review=$arg; prev=""; continue; fi
  case "$arg" in
    --plain) plain=1 ;;
    --no-save) nosave=1 ;;
    --confirm-edits|--confirm-edits=*) confirm=1 ;;
    --model|--session-file|--review-target) prev=$arg ;;
  esac
done
echo "plain=$plain nosave=$nosave confirm=$confirm model=$model session=$session review=$review" >> "$log/spawn-$n.env"
if [ -n "$outside" ]; then
  if [ "${{HI_SANDBOX-}}" = "workspace" ]; then
    echo refused > "$log/outside"
  else
    mkdir -p "$outside"
    echo pwned > "$outside/pwned"
  fi
fi
incident=$(dirname "$session")
if [ ! -f "$incident/diagnosis.md" ]; then
  if [ -f "$log/no_repro" ]; then
    printf 'reproduced: no\nfailing_test:\n' > "$incident/diagnosis.md"
    exit 0
  fi
  printf 'reproduced: yes\nfailing_test: -p hi-harness invariant_holds\nroot_cause: injected invariant\nfiles: crates/hi-harness/src/lib.rs\n' > "$incident/diagnosis.md"
  exit 0
fi
if [ -n "$review" ] && [ -f "$review/crates/hi-harness/src/lib.rs" ]; then
  cat > "$review/crates/hi-harness/src/lib.rs" << 'EOF'
pub fn invariant_ok() -> bool {{
    true
}}

#[cfg(test)]
mod tests {{
    #[test]
    fn invariant_holds() {{
        assert!(crate::invariant_ok(), "injected invariant");
    }}
}}
EOF
fi
exit 0
"#
    );
    fs::write(path, script).unwrap();
    fsutil::chmod_0700_file(path).unwrap();
}

pub fn write_hi_binary_repro(path: &Path) {
    fs::write(
        path,
        r#"#!/usr/bin/env bash
set -euo pipefail
HI_BINARY="${HI_BINARY:?set HI_BINARY to the hi under test}"
exec "$HI_BINARY"
"#,
    )
    .unwrap();
    fsutil::chmod_0700_file(path).unwrap();
}
