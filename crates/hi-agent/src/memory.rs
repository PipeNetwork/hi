//! Project memory: hierarchical location, distillation prompt, capping,
//! groundedness checks, recall-based decay, and concurrency-safe persistence.
//!
//! Memory is **hierarchical** — a project file (`.hi/memory.md`) for facts
//! specific to this codebase, and a global user file (`~/.config/hi/memory.md`)
//! for cross-project preferences that every repo would otherwise relearn from
//! scratch ("use pnpm", "no external API keys", "terse output"). Both are
//! loaded as context; the distiller routes each fact to the right layer.
//!
//! Because two `hi` processes may quit at once in the same directory, writes
//! are serialized with an exclusive lock file (`.hi/.memory.lock`, via
//! `O_EXCL` creation — pure std, no extra dep) and the memory body is written
//! via temp-file + atomic `rename`, so a torn write or a concurrent
//! distillation can never corrupt or truncate the file.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Backstop cap on the distilled memory file. The prompt does the real shaping
/// (≤ ~20 short bullets); this just stops a runaway response from bloating the
/// file — and thus every future session's context.
const MEMORY_MAX_CHARS: usize = 2_000;

/// Schema marker written as the first line of the memory file. Lets a future
/// format change detect an old layout and migrate (or skip) instead of feeding
/// a stale-shape body into a prompt that expects a new one.
const MEMORY_HEADER: &str = "# hi memory v1";

/// Tag the distiller prepends to a fact so the router knows to send it to the
/// global (user-level) layer instead of the project layer.
const GLOBAL_TAG: &str = "global:";

// ---------------------------------------------------------------------------
// Paths — hierarchical (project + global user)
// ---------------------------------------------------------------------------

/// Where the project memory lives — `.hi/memory.md` under the working directory,
/// overridable via `HI_MEMORY_FILE` (which also makes the file IO testable). The
/// frontend reads the same path to load it as context.
pub fn memory_file() -> PathBuf {
    memory_file_at(Path::new("."))
}

/// Workspace-explicit project memory path used by an [`crate::WorkspaceRuntime`].
pub(crate) fn memory_file_at(root: &Path) -> PathBuf {
    std::env::var_os("HI_MEMORY_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join(".hi").join("memory.md"))
}

/// Where the global (cross-project) user memory lives — `$XDG_CONFIG_HOME/hi`
/// or `~/.config/hi/memory.md`. Overridable via `HI_GLOBAL_MEMORY_FILE`. This is
/// the layer for facts that apply to the *user* across every repo, so each new
/// project doesn't relearn "prefer pnpm" or "no external API keys".
pub fn global_memory_file() -> PathBuf {
    std::env::var_os("HI_GLOBAL_MEMORY_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let base = std::env::var_os("XDG_CONFIG_HOME")
                .map(PathBuf::from)
                .or_else(|| {
                    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config"))
                });
            base.unwrap_or_else(|| PathBuf::from(".config"))
                .join("hi")
                .join("memory.md")
        })
}

/// Read a memory file and return its body (schema header stripped). Returns an
/// empty string when the file is missing or unreadable.
fn read_layer(path: &Path) -> String {
    let raw = fs::read_to_string(path).unwrap_or_default();
    strip_header(&raw)
}

/// Read the project memory file (header stripped). Returns "" when absent.
pub fn read_memory() -> String {
    read_layer(&memory_file())
}

/// Read the global (user-level) memory file (header stripped). Returns "" when
/// absent — which is the common case until the distiller has run at least once.
pub fn read_global_memory() -> String {
    read_layer(&global_memory_file())
}

/// A single memory bullet after read-time verification.
#[derive(Debug, Clone)]
pub struct AnnotatedBullet {
    /// The bullet text, cleaned of any prior annotation.
    pub text: String,
    /// `true` when the bullet references a path/command that still resolves in
    /// the current workspace, or the bullet isn't path/command-shaped at all.
    /// `false` when it looks like a path/command but no longer checks out — a
    /// strong stale signal.
    pub verified: bool,
}

impl AnnotatedBullet {
    /// Render the bullet for the system prompt. Stale bullets are marked so the
    /// model treats them skeptically rather than acting on a broken command.
    pub fn render(&self) -> String {
        if self.verified {
            self.text.clone()
        } else {
            format!("{}  ⚠ (may be stale)", self.text)
        }
    }
}

/// Read a memory layer and annotate each bullet with read-time groundedness.
/// Path/command-shaped bullets are checked against the current workspace; a
/// bullet that looks like a build command whose manifest is gone, or a path that
/// no longer exists, is marked as potentially stale. Non-path bullets pass
/// through verified. This is the read-side complement to [`verify_grounded`]
/// (which drops bad bullets at write time); this just warns at read time so a
/// stale fact is visible instead of silently trusted.
pub fn read_annotated(path: &Path) -> Vec<AnnotatedBullet> {
    let body = read_layer(path);
    body.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            // Strip a prior ⚠ marker so we don't accumulate them across reads.
            let cleaned = line.trim().trim_end_matches("⚠ (may be stale)").trim_end();
            let text = cleaned.to_string();
            let verified = match looks_like_path_or_command(&text) {
                Some(token) => is_plausible(&token),
                None => true,
            };
            AnnotatedBullet { text, verified }
        })
        .collect()
}

/// Read and annotate the project memory layer.
pub fn read_project_annotated() -> Vec<AnnotatedBullet> {
    read_annotated(&memory_file())
}

// ---------------------------------------------------------------------------
// Distillation prompt
// ---------------------------------------------------------------------------

/// The session-end distillation prompt, folding in the current memory so the
/// model revises it (merge / de-dupe / drop-stale) instead of appending.
///
/// `corrections` and `recalled` are optional enrichment sections injected into
/// the prompt so the distiller focuses on the highest-signal material.
pub(crate) fn memory_prompt(
    existing: &str,
    global: &str,
    corrections: &str,
    recalled: &str,
) -> String {
    let existing = layer_block("Current project memory", existing);
    let global = layer_block("Current global (user) memory", global);
    let corrections = if corrections.trim().is_empty() {
        String::new()
    } else {
        format!(
            "\nCorrections the user made this session (high-signal):\n---\n{corrections}\n---\n"
        )
    };
    let recalled = if recalled.trim().is_empty() {
        String::new()
    } else {
        format!(
            "\nRecalled memory: these bullets were NOT referenced in this session. They may be stale — consider dropping or merging them if this session supersedes them:\n---\n{recalled}\n---\n"
        )
    };
    format!(
        "This coding session is ending. Maintain a small, durable memory for future work \
         in this project — reusable notes, not a transcript.\n\n{existing}\n\n{global}\n{corrections}{recalled}\n\
         Revise the project memory using only what THIS session actually established: keep facts \
         that save time next time — project conventions, key decisions and constraints, \
         non-obvious gotchas, important file locations, and the exact build/test/run commands. \
         Drop anything transient, already obvious from the code or HI.md, or now outdated. Merge \
         and de-duplicate.\n\nFor facts that are about the USER and apply across ALL projects \
         (editor prefs, preferred package manager, communication style, API-key policy), prefix \
         the bullet with `{GLOBAL_TAG}` — those are routed to the global memory. Everything else \
         stays in project memory.\n\nOutput ONLY the updated project memory as at most ~20 short \
         bullet points (a few words to one line each), `{GLOBAL_TAG}`-prefixed bullets included \
         inline, no preamble. If nothing durable is worth keeping, output the current memory \
         unchanged (or nothing if it was empty)."
    )
}

fn layer_block(label: &str, body: &str) -> String {
    let body = if body.trim().is_empty() {
        "(empty)"
    } else {
        body.trim()
    };
    format!("{label}:\n---\n{body}\n---")
}

// ---------------------------------------------------------------------------
// Correction capture
// ---------------------------------------------------------------------------

/// Extract correction-shaped user messages from the transcript. A correction is
/// a short user message that looks like it's fixing the agent's course — it
/// starts with a negation/repair word, or follows a tool result that looks like
/// an error. These are the highest-signal inputs for memory distillation, and
/// without surfacing them explicitly the end-of-session model may miss them in
/// a long transcript.
pub(crate) fn extract_corrections(messages: &[Message]) -> String {
    let mut out = Vec::new();
    for (i, msg) in messages.iter().enumerate() {
        if msg.role != Role::User {
            continue;
        }
        let raw = msg.text();
        let text = raw.trim();
        // Slash commands and pasted code aren't corrections.
        if text.starts_with('/') || text.lines().count() > 3 {
            continue;
        }
        if looks_like_correction(text)
            || (i > 0 && prev_was_error(&messages[i.saturating_sub(1)]) && !is_tool_result(msg))
        {
            // Strip paste-folded stdin markers that clutter the snippet.
            let clean = text.replace("\n<<<STDIN>>>:", " ");
            out.push(format!("- {clean}"));
        }
    }
    out.join("\n")
}

fn looks_like_correction(text: &str) -> bool {
    // Lead with a negation / repair marker. Match on the first word so "no, use
    // pnpm" / "actually it's src/main.rs" both qualify.
    let lower = text.to_ascii_lowercase();
    let lead = lower
        .split_whitespace()
        .take(2)
        .collect::<Vec<_>>()
        .join(" ");
    const MARKERS: &[&str] = &[
        "no",
        "no,",
        "no.",
        "not",
        "wrong",
        "actually",
        "wait",
        "stop",
        "don't",
        "dont",
        "instead",
        "use",
        "should",
        "needs",
        "missing",
        "incorrect",
        "typo",
        "prefer",
        "i prefer",
        "my preference",
        "please always",
    ];
    MARKERS
        .iter()
        .any(|m| lower.starts_with(m) || lead.starts_with(m))
}

fn prev_was_error(msg: &Message) -> bool {
    // A tool result role carrying an error-ish payload.
    msg.role == Role::Tool && msg.text().to_ascii_lowercase().contains("error")
}

fn is_tool_result(msg: &Message) -> bool {
    msg.role == Role::Tool
}

// ---------------------------------------------------------------------------
// Recall signal — decay unused bullets
// ---------------------------------------------------------------------------

/// Compute which existing memory bullets were NOT referenced anywhere in this
/// session's transcript (case-insensitive substring match on a distinguishing
/// fragment of each bullet). These are recall candidates: facts the session
/// succeeded without, so they're decay/merge candidates.
///
/// Returns the bullet lines joined by newlines (without the leading `- ` stripped)
/// so the prompt can present them as a list. Only bullets long enough to be
/// distinguishable are considered — very short tokens produce false positives.
pub(crate) fn unreferenced_bullets(existing: &str, transcript: &str) -> String {
    let haystack = transcript.to_ascii_lowercase();
    let mut out = Vec::new();
    for line in existing.lines() {
        let bullet = line.trim_start_matches('-').trim();
        if bullet.is_empty() {
            continue;
        }
        // Use a distinguishing fragment: the longest whitespace-delimited token.
        // This avoids matching on common lead-ins ("the", "use", "project").
        let frag = distinguishing_fragment(bullet);
        if frag.len() < 4 {
            continue; // too generic to trust a non-match on
        }
        if !haystack.contains(&frag) {
            out.push(format!("- {bullet}"));
        }
    }
    out.join("\n")
}

/// Pick the longest alphanumeric token in the bullet as the match key.
fn distinguishing_fragment(bullet: &str) -> String {
    bullet
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric() && c != '/' && c != '-' && c != '_'))
        .filter(|w| w.len() >= 4)
        .max_by_key(|w| w.len())
        .unwrap_or("")
        .to_ascii_lowercase()
}

// ---------------------------------------------------------------------------
// Groundedness — verify path/command bullets against the filesystem
// ---------------------------------------------------------------------------

/// A bullet whose leading term looks like a filesystem path (contains `/` or a
/// known extension) or a build command. We verify these against the workspace
/// after distillation so memory doesn't persist a hallucinated path or a build
/// command that no longer works.
fn looks_like_path_or_command(bullet: &str) -> Option<String> {
    let body = bullet.trim_start_matches('-').trim();
    if body.is_empty() {
        return None;
    }
    // A command-shaped bullet: starts with a known runner.
    let lower = body.to_ascii_lowercase();
    const COMMANDS: &[&str] = &[
        "cargo", "npm", "pnpm", "yarn", "make", "go ", "python", "python3", "pytest", "ruff",
        "just ", "bun ", "deno", "tsc", "eslint", "prettier", "gradle", "mvn", "cmake", "bash",
    ];
    if COMMANDS.iter().any(|c| lower.starts_with(c)) {
        return Some(body.to_string());
    }
    // A path-shaped bullet: scan every whitespace-delimited token for one that
    // looks like a path (contains `/` or a file extension). Memory prose like
    // "entry is src/main.rs" puts the path mid-sentence, so the first word
    // alone isn't enough.
    for word in body.split_whitespace() {
        let word =
            word.trim_matches(|c: char| !c.is_alphanumeric() && c != '/' && c != '.' && c != '-');
        // A trailing '.' is sentence punctuation, not part of a path: without
        // this, "src/main.rs." never exists and any prose word ending a
        // sentence ("messages.") reads as having an (empty) extension — either
        // way the bullet gets dropped as an implausible path.
        let word = word.trim_end_matches('.');
        if word.contains('/') || has_extension(word) {
            return Some(word.to_string());
        }
    }
    None
}

fn has_extension(s: &str) -> bool {
    Path::new(s).extension().is_some_and(|e| {
        let e = e.to_string_lossy();
        // Real file extensions (rs, py, json, mp4, …) are short alphanumerics
        // with at least one letter. This keeps version numbers ("v1.2") and
        // abbreviations from classifying prose as a path to be verified.
        !e.is_empty()
            && e.len() <= 6
            && e.chars().all(|c| c.is_ascii_alphanumeric())
            && e.chars().any(|c| c.is_ascii_alphabetic())
    })
}

/// Verify distilled bullets that look like paths or commands against the current
/// workspace. Path-shaped bullets are checked for existence; command-shaped
/// bullets are checked for the presence of the relevant manifest file (e.g.
/// `cargo` → Cargo.toml, `pnpm` → pnpm-lock.yaml or package.json). Bullets that
/// fail verification are dropped — a hallucinated path or a stale build command
/// is worse than no memory.
///
/// This is best-effort and conservative: when in doubt we keep the bullet. The
/// cost of a false drop (losing a real fact) is higher than a false keep.
pub(crate) fn verify_grounded(bullets: &str) -> String {
    bullets
        .lines()
        .filter(|line| {
            let line = line.trim();
            if line.is_empty() {
                return true; // preserve blank separators
            }
            // Don't verify global-routed bullets here — they're extracted before
            // this runs and may reference paths in other projects.
            if line.trim_start_matches('-').trim().starts_with(GLOBAL_TAG) {
                return true;
            }
            match looks_like_path_or_command(line) {
                Some(token) => is_plausible(&token),
                None => true, // not path/command-shaped → keep
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Is this token a plausible path or command in the current workspace?
fn is_plausible(token: &str) -> bool {
    let lower = token.to_ascii_lowercase();
    // Command → require the corresponding manifest to exist.
    let manifest = if lower.starts_with("cargo") {
        Some("Cargo.toml")
    } else if lower.starts_with("go ") || lower.starts_with("go\t") || lower == "go" {
        Some("go.mod")
    } else if lower.starts_with("pnpm") {
        // pnpm is fine with either lock file or a plain package.json.
        return Path::new("pnpm-lock.yaml").exists()
            || Path::new("package.json").exists()
            || Path::new("npm-shrinkwrap.json").exists();
    } else if lower.starts_with("npm") {
        return Path::new("package.json").exists() || Path::new("package-lock.json").exists();
    } else if lower.starts_with("yarn") {
        return Path::new("package.json").exists() || Path::new("yarn.lock").exists();
    } else if lower.starts_with("bun ") || lower == "bun" {
        return Path::new("package.json").exists()
            || Path::new("bun.lockb").exists()
            || Path::new("bun.lock").exists();
    } else if lower.starts_with("make") || lower == "make" {
        return Path::new("Makefile").exists() || Path::new("makefile").exists();
    } else if lower.starts_with("just") {
        return Path::new("justfile").exists() || Path::new("Justfile").exists();
    } else if lower.starts_with("pytest")
        || lower.starts_with("python")
        || lower.starts_with("ruff")
    {
        return Path::new("pyproject.toml").exists()
            || Path::new("setup.py").exists()
            || Path::new("requirements.txt").exists();
    } else if lower.starts_with("tsc") {
        return Path::new("tsconfig.json").exists() || Path::new("package.json").exists();
    } else if lower.starts_with("gradle") {
        return Path::new("build.gradle").exists()
            || Path::new("build.gradle.kts").exists()
            || Path::new("settings.gradle").exists();
    } else if lower.starts_with("mvn") {
        return Path::new("pom.xml").exists();
    } else if lower.starts_with("cmake") {
        return Path::new("CMakeLists.txt").exists();
    } else if lower.starts_with("deno") {
        return Path::new("deno.json").exists() || Path::new("deno.jsonc").exists();
    } else {
        None
    };
    if let Some(mf) = manifest {
        return Path::new(mf).exists();
    }
    // Path-shaped → check existence directly. A leading "build/" or "src/"
    // fragment is checked as-is; an absolute path is checked literally.
    if token.starts_with('/') {
        return Path::new(token).exists();
    }
    // For a token like "src/main.rs" or "crates/foo", check the path itself.
    if token.contains('.') {
        return Path::new(token).exists();
    }
    // A directory-ish fragment: check if it exists as a dir.
    Path::new(token).exists()
}

// ---------------------------------------------------------------------------
// Split / route — separate global-routed bullets from project bullets
// ---------------------------------------------------------------------------

/// Split distilled bullets into (project, global) by the `{GLOBAL_TAG}` prefix.
/// Strips the tag from the global lines so the stored file is clean markdown.
pub(crate) fn split_layers(distilled: &str) -> (String, String) {
    let mut project = Vec::new();
    let mut global = Vec::new();
    for line in distilled.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix(GLOBAL_TAG) {
            global.push(format!("- {}", rest.trim()));
        } else {
            project.push(trimmed.to_string());
        }
    }
    (project.join("\n"), global.join("\n"))
}

// ---------------------------------------------------------------------------
// Capping + gating
// ---------------------------------------------------------------------------

/// Trim and cap the distilled memory at [`MEMORY_MAX_CHARS`], cutting back to the
/// last whole line so a bullet isn't sliced mid-word. Empty in → empty out.
pub(crate) fn cap_memory(s: &str) -> String {
    let s = s.trim();
    if s.chars().count() <= MEMORY_MAX_CHARS {
        return s.to_string();
    }
    let kept: String = s.chars().take(MEMORY_MAX_CHARS).collect();
    let kept = kept
        .rsplit_once('\n')
        .map(|(head, _)| head)
        .unwrap_or(&kept);
    format!("{}\n… (memory truncated)", kept.trim_end())
}

/// Whether to distill session memory at quit: only when enabled *and* the model
/// actually produced output this session, so an empty or command-only session
/// writes nothing. Shared by both frontends so the rule can't drift between them.
pub fn should_distill_memory(enabled: bool, output_tokens: u64) -> bool {
    enabled && output_tokens > 0
}

// ---------------------------------------------------------------------------
// Header strip + atomic/locked write (unchanged behaviour, now public helpers)
// ---------------------------------------------------------------------------

/// Strip the schema header line from a read-back memory body, so the distiller
/// and the context loader see only the bullets. Tolerates a missing header
/// (e.g. a hand-written file or a pre-versioning file migrated in) by returning
/// the body unchanged.
pub(crate) fn strip_header(raw: &str) -> String {
    let mut lines = raw.lines();
    match lines.next() {
        Some(first) if first.trim() == MEMORY_HEADER => lines.collect::<Vec<_>>().join("\n"),
        _ => raw.trim().to_string(),
    }
}

/// Atomically and exclusively write the distilled memory.
///
/// - **Exclusive**: takes `.memory.lock` via `O_EXCL` creation. If another
///   process holds it, this distillation is skipped (best-effort at quit — the
///   other writer's revision wins). The lock is an advisory mutex scoped to the
///   `write_memory` call via a guard that deletes it on drop.
/// - **Atomic**: writes to `<file>.tmp` then `rename`s over the target, so a
///   crash mid-write leaves the previous file intact rather than a truncated one.
/// - **Versioned**: prepends the [`MEMORY_HEADER`] schema marker.
///
/// Returns `Ok(notes)` with the count of non-empty lines written, or `Err` with
/// a human-readable status string for the UI.
pub(crate) fn write_memory(path: &Path, body: &str) -> Result<usize, String> {
    let body = body.trim();
    if body.is_empty() {
        return Ok(0);
    }
    let notes = body.lines().filter(|l| !l.trim().is_empty()).count();

    let Some(parent) = path.parent() else {
        return Err(format!("no parent directory for {}", path.display()));
    };
    if let Err(e) = fs::create_dir_all(parent) {
        return Err(format!("couldn't create {}: {e}", parent.display()));
    }

    // Exclusive lock — serialized across concurrent processes in this dir.
    let _lock = take_lock(parent)?;

    // Atomic publish: temp file + rename. fs::rename is atomic on POSIX when
    // source and destination are on the same filesystem (they are — both in
    // the memory file's parent dir). The temp name is PID-scoped (mirroring
    // `write_skill`): even if two processes ever hold the lock at once (the
    // stale-lock break is racy), distinct temp files mean each rename installs a
    // *complete* file — last-writer-wins, never the torn/empty result a shared
    // `memory.md.tmp` produced when one writer truncated it mid-rename.
    let mut tmp = path.to_path_buf();
    let mut name = tmp
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_else(|| std::ffi::OsString::from("memory.md"));
    name.push(format!(".{}.tmp", std::process::id()));
    tmp.set_file_name(name);

    let content = format!("{MEMORY_HEADER}\n{body}\n");
    if let Err(e) = write_tmp(&tmp, &content) {
        let _ = fs::remove_file(&tmp);
        return Err(format!("couldn't write {}: {e}", tmp.display()));
    }
    if let Err(e) = fs::rename(&tmp, path) {
        let _ = fs::remove_file(&tmp);
        return Err(format!("couldn't install {}: {e}", path.display()));
    }
    Ok(notes)
}

fn write_tmp(tmp: &Path, content: &str) -> std::io::Result<()> {
    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(tmp)?;
    f.write_all(content.as_bytes())?;
    f.sync_all()?;
    Ok(())
}

/// A dropped-on-release exclusive lock acquired via `O_EXCL` file creation.
struct LockGuard(PathBuf);
impl Drop for LockGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// A lock older than this is presumed abandoned (the holder crashed or was
/// killed before its `Drop` ran). Memory writes are sub-second, so minutes of
/// age means no live holder — without this, one SIGKILL would disable memory
/// persistence for every future session until the file is deleted by hand.
const LOCK_STALE_AFTER: std::time::Duration = std::time::Duration::from_secs(10 * 60);

fn take_lock(parent: &Path) -> Result<LockGuard, String> {
    let lock = parent.join(".memory.lock");
    for _ in 0..2 {
        match OpenOptions::new().write(true).create_new(true).open(&lock) {
            Ok(_) => return Ok(LockGuard(lock)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                let stale = fs::metadata(&lock)
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| t.elapsed().ok())
                    .is_some_and(|age| age > LOCK_STALE_AFTER);
                if !stale {
                    return Err("another session is updating memory".to_string());
                }
                // Break the stale lock and retry the exclusive create once (a
                // concurrent session may break it first — losing that race
                // lands in AlreadyExists again and errors out normally).
                let _ = fs::remove_file(&lock);
            }
            Err(e) => return Err(format!("couldn't take memory lock: {e}")),
        }
    }
    Err("another session is updating memory".to_string())
}

// Forward the message types so callers don't need a separate import.
pub(crate) use hi_ai::{Message, Role};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cap_memory_trims_and_backstops() {
        assert_eq!(cap_memory("  - a\n- b  "), "- a\n- b"); // trimmed, under budget
        assert_eq!(cap_memory("   "), ""); // empty in → empty out
        let big = "- a durable note\n".repeat(1000); // ≫ MEMORY_MAX_CHARS
        let capped = cap_memory(&big);
        assert!(
            capped.chars().count() <= MEMORY_MAX_CHARS + 40,
            "backstopped"
        );
        assert!(capped.ends_with("(memory truncated)"));
    }

    #[test]
    fn memory_prompt_folds_in_existing_memory() {
        let p = memory_prompt("- 4-space indent", "", "", "");
        assert!(p.contains("- 4-space indent"), "includes current memory");
        assert!(p.contains("Current project memory"));
        assert!(
            memory_prompt("   ", "   ", "   ", "   ").contains("(empty)"),
            "blank → (empty)"
        );
        // Corrections and recalled sections only appear when non-empty.
        let with = memory_prompt("", "", "- no, use pnpm", "");
        assert!(with.contains("Corrections the user made"));
        assert!(!with.contains("Recalled memory"));
        let rec = memory_prompt("", "", "", "- old fact");
        assert!(rec.contains("Recalled memory"));
    }

    #[test]
    fn should_distill_memory_gates_on_enabled_and_work() {
        assert!(should_distill_memory(true, 1), "enabled + work → distill");
        assert!(!should_distill_memory(true, 0), "no model output → skip");
        assert!(!should_distill_memory(false, 100), "disabled → skip");
    }

    #[test]
    fn strip_header_removes_version_marker() {
        let raw = format!("{MEMORY_HEADER}\n- note one\n- note two");
        assert_eq!(strip_header(&raw), "- note one\n- note two");
    }

    #[test]
    fn strip_header_passes_through_unversioned_body() {
        // A hand-written or pre-versioning file has no header — keep it intact.
        assert_eq!(strip_header("- legacy\n- notes"), "- legacy\n- notes");
        assert_eq!(strip_header(""), "");
    }

    /// Round-trip: write a body, read it back, strip the header, get the body.
    #[test]
    fn write_memory_round_trips_with_header() {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "hi-mem-rt-{}-{}.md",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let notes = write_memory(&path, "- alpha\n- beta").unwrap();
        assert_eq!(notes, 2);
        let raw = fs::read_to_string(&path).unwrap();
        assert!(raw.starts_with(MEMORY_HEADER), "header written");
        assert_eq!(strip_header(&raw), "- alpha\n- beta");
        let _ = fs::remove_file(&path);
        // Lock sibling cleaned up.
        let _ = fs::remove_file(path.with_file_name(".memory.lock"));
    }

    /// An empty body writes nothing and creates no file.
    #[test]
    fn write_memory_skips_empty_body() {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "hi-mem-empty-{}-{}.md",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        assert_eq!(write_memory(&path, "   \n  ").unwrap(), 0);
        assert!(!path.exists(), "no file created for empty body");
    }

    /// A second concurrent lock attempt is rejected while the first is held.
    #[test]
    fn concurrent_lock_is_exclusive() {
        let dir = std::env::temp_dir().join(format!(
            "hi-mem-lock-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let lock_a = take_lock(&dir);
        assert!(lock_a.is_ok(), "first taker succeeds");
        let lock_b = take_lock(&dir);
        assert!(lock_b.is_err(), "second taker rejected while first holds");
        drop(lock_a); // releases — Now it's free again.
        assert!(take_lock(&dir).is_ok(), "re-acquirable after release");
        let _ = fs::remove_dir_all(&dir);
    }

    // --- hierarchical routing ---

    #[test]
    fn split_layers_routes_global_tagged_bullets() {
        let distilled = "- build with cargo\nglobal: prefers pnpm\nglobal: no external API keys\n- tests in tests/";
        let (project, global) = split_layers(distilled);
        assert_eq!(project, "- build with cargo\n- tests in tests/");
        assert_eq!(global, "- prefers pnpm\n- no external API keys");
    }

    #[test]
    fn split_layers_all_project_when_no_global_tag() {
        let (project, global) = split_layers("- a\n- b");
        assert_eq!(project, "- a\n- b");
        assert!(global.is_empty());
    }

    // --- correction capture ---

    #[test]
    fn extract_corrections_catches_negation_and_repair_words() {
        let msgs = vec![
            Message::user("build the landing page"),
            Message::user("no, use pnpm not npm"),
            Message::user("actually the entry is src/main.rs"),
            Message::user("here is a long paste\nof code\nwith many lines\nmore"),
        ];
        let c = extract_corrections(&msgs);
        assert!(c.contains("no, use pnpm not npm"), "catches negation: {c}");
        assert!(
            c.contains("actually the entry is src/main.rs"),
            "catches repair: {c}"
        );
        // The long paste is excluded.
        assert!(!c.contains("here is a long paste"));
        // The plain instruction is excluded.
        assert!(!c.contains("build the landing page"));
    }

    #[test]
    fn extract_corrections_skips_slash_commands_and_empty() {
        let msgs = vec![Message::user("/compact"), Message::user("")];
        let c = extract_corrections(&msgs);
        assert!(c.is_empty(), "slash commands and empties ignored: {c}");
    }

    // --- recall / decay ---

    #[test]
    fn unreferenced_bullets_flags_unused_facts() {
        let existing = "- build with cargo\n- deploy uses kustomize\n- 4-space indent";
        // Transcript mentions cargo but not kustomize or indent.
        let transcript = "we ran cargo build and it worked";
        let unref = unreferenced_bullets(existing, transcript);
        assert!(unref.contains("kustomize"), "flags unused: {unref}");
        // "cargo" was referenced, so that bullet is NOT flagged.
        assert!(!unref.contains("build with cargo"));
    }

    #[test]
    fn unreferenced_bullets_ignores_too_short_fragments() {
        // Short tokens are too generic — don't trust a non-match.
        let existing = "- use npx\n";
        let unref = unreferenced_bullets(existing, "totally unrelated text");
        // "npx" is len 3 < 4, so it's skipped (not flagged).
        assert!(!unref.contains("npx"));
    }

    // --- groundedness ---

    #[test]
    fn verify_grounded_drops_missing_paths() {
        // This path almost certainly doesn't exist relative to cwd.
        let bullets = "- entry is src/nonexistent_main_xyz.rs\n- a normal note";
        let kept = verify_grounded(bullets);
        assert!(!kept.contains("nonexistent"), "drops missing path: {kept}");
        assert!(kept.contains("a normal note"), "keeps non-path bullets");
    }

    #[test]
    fn verify_grounded_keeps_manifest_backed_commands() {
        // cargo → Cargo.toml, which exists in this repo.
        let bullets = "- build with cargo\n- entry is src/nonexistent_xyz.rs";
        let kept = verify_grounded(bullets);
        assert!(kept.contains("cargo"), "keeps cargo (Cargo.toml present)");
        assert!(!kept.contains("nonexistent"), "drops missing path");
    }

    #[test]
    fn verify_grounded_passes_through_non_command_bullets() {
        let bullets = "- prefers terse output\n- decisions in ADR format";
        let kept = verify_grounded(bullets);
        assert_eq!(kept.trim(), bullets.trim(), "non-path/command bullets kept");
    }

    #[test]
    fn verify_grounded_skips_global_tagged_bullets() {
        // Global bullets may reference paths in other projects — don't verify.
        let bullets = "global: uses /home/david/bin/custom-tool";
        let kept = verify_grounded(bullets);
        assert!(kept.contains("global:"), "global bullets not verified");
    }

    // --- stale-on-read annotation ---

    #[test]
    fn read_annotated_marks_missing_paths_stale() {
        let dir = std::env::temp_dir().join(format!(
            "hi-mem-ann-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("memory.md");
        write_memory(
            &path,
            "- entry is src/totally_missing_xyz.rs\n- prefers terse output",
        )
        .unwrap();
        let annotated = read_annotated(&path);
        assert_eq!(annotated.len(), 2);
        // The missing path → stale.
        let stale = annotated
            .iter()
            .find(|a| a.text.contains("missing"))
            .unwrap();
        assert!(!stale.verified, "missing path marked stale");
        assert!(
            stale.render().contains("may be stale"),
            "rendered with warning"
        );
        // The plain bullet → verified.
        let ok = annotated.iter().find(|a| a.text.contains("terse")).unwrap();
        assert!(ok.verified, "non-path bullet verified");
        assert_eq!(ok.render(), ok.text, "no warning appended");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_annotated_strips_prior_warning_marker() {
        // A bullet that already carries the marker shouldn't accumulate a second.
        let dir = std::env::temp_dir().join(format!(
            "hi-mem-strip-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("memory.md");
        write_memory(&path, "- entry is src/missing_again.rs  ⚠ (may be stale)").unwrap();
        let annotated = read_annotated(&path);
        assert_eq!(annotated.len(), 1);
        // The stored text shouldn't double-up the marker.
        assert_eq!(
            annotated[0].text.matches("may be stale").count(),
            0,
            "marker stripped from stored text"
        );
        // But it's re-added on render since it's still stale.
        assert!(annotated[0].render().contains("may be stale"));
        let _ = fs::remove_dir_all(&dir);
    }
}
