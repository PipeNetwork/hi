//! Mutation-then-command fusion: one combined observation, no middle model round.
//!
//! The mutation runs first. A failed mutation skips the command. A failed
//! command keeps the mutation. Two fused mutations of the same file cannot
//! interleave between mutate and command because both steps hold one path lock.

use std::path::Path;

use crate::file_operation_lock::FileOperationLockManager;

pub const FUSED_COMMAND_SUCCEEDED: &str = "[fused_command:succeeded]";
pub const FUSED_COMMAND_FAILED: &str = "[fused_command:failed]";
pub const FUSED_COMMAND_SKIPPED: &str = "[fused_command:skipped]";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MutationStep {
    pub ok: bool,
    pub output: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommandStep {
    pub ok: bool,
    pub output: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FusionObservation {
    pub mutation_output: String,
    pub mutation_ok: bool,
    pub command_ran: bool,
    pub command_ok: Option<bool>,
    pub command_output: Option<String>,
}

impl FusionObservation {
    pub fn combined(&self) -> String {
        match (&self.command_output, self.command_ran, self.command_ok) {
            (Some(command), true, Some(true)) => {
                format!(
                    "{}\n\n{FUSED_COMMAND_SUCCEEDED}\n{command}",
                    self.mutation_output
                )
            }
            (Some(command), true, _) => {
                // Lead with the failure. A "Wrote N bytes" prefix plus a buried
                // cargo-check error reads as success to small models and they
                // stop, after which turn-end verify stalls the session.
                format!(
                    "{FUSED_COMMAND_FAILED}\n{command}\n\n{}",
                    self.mutation_output
                )
            }
            _ if !self.mutation_ok => {
                format!(
                    "{FUSED_COMMAND_SKIPPED} The file mutation did not complete successfully; the command was not run.\n\n{}",
                    self.mutation_output
                )
            }
            _ => self.mutation_output.clone(),
        }
    }
}

/// Run `mutate`, then optionally `command`, while holding an exclusive path lock.
pub async fn execute_mutation_then_command<Mut, MutFut, Cmd, CmdFut>(
    lock: &FileOperationLockManager,
    root: &Path,
    path: &str,
    mutate: Mut,
    command: Option<Cmd>,
) -> FusionObservation
where
    Mut: FnOnce() -> MutFut,
    MutFut: std::future::Future<Output = MutationStep>,
    Cmd: FnOnce() -> CmdFut,
    CmdFut: std::future::Future<Output = CommandStep>,
{
    lock.with_path_lock(root, path, || async move {
        let mutation = mutate().await;
        if command.is_none() {
            return FusionObservation {
                mutation_output: mutation.output,
                mutation_ok: mutation.ok,
                command_ran: false,
                command_ok: None,
                command_output: None,
            };
        }
        if !mutation.ok {
            return FusionObservation {
                mutation_output: mutation.output,
                mutation_ok: false,
                command_ran: false,
                command_ok: None,
                command_output: None,
            };
        }
        let command = (command.expect("checked is_some"))().await;
        FusionObservation {
            mutation_output: mutation.output,
            mutation_ok: true,
            command_ran: true,
            command_ok: Some(command.ok),
            command_output: Some(command.output),
        }
    })
    .await
}

pub fn is_filesystem_mutation_tool(name: &str) -> bool {
    matches!(name, "write" | "edit" | "multi_edit" | "apply_patch")
}

/// Consecutive mutation then bash in emission order, with bash depending only
/// on already-completed calls plus that mutation.
pub fn fused_command_index(
    names: &[&str],
    mutation_index: usize,
    completed: &[bool],
    deps: &[Vec<usize>],
) -> Option<usize> {
    let command_index = mutation_index.checked_add(1)?;
    if command_index >= names.len() {
        return None;
    }
    if completed.get(command_index).copied().unwrap_or(true) {
        return None;
    }
    if !is_filesystem_mutation_tool(names[mutation_index]) {
        return None;
    }
    if names[command_index] != "bash" {
        return None;
    }
    let command_deps = deps.get(command_index)?;
    if command_deps
        .iter()
        .all(|&dep| dep == mutation_index || completed.get(dep).copied().unwrap_or(false))
    {
        Some(command_index)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    #[test]
    fn fused_command_index_pairs_consecutive_write_then_unready_bash() {
        let names = ["write", "bash"];
        let completed = [false, false];
        let deps = [vec![], vec![0]];
        assert_eq!(
            fused_command_index(&names, 0, &completed, &deps),
            Some(1),
            "bash is typically not in ready because it depends on the write"
        );
    }

    #[test]
    fn fused_command_index_accepts_edit_and_apply_patch() {
        let completed = [false, false];
        let deps = [vec![], vec![0]];
        assert_eq!(
            fused_command_index(&["edit", "bash"], 0, &completed, &deps),
            Some(1)
        );
        assert_eq!(
            fused_command_index(&["multi_edit", "bash"], 0, &completed, &deps),
            Some(1)
        );
        assert_eq!(
            fused_command_index(&["apply_patch", "bash"], 0, &completed, &deps),
            Some(1)
        );
    }

    #[test]
    fn fused_command_index_requires_consecutive_mutation_then_bash() {
        let completed = [false, false, false];
        let deps = [vec![], vec![], vec![0]];
        assert_eq!(
            fused_command_index(&["write", "read", "bash"], 0, &completed, &deps),
            None
        );
        assert_eq!(
            fused_command_index(&["read", "bash"], 0, &[false, false], &[vec![], vec![0]]),
            None
        );
        assert_eq!(
            fused_command_index(&["write", "read"], 0, &[false, false], &[vec![], vec![0]]),
            None
        );
        assert_eq!(
            fused_command_index(&["write"], 0, &[false], &[vec![]]),
            None
        );
    }

    #[test]
    fn fused_command_index_skips_completed_bash_or_unmet_foreign_deps() {
        let names = ["write", "bash"];
        assert_eq!(
            fused_command_index(&names, 0, &[false, true], &[vec![], vec![0]]),
            None
        );
        assert_eq!(
            fused_command_index(&names, 0, &[false, false], &[vec![], vec![0, 9]]),
            None
        );
        assert_eq!(
            fused_command_index(
                &["read", "write", "bash"],
                1,
                &[true, false, false],
                &[vec![], vec![], vec![0, 1]]
            ),
            Some(2)
        );
        let two_pairs = ["write", "bash", "write", "bash"];
        let completed = [false, false, false, false];
        let deps = [vec![], vec![0], vec![0], vec![2]];
        assert_eq!(
            fused_command_index(&two_pairs, 0, &completed, &deps),
            Some(1)
        );
        assert_eq!(
            fused_command_index(&two_pairs, 2, &completed, &deps),
            Some(3)
        );
    }

    #[tokio::test]
    async fn success_is_one_combined_observation() {
        let lock = FileOperationLockManager::new();
        let dir = tempfile::tempdir().unwrap();
        let ran = Arc::new(AtomicUsize::new(0));
        let ran_command = ran.clone();
        let observation = execute_mutation_then_command(
            &lock,
            dir.path(),
            "a.rs",
            || async {
                MutationStep {
                    ok: true,
                    output: "edited".into(),
                }
            },
            Some(|| async move {
                ran_command.fetch_add(1, Ordering::SeqCst);
                CommandStep {
                    ok: true,
                    output: "tests passed".into(),
                }
            }),
        )
        .await;
        assert!(observation.mutation_ok);
        assert!(observation.command_ran);
        assert_eq!(observation.command_ok, Some(true));
        let combined = observation.combined();
        assert!(combined.contains("edited"), "{combined}");
        assert!(combined.contains(FUSED_COMMAND_SUCCEEDED), "{combined}");
        assert!(combined.contains("tests passed"), "{combined}");
        assert_eq!(ran.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn mutation_failure_skips_command() {
        let lock = FileOperationLockManager::new();
        let dir = tempfile::tempdir().unwrap();
        let ran = Arc::new(AtomicUsize::new(0));
        let ran_command = ran.clone();
        let observation = execute_mutation_then_command(
            &lock,
            dir.path(),
            "a.rs",
            || async {
                MutationStep {
                    ok: false,
                    output: "edit failed".into(),
                }
            },
            Some(|| async move {
                ran_command.fetch_add(1, Ordering::SeqCst);
                CommandStep {
                    ok: true,
                    output: "should not run".into(),
                }
            }),
        )
        .await;
        assert!(!observation.mutation_ok);
        assert!(!observation.command_ran);
        assert!(observation.combined().contains(FUSED_COMMAND_SKIPPED));
        assert!(!observation.combined().contains("should not run"));
        assert_eq!(ran.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn command_failure_keeps_mutation() {
        let lock = FileOperationLockManager::new();
        let dir = tempfile::tempdir().unwrap();
        let observation = execute_mutation_then_command(
            &lock,
            dir.path(),
            "a.rs",
            || async {
                MutationStep {
                    ok: true,
                    output: "edited".into(),
                }
            },
            Some(|| async {
                CommandStep {
                    ok: false,
                    output: "test failed".into(),
                }
            }),
        )
        .await;
        let combined = observation.combined();
        assert!(observation.mutation_ok);
        assert_eq!(observation.command_ok, Some(false));
        assert!(combined.contains("edited"), "{combined}");
        assert!(combined.contains(FUSED_COMMAND_FAILED), "{combined}");
        assert!(combined.contains("test failed"), "{combined}");
        assert!(
            combined.find(FUSED_COMMAND_FAILED).unwrap() < combined.find("edited").unwrap(),
            "failed fused command must lead the mutation observation: {combined}"
        );
    }

    #[tokio::test]
    async fn second_fused_mutation_cannot_observe_in_between_state() {
        let lock = FileOperationLockManager::new();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.rs");
        std::fs::write(&path, "start").unwrap();
        let seen_mid = Arc::new(AtomicUsize::new(0));
        let first_lock = lock.clone();
        let second_lock = lock.clone();
        let root = dir.path().to_path_buf();
        let root2 = root.clone();
        let path1 = path.clone();
        let path2 = path.clone();
        let seen = seen_mid.clone();

        let first = tokio::spawn(async move {
            execute_mutation_then_command(
                &first_lock,
                &root,
                "a.rs",
                || {
                    let path = path1.clone();
                    async move {
                        std::fs::write(&path, "MUTATING").unwrap();
                        tokio::task::yield_now().await;
                        MutationStep {
                            ok: true,
                            output: "mutated".into(),
                        }
                    }
                },
                Some(|| {
                    let path = path1.clone();
                    async move {
                        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
                        std::fs::write(&path, "DONE").unwrap();
                        CommandStep {
                            ok: true,
                            output: "command".into(),
                        }
                    }
                }),
            )
            .await
        });

        tokio::task::yield_now().await;
        let second = tokio::spawn(async move {
            execute_mutation_then_command(
                &second_lock,
                &root2,
                "a.rs",
                || {
                    let path = path2.clone();
                    let seen = seen.clone();
                    async move {
                        let current = std::fs::read_to_string(&path).unwrap();
                        if current == "MUTATING" {
                            seen.fetch_add(1, Ordering::SeqCst);
                        }
                        MutationStep {
                            ok: true,
                            output: current,
                        }
                    }
                },
                None::<fn() -> std::future::Ready<CommandStep>>,
            )
            .await
        });

        let first = first.await.unwrap();
        let second = second.await.unwrap();
        assert!(first.command_ran);
        assert_eq!(
            seen_mid.load(Ordering::SeqCst),
            0,
            "second fusion observed in-between MUTATING state: {}",
            second.mutation_output
        );
        assert_ne!(second.mutation_output, "MUTATING");
    }
}
