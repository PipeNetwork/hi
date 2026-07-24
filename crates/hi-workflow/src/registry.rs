use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::{DeclarativeWorkflow, MetaError, PhaseMeta, WorkflowMeta, extract_meta};

pub const MAX_WORKFLOW_SOURCE_BYTES: u64 = 1024 * 1024;

fn workflow_meta(source: &str) -> Result<WorkflowMeta, RegistryError> {
    if source.trim_start().starts_with('{') {
        let workflow = DeclarativeWorkflow::from_json(source)
            .map_err(|error| RegistryError::InvalidDefinition(error.to_string()))?;
        workflow.validate().map_err(|error| RegistryError::InvalidDefinition(error.to_string()))?;
        let mut phases = Vec::new();
        collect_declarative_phases(&workflow.steps, &mut phases);
        Ok(WorkflowMeta {
            name: workflow.metadata.name,
            description: workflow.metadata.description,
            when_to_use: None,
            phases,
        })
    } else {
        extract_meta(source).map_err(RegistryError::Meta)
    }
}

fn collect_declarative_phases(steps: &[crate::DeclarativeStep], phases: &mut Vec<PhaseMeta>) {
    for step in steps {
        match step {
            crate::DeclarativeStep::Phase { title } if !phases.iter().any(|p| p.title == *title) => {
                phases.push(PhaseMeta { title: title.clone(), detail: None });
            }
            crate::DeclarativeStep::IfAgentSuccess { then_steps, else_steps, .. } => {
                collect_declarative_phases(then_steps, phases);
                collect_declarative_phases(else_steps, phases);
            }
            _ => {}
        }
    }
}

const BUILTINS: &[(&str, &str)] = &[
    (
        "deep-research",
        include_str!("../workflows/deep-research.workflow.json"),
    ),
    (
        "review-and-fix",
        include_str!("../workflows/review-and-fix.workflow.json"),
    ),
    ("large-review", include_str!("../workflows/large-review.workflow.json")),
    ("port-feature", include_str!("../workflows/port-feature.workflow.json")),
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkflowSource {
    Builtin,
    Project(PathBuf),
    User(PathBuf),
}

#[derive(Debug, Clone)]
pub struct RegisteredWorkflow {
    pub name: String,
    pub script: String,
    pub meta: WorkflowMeta,
    pub source: WorkflowSource,
}

#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("unknown workflow: {0}")]
    Unknown(String),
    #[error("invalid workflow name: {0}")]
    InvalidName(String),
    #[error("workflow source is too large: {path} ({size} bytes)")]
    TooLarge { path: PathBuf, size: u64 },
    #[error("workflow path is not a regular, non-symlink file: {0}")]
    UnsafePath(PathBuf),
    #[error("workflow filename '{filename}' does not match meta.name '{metadata}'")]
    NameMismatch { filename: String, metadata: String },
    #[error("workflow metadata is invalid: {0}")]
    Meta(#[from] MetaError),
    #[error("workflow definition is invalid: {0}")]
    InvalidDefinition(String),
    #[error("workflow I/O failed at {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("workflow already exists: {0}")]
    Exists(PathBuf),
    #[error("duplicate workflow '{name}' at {path}; already defined by {existing}")]
    Duplicate {
        name: String,
        path: PathBuf,
        existing: String,
    },
}

#[derive(Debug, Clone)]
pub struct WorkflowRegistry {
    entries: BTreeMap<String, RegisteredWorkflow>,
}

impl WorkflowRegistry {
    pub fn scan(project_root: Option<&Path>, trust_project: bool) -> Result<Self, RegistryError> {
        let user = user_workflows_dir();
        Self::scan_dirs(
            project_root
                .filter(|_| trust_project)
                .map(|root| root.join(".hi/workflows")),
            user,
        )
    }

    pub fn scan_dirs(
        project: Option<PathBuf>,
        user: Option<PathBuf>,
    ) -> Result<Self, RegistryError> {
        let mut entries = BTreeMap::new();
        for (name, script) in BUILTINS {
            let meta = workflow_meta(script)?;
            entries.insert(
                (*name).to_string(),
                RegisteredWorkflow {
                    name: (*name).to_string(),
                    script: (*script).to_string(),
                    meta,
                    source: WorkflowSource::Builtin,
                },
            );
        }
        if let Some(dir) = user {
            scan_dir(&dir, false, &mut entries)?;
        }
        if let Some(dir) = project {
            scan_dir(&dir, true, &mut entries)?;
        }
        Ok(Self { entries })
    }

    pub fn resolve(&self, name: &str) -> Result<&RegisteredWorkflow, RegistryError> {
        if !valid_workflow_name(name) {
            return Err(RegistryError::InvalidName(name.into()));
        }
        self.entries
            .get(name)
            .ok_or_else(|| RegistryError::Unknown(name.into()))
    }

    pub fn list(&self) -> impl Iterator<Item = &RegisteredWorkflow> {
        self.entries.values()
    }
}

pub fn valid_workflow_name(name: &str) -> bool {
    let b = name.as_bytes();
    !b.is_empty()
        && b.len() <= crate::MAX_WORKFLOW_NAME_LEN
        && b.first()
            .is_some_and(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        && b.last()
            .is_some_and(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        && b.iter()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == b'-')
        && !b.windows(2).any(|p| p == b"--")
}

pub fn user_workflows_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))?;
    Some(base.join("hi/workflows"))
}

pub fn save_project_workflow(root: &Path, script: &str) -> Result<PathBuf, RegistryError> {
    if script.len() as u64 > MAX_WORKFLOW_SOURCE_BYTES {
        return Err(RegistryError::TooLarge {
            path: root.to_path_buf(),
            size: script.len() as u64,
        });
    }
    let meta = extract_meta(script)?;
    let dir = root.join(".hi/workflows");
    fs::create_dir_all(&dir).map_err(|source| RegistryError::Io {
        path: dir.clone(),
        source,
    })?;
    if fs::symlink_metadata(&dir)
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false)
    {
        return Err(RegistryError::UnsafePath(dir));
    }
    let path = dir.join(format!("{}.rhai", meta.name));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|source| {
            if source.kind() == std::io::ErrorKind::AlreadyExists {
                RegistryError::Exists(path.clone())
            } else {
                RegistryError::Io {
                    path: path.clone(),
                    source,
                }
            }
        })?;
    file.write_all(script.as_bytes())
        .and_then(|_| file.sync_all())
        .map_err(|source| RegistryError::Io {
            path: path.clone(),
            source,
        })?;
    Ok(path)
}

fn scan_dir(
    dir: &Path,
    project: bool,
    out: &mut BTreeMap<String, RegisteredWorkflow>,
) -> Result<(), RegistryError> {
    let meta = match fs::symlink_metadata(dir) {
        Ok(meta) => meta,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(source) => {
            return Err(RegistryError::Io {
                path: dir.to_path_buf(),
                source,
            });
        }
    };
    if !meta.is_dir() || meta.file_type().is_symlink() {
        return Err(RegistryError::UnsafePath(dir.to_path_buf()));
    }
    let mut paths = fs::read_dir(dir)
        .map_err(|source| RegistryError::Io {
            path: dir.into(),
            source,
        })?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|e| e == "rhai")
                || p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.ends_with(".workflow.json"))
        })
        .collect::<Vec<_>>();
    paths.sort();
    for path in paths {
        let file_meta = fs::symlink_metadata(&path).map_err(|source| RegistryError::Io {
            path: path.clone(),
            source,
        })?;
        if !file_meta.is_file() || file_meta.file_type().is_symlink() {
            return Err(RegistryError::UnsafePath(path));
        }
        if file_meta.len() > MAX_WORKFLOW_SOURCE_BYTES {
            return Err(RegistryError::TooLarge {
                path,
                size: file_meta.len(),
            });
        }
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .map(|name| name.strip_suffix(".workflow.json").or_else(|| name.strip_suffix(".rhai")).unwrap_or(name))
            .ok_or_else(|| RegistryError::InvalidName(path.display().to_string()))?
            .to_string();
        if !valid_workflow_name(&name) {
            return Err(RegistryError::InvalidName(name));
        }
        let mut options = fs::OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options.custom_flags(libc::O_NOFOLLOW);
        }
        let file = options.open(&path).map_err(|source| RegistryError::Io {
            path: path.clone(),
            source,
        })?;
        let opened_meta = file.metadata().map_err(|source| RegistryError::Io {
            path: path.clone(),
            source,
        })?;
        if !opened_meta.is_file() {
            return Err(RegistryError::UnsafePath(path));
        }
        let mut script = String::new();
        file.take(MAX_WORKFLOW_SOURCE_BYTES + 1)
            .read_to_string(&mut script)
            .map_err(|source| RegistryError::Io {
                path: path.clone(),
                source,
            })?;
        if script.len() as u64 > MAX_WORKFLOW_SOURCE_BYTES {
            return Err(RegistryError::TooLarge {
                path,
                size: script.len() as u64,
            });
        }
        let workflow_meta = workflow_meta(&script)?;
        if workflow_meta.name != name {
            return Err(RegistryError::NameMismatch {
                filename: name,
                metadata: workflow_meta.name,
            });
        }
        if let Some(existing) = out.get(&name) {
            let existing = match &existing.source {
                WorkflowSource::Builtin => "builtin".to_string(),
                WorkflowSource::Project(path) | WorkflowSource::User(path) => path.display().to_string(),
            };
            return Err(RegistryError::Duplicate { name, path, existing });
        }
        out.insert(
            name.clone(),
            RegisteredWorkflow {
                name,
                script,
                meta: workflow_meta,
                source: if project {
                    WorkflowSource::Project(path)
                } else {
                    WorkflowSource::User(path)
                },
            },
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workflow(name: &str) -> String {
        format!(r#"{{"metadata":{{"name":"{name}","description":"test"}},"steps":[{{"type":"complete","result":null}}]}}"#)
    }

    #[test]
    fn duplicate_names_are_rejected_instead_of_overwritten() {
        let dir = tempfile::tempdir().unwrap();
        let user = dir.path().join("user");
        let project = dir.path().join("project");
        std::fs::create_dir_all(&user).unwrap();
        std::fs::create_dir_all(&project).unwrap();
        std::fs::write(user.join("custom.workflow.json"), workflow("custom")).unwrap();
        std::fs::write(project.join("custom.workflow.json"), workflow("custom")).unwrap();
        assert!(matches!(
            WorkflowRegistry::scan_dirs(Some(project), Some(user)),
            Err(RegistryError::Duplicate { name, .. }) if name == "custom"
        ));
    }

    #[cfg(unix)]
    #[test]
    fn scan_rejects_symlinked_workflow_source() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let target = outside.path().join("custom.workflow.json");
        std::fs::write(&target, workflow("custom")).unwrap();
        symlink(target, dir.path().join("custom.workflow.json")).unwrap();
        assert!(matches!(
            WorkflowRegistry::scan_dirs(None, Some(dir.path().to_path_buf())),
            Err(RegistryError::UnsafePath(_))
        ));
    }
}
