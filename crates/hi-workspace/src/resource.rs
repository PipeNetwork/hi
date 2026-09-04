use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceScheme {
    Workspace,
    Artifact,
    Session,
    Job,
    Mcp,
}

impl ResourceScheme {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Workspace => "workspace",
            Self::Artifact => "artifact",
            Self::Session => "session",
            Self::Job => "job",
            Self::Mcp => "mcp",
        }
    }
}

impl FromStr for ResourceScheme {
    type Err = ResourceUriError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "workspace" => Ok(Self::Workspace),
            "artifact" => Ok(Self::Artifact),
            "session" => Ok(Self::Session),
            "job" => Ok(Self::Job),
            "mcp" => Ok(Self::Mcp),
            _ => Err(ResourceUriError::UnsupportedScheme(value.to_owned())),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ResourceUri {
    scheme: ResourceScheme,
    reference: String,
}

impl ResourceUri {
    pub fn parse(value: impl AsRef<str>) -> Result<Self, ResourceUriError> {
        value.as_ref().parse()
    }

    pub fn workspace(path: impl AsRef<Path>) -> Result<Self, ResourceUriError> {
        let value = path
            .as_ref()
            .to_str()
            .ok_or(ResourceUriError::NonUtf8WorkspacePath)?;
        Self::new(ResourceScheme::Workspace, value)
    }

    pub fn new(
        scheme: ResourceScheme,
        reference: impl Into<String>,
    ) -> Result<Self, ResourceUriError> {
        let reference = reference.into();
        validate_reference(scheme, &reference)?;
        Ok(Self { scheme, reference })
    }

    pub const fn scheme(&self) -> ResourceScheme {
        self.scheme
    }

    pub fn reference(&self) -> &str {
        &self.reference
    }

    pub fn workspace_path(&self) -> Option<PathBuf> {
        (self.scheme == ResourceScheme::Workspace).then(|| PathBuf::from(&self.reference))
    }
}

impl fmt::Display for ResourceUri {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}://{}", self.scheme.as_str(), self.reference)
    }
}

impl FromStr for ResourceUri {
    type Err = ResourceUriError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let (scheme, reference) = value
            .split_once("://")
            .ok_or(ResourceUriError::MissingScheme)?;
        Self::new(scheme.parse()?, reference)
    }
}

impl TryFrom<String> for ResourceUri {
    type Error = ResourceUriError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}

impl From<ResourceUri> for String {
    fn from(value: ResourceUri) -> Self {
        value.to_string()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceReadRequest {
    pub uri: ResourceUri,
    pub offset: u64,
    pub limit: Option<u64>,
}

impl ResourceReadRequest {
    pub fn new(uri: ResourceUri, offset: u64, limit: Option<u64>) -> Self {
        Self { uri, offset, limit }
    }

    /// Backward-compatible path entry point. The path remains relative to the
    /// workspace root and receives the same traversal validation as a URI.
    pub fn from_workspace_path(
        path: impl AsRef<Path>,
        offset: u64,
        limit: Option<u64>,
    ) -> Result<Self, ResourceUriError> {
        Ok(Self::new(ResourceUri::workspace(path)?, offset, limit))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ResourceUriError {
    #[error("resource URI must contain ://")]
    MissingScheme,
    #[error("unsupported resource URI scheme {0:?}")]
    UnsupportedScheme(String),
    #[error("resource reference must not be empty")]
    EmptyReference,
    #[error("resource reference contains control characters")]
    ControlCharacter,
    #[error("workspace resource paths must be relative, normalized UTF-8 paths")]
    UnsafeWorkspacePath,
    #[error("workspace resource path is not UTF-8")]
    NonUtf8WorkspacePath,
}

fn validate_reference(scheme: ResourceScheme, reference: &str) -> Result<(), ResourceUriError> {
    if reference.is_empty() {
        return Err(ResourceUriError::EmptyReference);
    }
    if reference.chars().any(char::is_control) {
        return Err(ResourceUriError::ControlCharacter);
    }
    if scheme == ResourceScheme::Workspace {
        let path = Path::new(reference);
        let normalized = !path.is_absolute()
            && !reference.starts_with('/')
            && !reference.starts_with('\\')
            && !reference.contains('\\')
            && reference
                .split('/')
                .all(|segment| !segment.is_empty() && segment != "." && segment != "..")
            && path
                .components()
                .all(|component| matches!(component, std::path::Component::Normal(_)));
        if !normalized {
            return Err(ResourceUriError::UnsafeWorkspacePath);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_uri_is_normalized_and_path_compatible() {
        let request = ResourceReadRequest::from_workspace_path("src/lib.rs", 12, Some(64)).unwrap();
        assert_eq!(request.uri.to_string(), "workspace://src/lib.rs");
        assert_eq!(
            request.uri.workspace_path(),
            Some(PathBuf::from("src/lib.rs"))
        );
        assert_eq!(request.offset, 12);
        assert_eq!(request.limit, Some(64));
    }

    #[test]
    fn workspace_uri_rejects_escape_and_absolute_paths() {
        for value in [
            "workspace://../secret",
            "workspace:///etc/passwd",
            "workspace://src/./lib.rs",
            "workspace://src\\lib.rs",
        ] {
            assert!(matches!(
                ResourceUri::parse(value),
                Err(ResourceUriError::UnsafeWorkspacePath)
            ));
        }
    }

    #[test]
    fn all_public_schemes_round_trip() {
        for value in [
            "artifact://sha256/example",
            "session://current/transcript",
            "job://job-1/output",
            "mcp://server/resource-name",
        ] {
            let uri = ResourceUri::parse(value).unwrap();
            assert_eq!(uri.to_string(), value);
        }
    }
}
