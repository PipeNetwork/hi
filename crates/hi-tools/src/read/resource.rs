//! Resource-URI adaptation and routing for the `read` tool.

use std::fmt;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use hi_workspace::{
    DiagnosticField, DiagnosticRetryability, DiagnosticSeverity, ResourceReadRequest,
    ResourceScheme, ResourceUri, ToolDiagnostic,
};
use serde::Deserialize;

/// A typed failure returned when `read` understands a resource URI but no
/// resolver for its scheme is installed in the local tool host.
#[derive(Clone, Debug)]
pub struct ResourceReadRoutingError {
    pub diagnostic: Box<ToolDiagnostic>,
}

impl ResourceReadRoutingError {
    fn unsupported(request: &ResourceReadRequest) -> Self {
        let scheme = request.uri.scheme();
        let mut diagnostic = ToolDiagnostic::new(
            "resource.read.unsupported_route",
            DiagnosticSeverity::Error,
            format!(
                "the read tool has no resolver for {}:// resources",
                scheme.as_str()
            ),
            DiagnosticRetryability::UserAction,
        );
        diagnostic.fields.insert(
            "scheme".to_owned(),
            DiagnosticField::public(scheme.as_str()),
        );
        diagnostic.fields.insert(
            "reference".to_owned(),
            DiagnosticField::sensitive(request.uri.reference()),
        );
        Self {
            diagnostic: Box::new(diagnostic),
        }
    }

    fn unregistered(request: &ResourceReadRequest) -> Self {
        let scheme = request.uri.scheme();
        let mut diagnostic = ToolDiagnostic::new(
            "resource.read.not_registered",
            DiagnosticSeverity::Error,
            format!(
                "the {}:// resource is not registered in this tool host",
                scheme.as_str()
            ),
            DiagnosticRetryability::UserAction,
        );
        diagnostic.fields.insert(
            "scheme".to_owned(),
            DiagnosticField::public(scheme.as_str()),
        );
        diagnostic.fields.insert(
            "reference".to_owned(),
            DiagnosticField::sensitive(request.uri.reference()),
        );
        Self {
            diagnostic: Box::new(diagnostic),
        }
    }
}

impl fmt::Display for ResourceReadRoutingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.diagnostic.message)
    }
}

impl std::error::Error for ResourceReadRoutingError {}

/// Route a typed request to the resolver built into hi-tools.
///
/// The returned path remains relative and must still pass the workspace
/// resolver before I/O. Known non-workspace schemes return a typed diagnostic;
/// callers with an artifact/session/job/MCP resolver can dispatch them before
/// invoking this local route.
pub fn route_resource_read(
    request: &ResourceReadRequest,
) -> std::result::Result<PathBuf, ResourceReadRoutingError> {
    match request.uri.scheme() {
        ResourceScheme::Workspace => Ok(request
            .uri
            .workspace_path()
            .expect("workspace resource URI always carries a workspace path")),
        ResourceScheme::Artifact
        | ResourceScheme::Session
        | ResourceScheme::Job
        | ResourceScheme::Mcp => Err(ResourceReadRoutingError::unsupported(request)),
    }
}

pub(super) struct RoutedRead {
    pub display: String,
    pub source: RoutedReadSource,
    pub offset: Option<usize>,
    pub limit: Option<usize>,
}

pub(super) enum RoutedReadSource {
    WorkspacePath(String),
    ResourceBody(String),
}

impl RoutedReadSource {
    pub(super) async fn read(self, cache: &std::sync::Mutex<crate::ReadCache>) -> Result<String> {
        match self {
            Self::WorkspacePath(path) => super::read_one(cache, &path).await,
            Self::ResourceBody(body) => {
                if body.len() as u64 > super::MAX_READ_FILE_BYTES {
                    bail!(
                        "resource is too large to read ({} bytes; limit {} bytes)",
                        body.len(),
                        super::MAX_READ_FILE_BYTES
                    );
                }
                Ok(body)
            }
        }
    }
}

pub(super) enum RoutedReadPlan {
    Single(RoutedRead),
    Multiple(Vec<RoutedRead>),
}

#[derive(Deserialize)]
struct ReadArgs {
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    paths: Option<Vec<String>>,
    #[serde(default)]
    uri: Option<String>,
    #[serde(default)]
    offset: Option<u64>,
    #[serde(default)]
    limit: Option<u64>,
}

pub(super) async fn parse_and_route_read(
    root: &Path,
    cache: &std::sync::Mutex<crate::ReadCache>,
    mcp: Option<&dyn crate::McpBackend>,
    arguments: &str,
) -> Result<RoutedReadPlan> {
    let args: ReadArgs = crate::tools::parse(arguments)?;
    let selectors = usize::from(args.path.is_some())
        + usize::from(args.paths.is_some())
        + usize::from(args.uri.is_some());
    if selectors != 1 {
        bail!("`read` requires exactly one of `path`, `paths`, or `uri`");
    }

    let offset = args.offset.unwrap_or(0);
    if let Some(uri) = args.uri {
        let uri = ResourceUri::parse(uri).map_err(anyhow::Error::new)?;
        let display = uri.to_string();
        return Ok(RoutedReadPlan::Single(
            route_one(
                root,
                cache,
                mcp,
                ResourceReadRequest::new(uri, offset, args.limit),
                display,
            )
            .await?,
        ));
    }
    if let Some(path) = args.path {
        return Ok(RoutedReadPlan::Single(route_legacy_path(
            root, path, offset, args.limit,
        )?));
    }

    let paths = args.paths.expect("exactly one selector was present");
    if paths.is_empty() {
        bail!("`paths` must list at least one path");
    }
    if paths.len() > super::MAX_MULTI_READ_PATHS {
        bail!(
            "`paths` may contain at most {} files per call",
            super::MAX_MULTI_READ_PATHS
        );
    }
    let routed = paths
        .into_iter()
        .map(|path| route_legacy_path(root, path, offset, args.limit))
        .collect::<Result<Vec<_>>>()?;
    Ok(RoutedReadPlan::Multiple(routed))
}

async fn route_one(
    root: &Path,
    cache: &std::sync::Mutex<crate::ReadCache>,
    mcp: Option<&dyn crate::McpBackend>,
    request: ResourceReadRequest,
    display: String,
) -> Result<RoutedRead> {
    let (offset, limit) = checked_window(request.offset, request.limit)?;
    let source = match request.uri.scheme() {
        ResourceScheme::Workspace => {
            let path = route_resource_read(&request).map_err(anyhow::Error::new)?;
            RoutedReadSource::WorkspacePath(super::resolve(root, path.to_string_lossy().as_ref())?)
        }
        ResourceScheme::Artifact | ResourceScheme::Session | ResourceScheme::Job => {
            let body = cache
                .lock()
                .map_err(|_| anyhow::anyhow!("read resource cache lock is poisoned"))?
                .resource(&request.uri)
                .ok_or_else(|| {
                    anyhow::Error::new(ResourceReadRoutingError::unregistered(&request))
                })?;
            RoutedReadSource::ResourceBody(body)
        }
        ResourceScheme::Mcp => {
            if let Some(body) = cache
                .lock()
                .map_err(|_| anyhow::anyhow!("read resource cache lock is poisoned"))?
                .resource(&request.uri)
            {
                RoutedReadSource::ResourceBody(body)
            } else {
                let backend = mcp.ok_or_else(|| {
                    anyhow::Error::new(ResourceReadRoutingError::unsupported(&request))
                })?;
                let (server, uri) = split_mcp_reference(request.uri.reference())?;
                RoutedReadSource::ResourceBody(
                    backend
                        .read_resource(server, uri)
                        .await
                        .with_context(|| format!("reading {display}"))?,
                )
            }
        }
    };
    Ok(RoutedRead {
        display,
        source,
        offset,
        limit,
    })
}

// Compatibility paths keep the resolver's established handling of `./`,
// in-workspace `..`, and absolute paths that still resolve beneath `root`.
fn route_legacy_path(
    root: &Path,
    display: String,
    offset: u64,
    limit: Option<u64>,
) -> Result<RoutedRead> {
    let target = super::resolve(root, &display)?;
    let (offset, limit) = checked_window(offset, limit)?;
    Ok(RoutedRead {
        display,
        source: RoutedReadSource::WorkspacePath(target),
        offset,
        limit,
    })
}

fn split_mcp_reference(reference: &str) -> Result<(&str, &str)> {
    let (server, uri) = reference
        .split_once('/')
        .ok_or_else(|| anyhow::anyhow!("mcp:// resource must be mcp://SERVER/RESOURCE_URI"))?;
    if server.is_empty()
        || !server
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        || uri.is_empty()
    {
        bail!("mcp:// resource must contain a safe server name and non-empty resource URI");
    }
    Ok((server, uri))
}

fn checked_window(offset: u64, limit: Option<u64>) -> Result<(Option<usize>, Option<usize>)> {
    let offset = (offset != 0)
        .then(|| usize::try_from(offset).context("read offset exceeds platform limits"))
        .transpose()?;
    let limit = limit
        .map(|limit| usize::try_from(limit).context("read limit exceeds platform limits"))
        .transpose()?;
    Ok((offset, limit))
}

pub(crate) fn workspace_path_from_read_arguments(arguments: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(arguments).ok()?;
    value
        .get("path")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .or_else(|| {
            value
                .get("uri")
                .and_then(serde_json::Value::as_str)
                .and_then(|uri| ResourceUri::parse(uri).ok())
                .and_then(|uri| uri.workspace_path())
                .map(|path| path.to_string_lossy().into_owned())
        })
        .or_else(|| {
            value
                .get("paths")
                .and_then(serde_json::Value::as_array)
                .filter(|paths| paths.len() == 1)
                .and_then(|paths| paths[0].as_str())
                .map(str::to_owned)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_workspace::{ResourceUriError, TOOL_DIAGNOSTIC_SCHEMA_VERSION};

    #[tokio::test]
    async fn workspace_uri_uses_the_actual_read_path_and_window() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("notes.txt"), "alpha\nbravo\ncharlie\n").unwrap();
        let cache = std::sync::Mutex::new(crate::ReadCache::new());

        let output = crate::read::run_read(
            root.path(),
            &cache,
            r#"{"uri":"workspace://notes.txt","offset":2,"limit":1}"#,
        )
        .await
        .unwrap();
        assert!(output.content.contains("   2\tbravo"), "{}", output.content);
        assert!(!output.content.contains("alpha"));
        assert!(!output.content.contains("charlie"));

        let legacy = crate::read::run_read(
            root.path(),
            &cache,
            r#"{"path":"./notes.txt","offset":2,"limit":1}"#,
        )
        .await
        .unwrap();
        assert_eq!(legacy.content, output.content);
    }

    #[tokio::test]
    async fn known_non_workspace_scheme_returns_typed_routing_diagnostic() {
        let root = tempfile::tempdir().unwrap();
        let cache = std::sync::Mutex::new(crate::ReadCache::new());
        let error = crate::read::run_read(
            root.path(),
            &cache,
            r#"{"uri":"artifact://sha256/example","offset":0}"#,
        )
        .await
        .unwrap_err();
        let routing = error
            .downcast_ref::<ResourceReadRoutingError>()
            .expect("routing error retains its concrete diagnostic type");
        assert_eq!(
            routing.diagnostic.schema_version,
            TOOL_DIAGNOSTIC_SCHEMA_VERSION
        );
        assert_eq!(routing.diagnostic.code, "resource.read.not_registered");
        assert_eq!(routing.diagnostic.severity, DiagnosticSeverity::Error);
        assert_eq!(
            routing.diagnostic.retryability,
            DiagnosticRetryability::UserAction
        );
        assert_eq!(routing.diagnostic.fields["scheme"].value, "artifact");
        assert!(routing.diagnostic.fields["reference"].sensitive);
    }

    #[tokio::test]
    async fn host_registered_resources_use_the_same_bounded_line_window() {
        let root = tempfile::tempdir().unwrap();
        let cache = std::sync::Mutex::new(crate::ReadCache::new());
        for scheme in ["artifact", "session", "job"] {
            let uri = ResourceUri::parse(format!("{scheme}://example/output")).unwrap();
            cache
                .lock()
                .unwrap()
                .register_resource(uri.clone(), "alpha\nbravo\ncharlie\n".into())
                .unwrap();
            let output = crate::read::run_read(
                root.path(),
                &cache,
                &format!(r#"{{"uri":"{uri}","offset":2,"limit":1}}"#),
            )
            .await
            .unwrap();
            assert!(
                output.content.contains("   2\tbravo"),
                "{scheme}: {output:?}"
            );
            assert!(!output.content.contains("alpha"));
            assert!(!output.content.contains("charlie"));
            assert!(matches!(
                output.truncation,
                crate::TruncationState::Truncated { .. }
            ));
        }
    }

    struct ResourceMcp;

    #[async_trait::async_trait]
    impl crate::McpBackend for ResourceMcp {
        async fn search(&self, _query: Option<&str>) -> anyhow::Result<Vec<crate::McpToolInfo>> {
            Ok(Vec::new())
        }

        async fn call(
            &self,
            _server: &str,
            _tool: &str,
            _arguments: &serde_json::Value,
        ) -> anyhow::Result<String> {
            unreachable!("resource test does not invoke MCP tools")
        }

        async fn read_resource(&self, server: &str, uri: &str) -> anyhow::Result<String> {
            assert_eq!(server, "docs");
            assert_eq!(uri, "file:///guide.md");
            Ok("heading\nbody\nfooter\n".into())
        }
    }

    #[tokio::test]
    async fn mcp_resource_routes_to_the_connected_backend_and_pages_uniformly() {
        let root = tempfile::tempdir().unwrap();
        let cache = std::sync::Mutex::new(crate::ReadCache::new());
        let output = crate::read::run_read_with_mcp(
            root.path(),
            &cache,
            Some(&ResourceMcp),
            r#"{"uri":"mcp://docs/file:///guide.md","offset":2,"limit":1}"#,
        )
        .await
        .unwrap();
        assert!(output.content.contains("   2\tbody"), "{}", output.content);
        assert!(!output.content.contains("heading"));
        assert!(!output.content.contains("footer"));
        assert!(matches!(
            output.truncation,
            crate::TruncationState::Truncated { .. }
        ));
    }

    #[tokio::test]
    async fn mcp_resource_without_backend_is_a_typed_route_failure() {
        let root = tempfile::tempdir().unwrap();
        let cache = std::sync::Mutex::new(crate::ReadCache::new());
        let error = crate::read::run_read(
            root.path(),
            &cache,
            r#"{"uri":"mcp://docs/file:///guide.md"}"#,
        )
        .await
        .unwrap_err();
        let routing = error.downcast_ref::<ResourceReadRoutingError>().unwrap();
        assert_eq!(routing.diagnostic.code, "resource.read.unsupported_route");
    }

    #[tokio::test]
    async fn registered_resource_uses_the_production_redaction_boundary() {
        let root = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        let lsp = std::sync::Arc::new(hi_lsp::LspManager::new(root.path()).unwrap());
        let background = crate::BackgroundRegistry::default();
        let cache = std::sync::Mutex::new(crate::ReadCache::new());
        let repo_map = std::sync::Arc::new(std::sync::Mutex::new(crate::RepoMapCache::new()));
        let uri = ResourceUri::parse("artifact://candidate/secret").unwrap();
        let secret = "sk-abcdefghijklmnopqrstuvwxyz123456";
        cache
            .lock()
            .unwrap()
            .register_resource(uri.clone(), format!("token={secret}\n"))
            .unwrap();

        let output = crate::execute_in_runtime_shared_with(
            root.path(),
            state.path(),
            &lsp,
            &background,
            &cache,
            &repo_map,
            None,
            None,
            "read",
            &format!(r#"{{"uri":"{uri}"}}"#),
        )
        .await;
        assert_eq!(output.status, crate::ToolStatus::Succeeded);
        assert!(!output.content.contains(secret), "{}", output.content);
        assert!(
            output.content.contains("REDACTED_SECRET"),
            "{}",
            output.content
        );
    }

    #[tokio::test]
    async fn unsafe_workspace_uri_and_ambiguous_compatibility_input_are_rejected() {
        let root = tempfile::tempdir().unwrap();
        let cache = std::sync::Mutex::new(crate::ReadCache::new());
        let unsafe_uri =
            crate::read::run_read(root.path(), &cache, r#"{"uri":"workspace://../secret"}"#)
                .await
                .unwrap_err();
        assert!(unsafe_uri.downcast_ref::<ResourceUriError>().is_some());

        let ambiguous = crate::read::run_read(
            root.path(),
            &cache,
            r#"{"path":"notes.txt","uri":"workspace://notes.txt"}"#,
        )
        .await
        .unwrap_err();
        assert!(ambiguous.to_string().contains("exactly one"));
    }

    #[test]
    fn dependency_target_extraction_understands_workspace_uris_only() {
        assert_eq!(
            workspace_path_from_read_arguments(r#"{"uri":"workspace://src/lib.rs"}"#).as_deref(),
            Some("src/lib.rs")
        );
        assert!(workspace_path_from_read_arguments(r#"{"uri":"job://job-1/output"}"#).is_none());
    }
}
