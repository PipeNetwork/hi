//! Hugging Face Hub discovery and file resolution.
//!
//! This is separate from provider `/models` listing: Hub repositories may be
//! downloadable artifacts, but they are not necessarily callable by the active
//! chat provider.

use std::fmt;

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use reqwest::header;
use serde::Deserialize;
use serde_json::Value;

const DEFAULT_HF_ENDPOINT: &str = "https://huggingface.co";
const DEFAULT_REVISION: &str = "main";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HfRepoRef {
    pub repo_id: String,
    pub revision: String,
    pub filename: Option<String>,
}

impl HfRepoRef {
    pub fn parse(input: &str) -> Result<Self> {
        let input = input.trim();
        if input.is_empty() {
            bail!("Hugging Face repo reference is empty");
        }
        if input.starts_with("http://") || input.starts_with("https://") {
            return parse_hf_url(input);
        }

        let short = input
            .trim_start_matches("huggingface.co/")
            .trim_start_matches("hf.co/");
        let (repo_rev, filename) = split_once(short, ':');
        let (repo_id, revision) = split_once(repo_rev, '@');
        validate_repo_id(repo_id)?;
        let revision = revision.unwrap_or(DEFAULT_REVISION).trim();
        if revision.is_empty() {
            bail!("Hugging Face revision is empty");
        }
        let filename = filename.map(str::trim).filter(|s| !s.is_empty());
        Ok(Self {
            repo_id: repo_id.to_string(),
            revision: revision.to_string(),
            filename: filename.map(str::to_string),
        })
    }

    pub fn with_filename(mut self, filename: impl Into<String>) -> Self {
        self.filename = Some(filename.into());
        self
    }

    pub fn suggested_output_name(&self) -> String {
        self.filename
            .as_deref()
            .and_then(|file| file.rsplit('/').next())
            .filter(|name| !name.is_empty())
            .unwrap_or("download")
            .to_string()
    }
}

impl std::str::FromStr for HfRepoRef {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        Self::parse(s)
    }
}

impl fmt::Display for HfRepoRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}", self.repo_id, self.revision)?;
        if let Some(filename) = &self.filename {
            write!(f, ":{filename}")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HfModelInfo {
    pub id: String,
    #[serde(default)]
    pub tags: Vec<String>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
    pub updated_at: Option<String>,
    #[serde(default)]
    pub files: Vec<HfFileInfo>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HfFileInfo {
    pub path: String,
    pub size: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ModelSource {
    Provider,
    PipeMcp,
    HuggingFace,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ModelCandidate {
    pub id: String,
    pub source: ModelSource,
    pub runnable: bool,
    pub downloadable: bool,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub files: Vec<HfFileInfo>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
    pub updated_at: Option<String>,
    pub context_window: Option<u32>,
    pub max_output_tokens: Option<u32>,
    pub price: Option<(f64, f64)>,
    #[serde(default)]
    pub capabilities: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct ModelDiscoveryQuery {
    pub query: String,
    pub limit: usize,
}

impl ModelDiscoveryQuery {
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            limit: 10,
        }
    }
}

#[async_trait]
pub trait ModelDiscovery: Send + Sync {
    async fn search(&self, query: ModelDiscoveryQuery) -> Result<Vec<ModelCandidate>>;
}

#[derive(Clone)]
pub struct HuggingFaceHubClient {
    http: reqwest::Client,
    endpoint: String,
    token: Option<String>,
}

impl HuggingFaceHubClient {
    pub fn from_env() -> Self {
        let endpoint =
            non_empty_env("HI_HF_ENDPOINT").unwrap_or_else(|| DEFAULT_HF_ENDPOINT.to_string());
        let token = non_empty_env("HI_HF_TOKEN");
        Self::new(endpoint, token)
    }

    pub fn new(endpoint: impl Into<String>, token: Option<String>) -> Self {
        Self {
            http: crate::http::agent_http_client(),
            endpoint: endpoint.into().trim_end_matches('/').to_string(),
            token,
        }
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub async fn search_models(
        &self,
        query: impl AsRef<str>,
        limit: usize,
    ) -> Result<Vec<ModelCandidate>> {
        let query = query.as_ref().trim();
        if query.is_empty() {
            bail!("Hugging Face search query is empty");
        }
        let limit = limit.clamp(1, 50).to_string();
        let response = self
            .with_auth(
                self.http
                    .get(format!("{}/api/models", self.endpoint))
                    .query(&[
                        ("search", query),
                        ("limit", limit.as_str()),
                        ("full", "true"),
                    ])
                    .header(header::ACCEPT, "application/json"),
            )
            .send()
            .await
            .context("searching Hugging Face models")?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            bail!(
                "Hugging Face search returned {status}: {}",
                clip(&body, 240)
            );
        }
        let value: Value = response
            .json()
            .await
            .context("parsing Hugging Face search")?;
        Ok(parse_hub_models(&value)
            .into_iter()
            .map(HubModel::into_candidate)
            .collect())
    }

    pub async fn author_models(
        &self,
        author: impl AsRef<str>,
        limit: usize,
    ) -> Result<Vec<ModelCandidate>> {
        let author = author.as_ref().trim();
        if author.is_empty() {
            bail!("Hugging Face author is empty");
        }
        let max_models = limit.max(1);
        let mut cursor: Option<String> = None;
        let mut models = Vec::new();
        while models.len() < max_models {
            let page_limit = (max_models - models.len()).clamp(1, 100).to_string();
            let mut request = self
                .http
                .get(format!("{}/api/models", self.endpoint))
                .query(&[
                    ("author", author),
                    ("limit", page_limit.as_str()),
                    ("full", "true"),
                ])
                .header(header::ACCEPT, "application/json");
            if let Some(cursor) = &cursor {
                request = request.query(&[("cursor", cursor.as_str())]);
            }
            let response = self
                .with_auth(request)
                .send()
                .await
                .with_context(|| format!("listing Hugging Face models for author {author}"))?;
            let status = response.status();
            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                bail!(
                    "Hugging Face author listing returned {status}: {}",
                    clip(&body, 240)
                );
            }
            let next_cursor = next_cursor_from_headers(response.headers());
            let value: Value = response
                .json()
                .await
                .context("parsing Hugging Face author listing")?;
            let before = models.len();
            models.extend(
                parse_hub_models(&value)
                    .into_iter()
                    .map(HubModel::into_candidate),
            );
            if models.len() >= max_models {
                models.truncate(max_models);
                break;
            }
            if models.len() == before {
                break;
            }
            let Some(next_cursor) = next_cursor else {
                break;
            };
            cursor = Some(next_cursor);
        }
        Ok(models)
    }

    pub async fn model_info(&self, repo: &HfRepoRef) -> Result<HfModelInfo> {
        let url = format!(
            "{}/api/models/{}/revision/{}",
            self.endpoint,
            encode_repo_id(&repo.repo_id),
            encode_path(&repo.revision),
        );
        let response = self
            .with_auth(
                self.http
                    .get(url)
                    .header(header::ACCEPT, "application/json"),
            )
            .send()
            .await
            .with_context(|| format!("fetching Hugging Face model info for {}", repo.repo_id))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            bail!(
                "Hugging Face model info returned {status}: {}",
                clip(&body, 240)
            );
        }
        let value: Value = response
            .json()
            .await
            .context("parsing Hugging Face model info")?;
        let model = parse_hub_model(&value)
            .ok_or_else(|| anyhow!("Hugging Face model info response missing model id"))?;
        Ok(model.into_info())
    }

    pub async fn list_files(&self, repo: &HfRepoRef) -> Result<Vec<HfFileInfo>> {
        let url = format!(
            "{}/api/models/{}/tree/{}",
            self.endpoint,
            encode_repo_id(&repo.repo_id),
            encode_path(&repo.revision),
        );
        let response = self
            .with_auth(
                self.http
                    .get(url)
                    .query(&[("recursive", "true")])
                    .header(header::ACCEPT, "application/json"),
            )
            .send()
            .await
            .with_context(|| format!("listing Hugging Face files for {}", repo.repo_id))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            bail!(
                "Hugging Face file listing returned {status}: {}",
                clip(&body, 240)
            );
        }
        let entries: Vec<HubTreeEntry> = response
            .json()
            .await
            .context("parsing Hugging Face file tree")?;
        Ok(entries
            .into_iter()
            .filter(|entry| entry.kind.as_deref() == Some("file"))
            .filter_map(|entry| {
                entry.path.map(|path| HfFileInfo {
                    path,
                    size: entry.size,
                })
            })
            .collect())
    }

    pub fn resolve_file_url(&self, repo: &HfRepoRef) -> Result<String> {
        let filename = repo
            .filename
            .as_deref()
            .ok_or_else(|| anyhow!("Hugging Face filename is required"))?;
        Ok(format!(
            "{}/{}/resolve/{}/{}",
            self.endpoint,
            encode_repo_id(&repo.repo_id),
            encode_path(&repo.revision),
            encode_path(filename),
        ))
    }

    fn with_auth(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match self
            .token
            .as_deref()
            .filter(|token| !token.trim().is_empty())
        {
            Some(token) => builder.bearer_auth(token),
            None => builder,
        }
    }
}

#[async_trait]
impl ModelDiscovery for HuggingFaceHubClient {
    async fn search(&self, query: ModelDiscoveryQuery) -> Result<Vec<ModelCandidate>> {
        self.search_models(query.query, query.limit).await
    }
}

#[derive(Debug)]
struct HubModel {
    id: String,
    tags: Vec<String>,
    downloads: Option<u64>,
    likes: Option<u64>,
    updated_at: Option<String>,
    siblings: Vec<HubSibling>,
}

impl HubModel {
    fn into_candidate(self) -> ModelCandidate {
        ModelCandidate {
            id: self.id,
            source: ModelSource::HuggingFace,
            runnable: false,
            downloadable: true,
            tags: self.tags,
            files: siblings_to_files(self.siblings),
            downloads: self.downloads,
            likes: self.likes,
            updated_at: self.updated_at,
            context_window: None,
            max_output_tokens: None,
            price: None,
            capabilities: Vec::new(),
        }
    }

    fn into_info(self) -> HfModelInfo {
        HfModelInfo {
            id: self.id,
            tags: self.tags,
            downloads: self.downloads,
            likes: self.likes,
            updated_at: self.updated_at,
            files: siblings_to_files(self.siblings),
        }
    }
}

#[derive(Debug)]
struct HubSibling {
    filename: String,
    size: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct HubTreeEntry {
    path: Option<String>,
    #[serde(default, rename = "type")]
    kind: Option<String>,
    size: Option<u64>,
}

fn siblings_to_files(siblings: Vec<HubSibling>) -> Vec<HfFileInfo> {
    siblings
        .into_iter()
        .filter(|s| !s.filename.is_empty())
        .map(|s| HfFileInfo {
            path: s.filename,
            size: s.size,
        })
        .collect()
}

fn parse_hub_models(value: &Value) -> Vec<HubModel> {
    let items = value
        .as_array()
        .or_else(|| value.get("models").and_then(Value::as_array))
        .into_iter()
        .flatten();
    items.filter_map(parse_hub_model).collect()
}

fn parse_hub_model(value: &Value) -> Option<HubModel> {
    let id = string_field(value, &["id", "modelId"])?;
    Some(HubModel {
        id,
        tags: string_array_field(value, "tags"),
        downloads: u64_field(value, "downloads"),
        likes: u64_field(value, "likes"),
        updated_at: string_field(value, &["lastModified", "updated_at"]),
        siblings: value
            .get("siblings")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(parse_hub_sibling)
            .collect(),
    })
}

fn parse_hub_sibling(value: &Value) -> Option<HubSibling> {
    Some(HubSibling {
        filename: string_field(value, &["rfilename", "filename", "path"])?,
        size: u64_field(value, "size"),
    })
}

fn string_field(value: &Value, names: &[&str]) -> Option<String> {
    names
        .iter()
        .find_map(|name| value.get(*name).and_then(Value::as_str))
        .map(str::to_string)
}

fn string_array_field(value: &Value, name: &str) -> Vec<String> {
    value
        .get(name)
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect()
}

fn u64_field(value: &Value, name: &str) -> Option<u64> {
    match value.get(name)? {
        Value::Number(n) => n.as_u64().or_else(|| n.as_f64().map(|v| v.max(0.0) as u64)),
        Value::String(s) => s.parse::<u64>().ok(),
        _ => None,
    }
}

fn parse_hf_url(input: &str) -> Result<HfRepoRef> {
    let without_scheme = input
        .strip_prefix("https://")
        .or_else(|| input.strip_prefix("http://"))
        .unwrap_or(input);
    let (host, path) = split_once(without_scheme, '/');
    let host = host.trim_end_matches('/');
    if !matches!(host, "huggingface.co" | "hf.co") {
        bail!("not a Hugging Face URL: {input}");
    }
    let path = path.unwrap_or("").trim_matches('/');
    let parts: Vec<&str> = path.split('/').collect();
    if parts.len() < 5 || parts[2] != "resolve" {
        bail!("Hugging Face URL must be a /org/model/resolve/revision/file URL");
    }
    let repo_id = format!("{}/{}", parts[0], parts[1]);
    validate_repo_id(&repo_id)?;
    let revision = parts[3];
    if revision.is_empty() {
        bail!("Hugging Face resolve URL has an empty revision");
    }
    let filename = parts[4..].join("/");
    if filename.is_empty() {
        bail!("Hugging Face resolve URL has an empty filename");
    }
    Ok(HfRepoRef {
        repo_id,
        revision: revision.to_string(),
        filename: Some(filename),
    })
}

fn validate_repo_id(repo_id: &str) -> Result<()> {
    let mut parts = repo_id.split('/');
    let Some(owner) = parts.next() else {
        bail!("invalid Hugging Face repo id");
    };
    let Some(name) = parts.next() else {
        bail!("invalid Hugging Face repo id `{repo_id}`; expected `org/model`");
    };
    if parts.next().is_some()
        || owner.is_empty()
        || name.is_empty()
        || !valid_repo_component(owner)
        || !valid_repo_component(name)
    {
        bail!("invalid Hugging Face repo id `{repo_id}`; expected `org/model`");
    }
    Ok(())
}

fn valid_repo_component(s: &str) -> bool {
    s.chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

fn split_once(s: &str, needle: char) -> (&str, Option<&str>) {
    match s.split_once(needle) {
        Some((left, right)) => (left, Some(right)),
        None => (s, None),
    }
}

fn encode_repo_id(repo_id: &str) -> String {
    repo_id
        .split('/')
        .map(percent_encode)
        .collect::<Vec<_>>()
        .join("/")
}

fn encode_path(path: &str) -> String {
    path.split('/')
        .map(percent_encode)
        .collect::<Vec<_>>()
        .join("/")
}

fn percent_encode(input: &str) -> String {
    let mut out = String::new();
    for byte in input.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

fn next_cursor_from_headers(headers: &header::HeaderMap) -> Option<String> {
    headers
        .get(header::LINK)
        .and_then(|value| value.to_str().ok())
        .and_then(next_cursor_from_link)
}

fn next_cursor_from_link(link: &str) -> Option<String> {
    for part in link.split(',') {
        let part = part.trim();
        if !(part.contains("rel=\"next\"") || part.contains("rel=next")) {
            continue;
        }
        let url = part.strip_prefix('<')?.split_once('>')?.0;
        let query = url.split_once('?')?.1;
        for param in query.split('&') {
            let (key, value) = param.split_once('=').unwrap_or((param, ""));
            if key == "cursor" {
                return Some(percent_decode(value));
            }
        }
    }
    None
}

fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let (Some(hi), Some(lo)) = (hex_value(bytes[i + 1]), hex_value(bytes[i + 2]))
        {
            out.push((hi << 4) | lo);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn non_empty_env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|s| !s.trim().is_empty())
}

fn clip(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let end = s.char_indices().nth(max).map(|(i, _)| i).unwrap_or(s.len());
        format!("{}...", s[..end].trim_end())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::{Arc, Mutex};

    #[test]
    fn parses_basic_repo() {
        let r = HfRepoRef::parse("org/model").unwrap();
        assert_eq!(r.repo_id, "org/model");
        assert_eq!(r.revision, "main");
        assert_eq!(r.filename, None);
    }

    #[test]
    fn parses_repo_revision() {
        let r = HfRepoRef::parse("org/model@refs-pr-1").unwrap();
        assert_eq!(r.repo_id, "org/model");
        assert_eq!(r.revision, "refs-pr-1");
    }

    #[test]
    fn parses_repo_filename() {
        let r = HfRepoRef::parse("org/model:file.gguf").unwrap();
        assert_eq!(r.revision, "main");
        assert_eq!(r.filename.as_deref(), Some("file.gguf"));
    }

    #[test]
    fn parses_repo_revision_filename() {
        let r = HfRepoRef::parse("org/model@v1:nested/file.gguf").unwrap();
        assert_eq!(r.repo_id, "org/model");
        assert_eq!(r.revision, "v1");
        assert_eq!(r.filename.as_deref(), Some("nested/file.gguf"));
    }

    #[test]
    fn parses_hf_resolve_url() {
        let r = HfRepoRef::parse("https://huggingface.co/org/model/resolve/main/nested/file.gguf")
            .unwrap();
        assert_eq!(r.repo_id, "org/model");
        assert_eq!(r.revision, "main");
        assert_eq!(r.filename.as_deref(), Some("nested/file.gguf"));
    }

    #[test]
    fn rejects_invalid_repo_ids() {
        assert!(HfRepoRef::parse("model").is_err());
        assert!(HfRepoRef::parse("org/model/extra").is_err());
        assert!(HfRepoRef::parse("org!/model").is_err());
    }

    #[test]
    fn resolves_default_and_nested_download_urls() {
        let client = HuggingFaceHubClient::new("https://huggingface.co", None);
        let url = client
            .resolve_file_url(&HfRepoRef::parse("org/model:nested/file name.gguf").unwrap())
            .unwrap();
        assert_eq!(
            url,
            "https://huggingface.co/org/model/resolve/main/nested/file%20name.gguf"
        );
    }

    #[test]
    fn suggested_output_name_uses_file_basename() {
        let r = HfRepoRef::parse("org/model@v1:nested/file.gguf").unwrap();
        assert_eq!(r.suggested_output_name(), "file.gguf");
    }

    #[tokio::test]
    async fn search_maps_hub_json_without_marking_runnable() {
        let server = MockHub::new(
            200,
            r#"[{"id":"org/model","tags":["gguf","text-generation"],"downloads":42,"likes":7,"lastModified":"2026-01-02T03:04:05Z","siblings":[{"rfilename":"model.gguf","size":1024}]}]"#,
        );
        let client = HuggingFaceHubClient::new(server.url(), None);

        let models = client.search_models("model", 5).await.unwrap();

        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "org/model");
        assert_eq!(models[0].source, ModelSource::HuggingFace);
        assert!(!models[0].runnable);
        assert!(models[0].downloadable);
        assert_eq!(models[0].downloads, Some(42));
        assert_eq!(models[0].likes, Some(7));
        assert_eq!(models[0].files[0].path, "model.gguf");
    }

    #[tokio::test]
    async fn search_tolerates_nullable_or_unexpected_metadata_fields() {
        let server = MockHub::new(
            200,
            r#"[{"modelId":"org/model","tags":null,"downloads":"42","likes":null,"siblings":null},{"tags":["missing-id"]}]"#,
        );
        let client = HuggingFaceHubClient::new(server.url(), None);

        let models = client.search_models("model", 5).await.unwrap();

        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "org/model");
        assert_eq!(models[0].tags, Vec::<String>::new());
        assert_eq!(models[0].downloads, Some(42));
        assert_eq!(models[0].likes, None);
        assert!(models[0].files.is_empty());
    }

    #[tokio::test]
    async fn author_models_queries_author_catalog() {
        let server = MockHub::new(
            200,
            r#"[{"id":"pipenetwork/model","downloads":7,"siblings":[{"rfilename":"model.gguf"}]}]"#,
        );
        let client = HuggingFaceHubClient::new(server.url(), None);

        let models = client.author_models("pipenetwork", 100).await.unwrap();

        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "pipenetwork/model");
        assert_eq!(models[0].downloads, Some(7));
        assert_eq!(models[0].files[0].path, "model.gguf");
        let request = server.request();
        assert!(request.contains("/api/models?"));
        assert!(request.contains("author=pipenetwork"));
        assert!(request.contains("limit=100"));
        assert!(request.contains("full=true"));
    }

    #[tokio::test]
    async fn author_models_follows_next_link_until_limit() {
        let server = MockHub::new_sequence(vec![
            MockReply {
                status: 200,
                body: r#"[{"id":"pipenetwork/one","siblings":[{"rfilename":"one.gguf"}]}]"#,
                headers: vec![(
                    "link",
                    r#"</api/models?author=pipenetwork&limit=1&full=true&cursor=page2>; rel="next""#,
                )],
            },
            MockReply {
                status: 200,
                body: r#"[{"id":"pipenetwork/two","siblings":[{"rfilename":"two.gguf"}]}]"#,
                headers: Vec::new(),
            },
        ]);
        let client = HuggingFaceHubClient::new(server.url(), None);

        let models = client.author_models("pipenetwork", 2).await.unwrap();

        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "pipenetwork/one");
        assert_eq!(models[1].id, "pipenetwork/two");
        let requests = server.requests();
        assert_eq!(requests.len(), 2);
        assert!(requests[0].contains("limit=2"));
        assert!(requests[1].contains("cursor=page2"));
    }

    #[tokio::test]
    async fn list_files_maps_tree_and_sends_token_only_when_configured() {
        let server = MockHub::new(
            200,
            r#"[{"path":"model.gguf","type":"file","size":2048},{"path":"docs","type":"directory"}]"#,
        );
        let client = HuggingFaceHubClient::new(server.url(), Some("secret-token".into()));

        let files = client
            .list_files(&HfRepoRef::parse("org/model@v1").unwrap())
            .await
            .unwrap();

        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "model.gguf");
        assert_eq!(files[0].size, Some(2048));
        assert!(
            server
                .request()
                .contains("authorization: Bearer secret-token")
        );
    }

    #[tokio::test]
    async fn hub_errors_are_concise() {
        let server = MockHub::new(404, r#"{"error":"missing"}"#);
        let client = HuggingFaceHubClient::new(server.url(), None);

        let err = client.search_models("missing", 1).await.unwrap_err();

        assert!(err.to_string().contains("Hugging Face search returned 404"));
    }

    struct MockReply {
        status: u16,
        body: &'static str,
        headers: Vec<(&'static str, &'static str)>,
    }

    struct MockHub {
        url: String,
        requests: Arc<Mutex<Vec<String>>>,
    }

    impl MockHub {
        fn new(status: u16, body: &'static str) -> Self {
            Self::new_sequence(vec![MockReply {
                status,
                body,
                headers: Vec::new(),
            }])
        }

        fn new_sequence(responses: Vec<MockReply>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let requests = Arc::new(Mutex::new(Vec::new()));
            let thread_requests = requests.clone();
            std::thread::spawn(move || {
                for reply in responses {
                    let Ok((mut stream, _)) = listener.accept() else {
                        return;
                    };
                    let raw = read_request(&mut stream);
                    thread_requests.lock().unwrap().push(raw);
                    let reason = if reply.status == 200 { "OK" } else { "Status" };
                    let mut response = format!(
                        "HTTP/1.1 {} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n",
                        reply.status,
                        reply.body.len()
                    );
                    for (name, value) in reply.headers {
                        response.push_str(name);
                        response.push_str(": ");
                        response.push_str(value);
                        response.push_str("\r\n");
                    }
                    response.push_str("\r\n");
                    response.push_str(reply.body);
                    let _ = stream.write_all(response.as_bytes());
                }
            });
            Self { url, requests }
        }

        fn url(&self) -> String {
            self.url.clone()
        }

        fn request(&self) -> String {
            self.requests
                .lock()
                .unwrap()
                .first()
                .cloned()
                .unwrap_or_default()
        }

        fn requests(&self) -> Vec<String> {
            self.requests.lock().unwrap().clone()
        }
    }

    fn read_request(stream: &mut TcpStream) -> String {
        let mut bytes = Vec::new();
        let mut buf = [0u8; 1024];
        loop {
            let n = stream.read(&mut buf).unwrap();
            if n == 0 {
                break;
            }
            bytes.extend_from_slice(&buf[..n]);
            if bytes.windows(4).any(|w| w == b"\r\n\r\n") {
                break;
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}
