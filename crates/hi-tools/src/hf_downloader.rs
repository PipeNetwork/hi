//! In-process, resumable Hugging Face downloads for managed local runtimes.
//!
//! The generic web/Hugging Face tools still have their shell-based download
//! compatibility path. Managed MLX provisioning uses this module instead so
//! it never depends on `curl`, `aria2c`, a shell, or a package manager.

use anyhow::{Context, Result, anyhow, bail};
use futures_util::StreamExt;
use reqwest::{StatusCode, header};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex as StdMutex};
use std::time::Duration;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

const MANIFEST_VERSION: u8 = 1;
const DEFAULT_MAX_PARALLEL_REQUESTS: usize = 8;
const DEFAULT_RANGE_SIZE: u64 = 64 * 1024 * 1024;
const DEFAULT_MAX_RETRIES: usize = 5;
const PROGRESS_PERSIST_BYTES: u64 = 1024 * 1024;
const ERROR_BODY_LIMIT: usize = 4096;

static DOWNLOAD_LOCKS: LazyLock<StdMutex<HashMap<PathBuf, Arc<tokio::sync::Mutex<()>>>>> =
    LazyLock::new(|| StdMutex::new(HashMap::new()));

fn download_lock(output_dir: &Path) -> Arc<tokio::sync::Mutex<()>> {
    let key = output_dir.to_path_buf();
    let mut locks = DOWNLOAD_LOCKS
        .lock()
        .expect("download lock registry poisoned");
    locks
        .entry(key)
        .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
        .clone()
}

fn manifest_completed_bytes(path: &Path) -> u64 {
    let Some(raw_destination) = path
        .to_str()
        .and_then(|path| path.strip_suffix(".hi-part.json"))
    else {
        return 0;
    };
    // A crash can happen after the final atomic rename and before manifest
    // cleanup. The final file is already counted separately, so never count
    // this stale manifest twice.
    if Path::new(raw_destination).is_file() {
        return 0;
    }
    fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_json::from_str::<DownloadManifest>(&raw).ok())
        .map(|manifest| manifest.completed_bytes)
        .unwrap_or(0)
}

/// Configuration for a managed Hugging Face repository download.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HfDownloadOptions {
    pub max_parallel_requests: usize,
    pub range_size: u64,
    pub max_retries: usize,
}

impl Default for HfDownloadOptions {
    fn default() -> Self {
        Self {
            max_parallel_requests: DEFAULT_MAX_PARALLEL_REQUESTS,
            range_size: DEFAULT_RANGE_SIZE,
            max_retries: DEFAULT_MAX_RETRIES,
        }
    }
}

impl HfDownloadOptions {
    fn normalized(&self) -> Self {
        Self {
            max_parallel_requests: self.max_parallel_requests.max(1),
            range_size: self.range_size.max(1),
            max_retries: self.max_retries,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum DownloadMode {
    Ranges,
    Sequential,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DownloadManifest {
    version: u8,
    repo: String,
    revision: String,
    path: String,
    expected_bytes: Option<u64>,
    range_size: u64,
    mode: DownloadMode,
    #[serde(default)]
    completed_ranges: Vec<bool>,
    #[serde(default)]
    completed_bytes: u64,
    #[serde(default)]
    etag: Option<String>,
}

impl DownloadManifest {
    fn new(repo: &hi_ai::HfRepoRef, file: &hi_ai::HfFileInfo, options: &HfDownloadOptions) -> Self {
        let completed_ranges = file
            .size
            .map(|size| range_count(size, options.range_size))
            .map(|count| vec![false; count])
            .unwrap_or_default();
        Self {
            version: MANIFEST_VERSION,
            repo: repo.repo_id.clone(),
            revision: repo.revision.clone(),
            path: file.path.clone(),
            expected_bytes: file.size,
            range_size: options.range_size,
            mode: DownloadMode::Ranges,
            completed_ranges,
            completed_bytes: 0,
            etag: None,
        }
    }

    fn is_compatible(
        &self,
        repo: &hi_ai::HfRepoRef,
        file: &hi_ai::HfFileInfo,
        options: &HfDownloadOptions,
    ) -> bool {
        self.version == MANIFEST_VERSION
            && self.repo == repo.repo_id
            && self.revision == repo.revision
            && self.path == file.path
            && self.expected_bytes == file.size
            && self.range_size == options.range_size
            && (file.size.is_none()
                || self.completed_ranges.len()
                    == range_count(file.size.unwrap_or_default(), options.range_size))
    }

    fn recompute_completed_bytes(&mut self) {
        self.completed_bytes = match (self.expected_bytes, self.mode) {
            (Some(size), DownloadMode::Ranges) => self
                .completed_ranges
                .iter()
                .enumerate()
                .filter(|(_, completed)| **completed)
                .map(|(index, _)| range_len(index, size, self.range_size))
                .sum(),
            (_, DownloadMode::Sequential) => self.completed_bytes,
            (None, DownloadMode::Ranges) => 0,
        };
    }
}

#[derive(Clone, Debug)]
struct PartialPaths {
    destination: PathBuf,
    part: PathBuf,
    manifest: PathBuf,
}

#[derive(Debug)]
enum RangeFailure {
    Unsupported,
    Error(anyhow::Error),
}

impl From<anyhow::Error> for RangeFailure {
    fn from(error: anyhow::Error) -> Self {
        Self::Error(error)
    }
}

#[derive(Debug)]
struct RangeResult {
    index: usize,
    etag: Option<String>,
}

/// Download every file in a Hugging Face repository using in-process HTTP.
///
/// The semaphore is shared across files and ranges, so a repository with many
/// shards cannot exceed the configured connection budget. Dropping the future
/// drops the range tasks and reqwest responses, leaving the manifest-backed
/// partial state for the next invocation.
pub async fn download_repo(
    client: &hi_ai::HuggingFaceHubClient,
    repo: &hi_ai::HfRepoRef,
    files: &[hi_ai::HfFileInfo],
    output_dir: &Path,
    options: HfDownloadOptions,
) -> Result<()> {
    if files.is_empty() {
        bail!("no files found in {}@{}", repo.repo_id, repo.revision);
    }
    let options = options.normalized();
    // Prevent two provider/runtime paths in the same process from mutating
    // one manifest-backed cache concurrently. The second caller waits and then
    // observes the first caller's atomically finalized files.
    let _download_guard = download_lock(output_dir).lock_owned().await;
    tokio::fs::create_dir_all(output_dir)
        .await
        .with_context(|| format!("creating model directory {}", output_dir.display()))?;

    let semaphore = Arc::new(Semaphore::new(options.max_parallel_requests));
    let mut jobs = JoinSet::new();
    for file in files.iter().cloned() {
        let client = client.clone();
        let repo = repo.clone();
        let output_dir = output_dir.to_path_buf();
        let semaphore = semaphore.clone();
        let options = options.clone();
        jobs.spawn(async move {
            download_file(&client, &repo, &file, &output_dir, &options, semaphore)
                .await
                .with_context(|| format!("downloading {}", file.path))
        });
    }

    while let Some(result) = jobs.join_next().await {
        match result {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                jobs.abort_all();
                while jobs.join_next().await.is_some() {}
                return Err(error);
            }
            Err(error) => {
                jobs.abort_all();
                while jobs.join_next().await.is_some() {}
                return Err(anyhow!("Hugging Face download task failed: {error}"));
            }
        }
    }
    Ok(())
}

async fn download_file(
    client: &hi_ai::HuggingFaceHubClient,
    repo: &hi_ai::HfRepoRef,
    file: &hi_ai::HfFileInfo,
    output_dir: &Path,
    options: &HfDownloadOptions,
    semaphore: Arc<Semaphore>,
) -> Result<()> {
    let paths = partial_paths(output_dir, &file.path)?;
    if let Some(parent) = paths.destination.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating model file directory {}", parent.display()))?;
    }
    if paths.destination.is_file() && crate::hf::cached_file_is_valid(&paths.destination, file) {
        cleanup_partial_files(&paths)?;
        return Ok(());
    }
    if paths.destination.is_file() {
        preserve_legacy_file(&paths.destination)?;
    } else if paths.destination.exists() {
        bail!(
            "Hugging Face model path is not a regular file: {}",
            paths.destination.display()
        );
    }

    let mut manifest = load_manifest(&paths, repo, file, options)?;
    if file.size.is_none() {
        manifest.mode = DownloadMode::Sequential;
        write_manifest(&paths.manifest, &manifest)?;
        return download_sequential_file(client, repo, file, &paths, manifest, semaphore).await;
    }

    if manifest.mode == DownloadMode::Sequential {
        return download_sequential_file(client, repo, file, &paths, manifest, semaphore).await;
    }

    if file.size == Some(0) {
        let file_handle = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&paths.part)
            .with_context(|| format!("creating {}", paths.part.display()))?;
        file_handle.sync_all()?;
        fs::rename(&paths.part, &paths.destination).with_context(|| {
            format!(
                "moving empty download {} into place",
                paths.destination.display()
            )
        })?;
        let _ = fs::remove_file(&paths.manifest);
        return Ok(());
    }

    let size = file.size.expect("checked above");
    let pending = manifest
        .completed_ranges
        .iter()
        .enumerate()
        .filter_map(|(index, completed)| (!completed).then_some(index))
        .collect::<Vec<_>>();
    let manifest_state = Arc::new(tokio::sync::Mutex::new(manifest));
    let mut jobs = JoinSet::new();
    for index in pending {
        let client = client.clone();
        let repo = repo.clone();
        let file = file.clone();
        let paths = paths.clone();
        let semaphore = semaphore.clone();
        let options = options.clone();
        jobs.spawn(async move {
            download_range_with_retries(
                &client,
                &repo,
                &file,
                &paths.part,
                index,
                size,
                &options,
                semaphore,
            )
            .await
        });
    }

    let mut unsupported = false;
    while let Some(result) = jobs.join_next().await {
        match result {
            Ok(Ok(range)) => {
                let mut manifest = manifest_state.lock().await;
                if let Some(etag) = range.etag {
                    merge_etag(&mut manifest.etag, etag)?;
                }
                if let Some(completed) = manifest.completed_ranges.get_mut(range.index) {
                    *completed = true;
                }
                manifest.recompute_completed_bytes();
                write_manifest(&paths.manifest, &manifest)?;
            }
            Ok(Err(RangeFailure::Unsupported)) => {
                unsupported = true;
                jobs.abort_all();
                while jobs.join_next().await.is_some() {}
                break;
            }
            Ok(Err(RangeFailure::Error(error))) => {
                jobs.abort_all();
                while jobs.join_next().await.is_some() {}
                return Err(error);
            }
            Err(error) => {
                jobs.abort_all();
                while jobs.join_next().await.is_some() {}
                return Err(anyhow!("range download task failed: {error}"));
            }
        }
    }

    if unsupported {
        let mut manifest = manifest_state.lock().await;
        manifest.mode = DownloadMode::Sequential;
        manifest.completed_ranges.fill(false);
        manifest.completed_bytes = 0;
        manifest.etag = None;
        reset_partial_file(&paths.part)?;
        write_manifest(&paths.manifest, &manifest)?;
        return download_sequential_file(client, repo, file, &paths, manifest.clone(), semaphore)
            .await;
    }

    let manifest = manifest_state.lock().await;
    if !manifest.completed_ranges.iter().all(|completed| *completed) {
        bail!("download manifest for {} is incomplete", file.path);
    }
    drop(manifest);
    finalize_partial_file(&paths, file.size)?;
    Ok(())
}

async fn download_range_with_retries(
    client: &hi_ai::HuggingFaceHubClient,
    repo: &hi_ai::HfRepoRef,
    file: &hi_ai::HfFileInfo,
    part_path: &Path,
    index: usize,
    expected_size: u64,
    options: &HfDownloadOptions,
    semaphore: Arc<Semaphore>,
) -> std::result::Result<RangeResult, RangeFailure> {
    let start = index as u64 * options.range_size;
    let end = (start + options.range_size)
        .min(expected_size)
        .saturating_sub(1);
    let expected_len = end.saturating_sub(start).saturating_add(1);
    let repo_file = repo.clone().with_filename(file.path.clone());
    let mut last_error = None;

    for attempt in 0..=options.max_retries {
        let _permit = semaphore
            .acquire()
            .await
            .map_err(|_| RangeFailure::Error(anyhow!("download semaphore closed")))?;
        let response = match client.fetch_file_range(&repo_file, start, end).await {
            Ok(response) => response,
            Err(error) => {
                last_error = Some(error);
                if attempt < options.max_retries {
                    retry_delay(attempt).await;
                    continue;
                }
                break;
            }
        };
        let status = response.status();
        if status == StatusCode::OK {
            return Err(RangeFailure::Unsupported);
        }
        if status != StatusCode::PARTIAL_CONTENT {
            let retry = retryable_status(status);
            let body = read_error_body(response).await;
            last_error = Some(anyhow!(
                "Hugging Face range {}-{} returned {}: {}",
                start,
                end,
                status,
                body
            ));
            if retry && attempt < options.max_retries {
                retry_delay(attempt).await;
                continue;
            }
            break;
        }

        if let Err(error) = validate_range_headers(&response, start, end, expected_size) {
            last_error = Some(error);
            if attempt < options.max_retries {
                retry_delay(attempt).await;
                continue;
            }
            break;
        }

        let etag = response
            .headers()
            .get(header::ETAG)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        match write_response_range(response, part_path, start, expected_len).await {
            Ok(()) => {
                return Ok(RangeResult { index, etag });
            }
            Err(error) => {
                last_error = Some(error);
                if attempt < options.max_retries {
                    retry_delay(attempt).await;
                }
            }
        }
    }

    Err(RangeFailure::Error(last_error.unwrap_or_else(|| {
        anyhow!("Hugging Face range {}-{} failed", start, end)
    })))
}

async fn write_response_range(
    response: reqwest::Response,
    part_path: &Path,
    start: u64,
    expected_len: u64,
) -> Result<()> {
    let path = part_path.to_path_buf();
    let file =
        tokio::task::spawn_blocking(move || OpenOptions::new().create(true).write(true).open(path))
            .await??;
    let mut offset = start;
    let mut received = 0u64;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("reading Hugging Face range response")?;
        let chunk_len = chunk.len() as u64;
        if received.saturating_add(chunk_len) > expected_len {
            bail!(
                "Hugging Face range response exceeded expected length: got more than {} bytes",
                expected_len
            );
        }
        write_all_at(&file, &chunk, offset)?;
        offset = offset.saturating_add(chunk_len);
        received = received.saturating_add(chunk_len);
    }
    if received != expected_len {
        bail!(
            "Hugging Face range response was truncated: expected {} bytes, received {}",
            expected_len,
            received
        );
    }
    file.sync_data().context("flushing downloaded range")?;
    Ok(())
}

async fn download_sequential_file(
    client: &hi_ai::HuggingFaceHubClient,
    repo: &hi_ai::HfRepoRef,
    file: &hi_ai::HfFileInfo,
    paths: &PartialPaths,
    mut manifest: DownloadManifest,
    semaphore: Arc<Semaphore>,
) -> Result<()> {
    let existing = fs::metadata(&paths.part)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    let known_size = file.size;
    let repo_file = repo.clone().with_filename(file.path.clone());
    let _permit = semaphore
        .acquire()
        .await
        .map_err(|_| anyhow!("download semaphore closed"))?;
    let mut response = if existing > 0 {
        match known_size {
            Some(size) if existing >= size => client.fetch_file(&repo_file).await?,
            Some(size) => {
                client
                    .fetch_file_range(&repo_file, existing, size - 1)
                    .await?
            }
            None => client.fetch_file_from_offset(&repo_file, existing).await?,
        }
    } else {
        client.fetch_file(&repo_file).await?
    };

    let mut offset = existing;
    if existing > 0 && response.status() == StatusCode::OK {
        reset_partial_file(&paths.part)?;
        manifest.completed_bytes = 0;
        manifest.etag = None;
        offset = 0;
        response = client.fetch_file(&repo_file).await?;
    }
    if !response.status().is_success() {
        let status = response.status();
        let body = read_error_body(response).await;
        bail!("Hugging Face sequential download returned {status}: {body}");
    }
    if existing > 0 && response.status() == StatusCode::PARTIAL_CONTENT {
        let (range_start, _, _) = parse_content_range_header(&response)?;
        if range_start != existing {
            bail!(
                "Hugging Face resume started at {}, expected {}",
                range_start,
                existing
            );
        }
    }
    if let Some(etag) = response
        .headers()
        .get(header::ETAG)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
    {
        merge_etag(&mut manifest.etag, etag)?;
    }

    let mut output = OpenOptions::new()
        .create(true)
        .write(true)
        .append(offset > 0)
        .truncate(offset == 0)
        .open(&paths.part)
        .with_context(|| format!("opening {}", paths.part.display()))?;
    let mut stream = response.bytes_stream();
    let mut since_manifest = 0u64;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("reading Hugging Face sequential response")?;
        output
            .write_all(&chunk)
            .with_context(|| format!("writing {}", paths.part.display()))?;
        offset = offset.saturating_add(chunk.len() as u64);
        since_manifest = since_manifest.saturating_add(chunk.len() as u64);
        manifest.completed_bytes = offset;
        if since_manifest >= PROGRESS_PERSIST_BYTES {
            write_manifest(&paths.manifest, &manifest)?;
            since_manifest = 0;
        }
        if known_size.is_some_and(|size| offset > size) {
            bail!("Hugging Face sequential response exceeded expected file size");
        }
    }
    output.sync_all()?;
    if let Some(expected) = known_size
        && offset != expected
    {
        bail!(
            "Hugging Face sequential download was incomplete: expected {} bytes, received {}",
            expected,
            offset
        );
    }
    manifest.completed_bytes = offset;
    write_manifest(&paths.manifest, &manifest)?;
    finalize_partial_file(paths, known_size)?;
    Ok(())
}

fn validate_range_headers(
    response: &reqwest::Response,
    expected_start: u64,
    expected_end: u64,
    expected_size: u64,
) -> Result<()> {
    let (start, end, total) = parse_content_range_header(response)?;
    if start != expected_start || end != expected_end || total != Some(expected_size) {
        bail!(
            "unexpected Content-Range: got bytes {}-{}/{}; expected bytes {}-{}/{}",
            start,
            end,
            total
                .map(|value| value.to_string())
                .unwrap_or_else(|| "*".to_string()),
            expected_start,
            expected_end,
            expected_size
        );
    }
    if let Some(length) = response.content_length()
        && length
            != expected_end
                .saturating_sub(expected_start)
                .saturating_add(1)
    {
        bail!(
            "unexpected ranged Content-Length: got {}, expected {}",
            length,
            expected_end
                .saturating_sub(expected_start)
                .saturating_add(1)
        );
    }
    Ok(())
}

fn parse_content_range_header(response: &reqwest::Response) -> Result<(u64, u64, Option<u64>)> {
    let raw = response
        .headers()
        .get(header::CONTENT_RANGE)
        .ok_or_else(|| anyhow!("Hugging Face range response omitted Content-Range"))?
        .to_str()
        .context("Hugging Face Content-Range was not valid UTF-8")?;
    let remainder = raw
        .strip_prefix("bytes ")
        .ok_or_else(|| anyhow!("invalid Hugging Face Content-Range '{raw}'"))?;
    let (range, total) = remainder
        .split_once('/')
        .ok_or_else(|| anyhow!("invalid Hugging Face Content-Range '{raw}'"))?;
    let (start, end) = range
        .split_once('-')
        .ok_or_else(|| anyhow!("invalid Hugging Face Content-Range '{raw}'"))?;
    let start = start.parse::<u64>()?;
    let end = end.parse::<u64>()?;
    let total = (total != "*").then(|| total.parse::<u64>()).transpose()?;
    if start > end {
        bail!("invalid Hugging Face Content-Range '{raw}'");
    }
    Ok((start, end, total))
}

fn retryable_status(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::REQUEST_TIMEOUT
            | StatusCode::TOO_MANY_REQUESTS
            | StatusCode::INTERNAL_SERVER_ERROR
            | StatusCode::BAD_GATEWAY
            | StatusCode::SERVICE_UNAVAILABLE
            | StatusCode::GATEWAY_TIMEOUT
    )
}

async fn retry_delay(attempt: usize) {
    let exponent = attempt.min(5) as u32;
    let seconds = 1u64 << exponent;
    tokio::time::sleep(Duration::from_secs(seconds.min(16))).await;
}

async fn read_error_body(mut response: reqwest::Response) -> String {
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.ok().flatten() {
        let remaining = ERROR_BODY_LIMIT.saturating_sub(body.len());
        if remaining == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
    }
    String::from_utf8_lossy(&body).trim().to_string()
}

fn merge_etag(current: &mut Option<String>, incoming: String) -> Result<()> {
    if let Some(existing) = current
        && existing != &incoming
    {
        bail!("Hugging Face changed the file ETag during download");
    }
    *current = Some(incoming);
    Ok(())
}

fn range_count(size: u64, range_size: u64) -> usize {
    if size == 0 {
        0
    } else {
        ((size - 1) / range_size + 1) as usize
    }
}

fn range_len(index: usize, size: u64, range_size: u64) -> u64 {
    let start = index as u64 * range_size;
    size.saturating_sub(start).min(range_size)
}

fn partial_paths(output_dir: &Path, relative: &str) -> Result<PartialPaths> {
    let relative_path = Path::new(relative);
    if relative_path.is_absolute()
        || relative_path
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::CurDir))
    {
        bail!("unsafe Hugging Face file path '{relative}'");
    }
    let destination = output_dir.join(relative_path);
    let part = PathBuf::from(format!("{}.hi-part", destination.display()));
    let manifest = PathBuf::from(format!("{}.json", part.display()));
    Ok(PartialPaths {
        destination,
        part,
        manifest,
    })
}

fn load_manifest(
    paths: &PartialPaths,
    repo: &hi_ai::HfRepoRef,
    file: &hi_ai::HfFileInfo,
    options: &HfDownloadOptions,
) -> Result<DownloadManifest> {
    let parsed = fs::read_to_string(&paths.manifest)
        .ok()
        .and_then(|raw| serde_json::from_str::<DownloadManifest>(&raw).ok())
        .filter(|manifest| manifest.is_compatible(repo, file, options));
    if let Some(mut manifest) = parsed
        && paths.part.is_file()
    {
        let part_len = fs::metadata(&paths.part)?.len();
        let required_len = manifest
            .completed_ranges
            .iter()
            .enumerate()
            .filter(|(_, completed)| **completed)
            .map(|(index, _)| {
                let size = file.size.unwrap_or_default();
                index as u64 * options.range_size + range_len(index, size, options.range_size)
            })
            .max()
            .unwrap_or(0);
        if file.size.is_none() || part_len >= required_len {
            manifest.recompute_completed_bytes();
            return Ok(manifest);
        }
    }

    if paths.part.exists() {
        preserve_legacy_file(&paths.part)?;
    }
    let _ = fs::remove_file(&paths.manifest);
    let manifest = DownloadManifest::new(repo, file, options);
    write_manifest(&paths.manifest, &manifest)?;
    Ok(manifest)
}

fn write_manifest(path: &Path, manifest: &DownloadManifest) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temp = PathBuf::from(format!("{}.tmp", path.display()));
    let body = serde_json::to_vec_pretty(manifest)?;
    let mut file = File::create(&temp).with_context(|| format!("creating {}", temp.display()))?;
    file.write_all(&body)?;
    file.sync_all()?;
    fs::rename(&temp, path).with_context(|| format!("installing {}", path.display()))?;
    Ok(())
}

fn reset_partial_file(path: &Path) -> Result<()> {
    if path.exists() {
        let file = OpenOptions::new().write(true).open(path)?;
        file.set_len(0)?;
        file.sync_all()?;
    } else if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
        File::create(path)?;
    }
    Ok(())
}

fn finalize_partial_file(paths: &PartialPaths, expected_size: Option<u64>) -> Result<()> {
    let metadata =
        fs::metadata(&paths.part).with_context(|| format!("reading {}", paths.part.display()))?;
    if let Some(expected_size) = expected_size
        && metadata.len() != expected_size
    {
        bail!(
            "partial file has {} bytes; expected {}",
            metadata.len(),
            expected_size
        );
    }
    let file = OpenOptions::new().write(true).open(&paths.part)?;
    file.sync_all()?;
    drop(file);
    fs::rename(&paths.part, &paths.destination).with_context(|| {
        format!(
            "moving {} into {}",
            paths.part.display(),
            paths.destination.display()
        )
    })?;
    let _ = fs::remove_file(&paths.manifest);
    Ok(())
}

fn cleanup_partial_files(paths: &PartialPaths) -> Result<()> {
    let _ = fs::remove_file(&paths.part);
    let _ = fs::remove_file(&paths.manifest);
    Ok(())
}

fn preserve_legacy_file(path: &Path) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    let mut backup = PathBuf::from(format!("{}.legacy-part", path.display()));
    let mut index = 1u32;
    while backup.exists() {
        backup = PathBuf::from(format!("{}.legacy-part.{index}", path.display()));
        index = index.saturating_add(1);
    }
    fs::rename(path, &backup).with_context(|| {
        format!(
            "preserving incomplete legacy download {} as {}",
            path.display(),
            backup.display()
        )
    })?;
    Ok(())
}

/// Return actual completed download bytes, excluding manifests and sparse
/// partial-file holes. This is the source used by the TUI progress display and
/// disk-fit calculation while a model is downloading.
pub fn download_progress_bytes(dir: &Path) -> u64 {
    fn visit(path: &Path) -> u64 {
        let Ok(entries) = fs::read_dir(path) else {
            return 0;
        };
        entries
            .flatten()
            .map(|entry| {
                let path = entry.path();
                let Ok(metadata) = entry.metadata() else {
                    return 0;
                };
                if metadata.is_dir() {
                    return visit(&path);
                }
                let name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("");
                if name.ends_with(".hi-part.json") {
                    return manifest_completed_bytes(&path);
                }
                if name.ends_with(".hi-part")
                    || name.ends_with(".aria2")
                    || name.contains(".legacy-part")
                {
                    return 0;
                }
                metadata.len()
            })
            .sum()
    }
    visit(dir)
}

/// As [`download_progress_bytes`], but only count final repository files that
/// match the Hub metadata and pass the existing content validation. This keeps
/// a stale or corrupt legacy file from making the disk-space preflight think
/// that bytes have already been downloaded.
pub fn download_progress_bytes_for_files(dir: &Path, files: &[hi_ai::HfFileInfo]) -> u64 {
    let expected = files
        .iter()
        .map(|file| (file.path.as_str(), file))
        .collect::<HashMap<_, _>>();

    fn visit(path: &Path, root: &Path, expected: &HashMap<&str, &hi_ai::HfFileInfo>) -> u64 {
        let Ok(entries) = fs::read_dir(path) else {
            return 0;
        };
        entries
            .flatten()
            .map(|entry| {
                let path = entry.path();
                let Ok(metadata) = entry.metadata() else {
                    return 0;
                };
                if metadata.is_dir() {
                    return visit(&path, root, expected);
                }
                let name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("");
                if name.ends_with(".hi-part.json") {
                    return manifest_completed_bytes(&path);
                }
                if name.ends_with(".hi-part")
                    || name.ends_with(".aria2")
                    || name.contains(".legacy-part")
                {
                    return 0;
                }
                let Ok(relative) = path.strip_prefix(root) else {
                    return 0;
                };
                let relative = relative.to_string_lossy();
                expected
                    .get(relative.as_ref())
                    .filter(|file| crate::hf::cached_file_is_valid(&path, file))
                    .map(|_| metadata.len())
                    .unwrap_or(0)
            })
            .sum()
    }

    visit(dir, dir, &expected)
}

/// Return the total size currently known for a managed repository download.
/// Complete files contribute their final size; active files contribute the
/// expected size recorded in their manifest. `None` is returned until at least
/// one file has been observed, or when a repository contains an unknown-sized
/// file that cannot support a meaningful percentage.
pub fn download_total_bytes(dir: &Path) -> Option<u64> {
    fn visit(path: &Path) -> (u64, bool, bool) {
        let Ok(entries) = fs::read_dir(path) else {
            return (0, false, false);
        };
        entries.flatten().fold((0, false, false), |acc, entry| {
            let path = entry.path();
            let Ok(metadata) = entry.metadata() else {
                return acc;
            };
            if metadata.is_dir() {
                let nested = visit(&path);
                return (acc.0 + nested.0, acc.1 || nested.1, acc.2 || nested.2);
            }
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("");
            if name.ends_with(".hi-part.json") {
                let Some(raw_destination) = path
                    .to_str()
                    .and_then(|path| path.strip_suffix(".hi-part.json"))
                else {
                    return acc;
                };
                if Path::new(raw_destination).is_file() {
                    return acc;
                }
                let Some(manifest) = fs::read_to_string(&path)
                    .ok()
                    .and_then(|raw| serde_json::from_str::<DownloadManifest>(&raw).ok())
                else {
                    return acc;
                };
                return match manifest.expected_bytes {
                    Some(expected) => (acc.0 + expected, true, acc.2),
                    None => (acc.0, acc.1, true),
                };
            }
            if name.ends_with(".hi-part")
                || name.ends_with(".aria2")
                || name.contains(".legacy-part")
            {
                return acc;
            }
            (acc.0 + metadata.len(), true, acc.2)
        })
    }

    let (total, observed, unknown) = visit(dir);
    (observed && !unknown).then_some(total)
}

#[cfg(unix)]
fn write_all_at(file: &File, mut bytes: &[u8], mut offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    while !bytes.is_empty() {
        let written = file.write_at(bytes, offset)?;
        if written == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "positional file write made no progress",
            ));
        }
        bytes = &bytes[written..];
        offset = offset.saturating_add(written as u64);
    }
    Ok(())
}

#[cfg(windows)]
fn write_all_at(file: &File, mut bytes: &[u8], mut offset: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !bytes.is_empty() {
        let written = file.seek_write(bytes, offset)?;
        if written == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "positional file write made no progress",
            ));
        }
        bytes = &bytes[written..];
        offset = offset.saturating_add(written as u64);
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn write_all_at(file: &File, bytes: &[u8], offset: u64) -> io::Result<()> {
    use std::io::{Seek, SeekFrom};
    let mut file = file.try_clone()?;
    file.seek(SeekFrom::Start(offset))?;
    file.write_all(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::Router;
    use axum::body::Body;
    use axum::extract::{Path as AxumPath, State};
    use axum::http::{HeaderMap, HeaderValue, Response, StatusCode as AxumStatusCode};
    use axum::routing::get;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use tokio::sync::oneshot;

    #[derive(Clone)]
    struct MockState {
        files: Arc<HashMap<String, Vec<u8>>>,
        active: Arc<AtomicUsize>,
        max_active: Arc<AtomicUsize>,
        requests: Arc<std::sync::Mutex<Vec<Option<String>>>>,
        fail_once: Arc<AtomicBool>,
        range_supported: bool,
        truncate: bool,
    }

    struct MockServer {
        base_url: String,
        shutdown: Option<oneshot::Sender<()>>,
        task: tokio::task::JoinHandle<()>,
    }

    impl MockServer {
        async fn stop(mut self) {
            if let Some(shutdown) = self.shutdown.take() {
                let _ = shutdown.send(());
            }
            let _ = self.task.await;
        }
    }

    async fn start_mock_server(state: MockState) -> MockServer {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock Hugging Face server");
        let address = listener.local_addr().expect("mock server address");
        let app = Router::new()
            .route("/{org}/{model}/resolve/{revision}/{file}", get(mock_file))
            .with_state(state);
        let (shutdown, signal) = oneshot::channel();
        let task = tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = signal.await;
                })
                .await
                .expect("mock Hugging Face server");
        });
        MockServer {
            base_url: format!("http://{address}"),
            shutdown: Some(shutdown),
            task,
        }
    }

    async fn mock_file(
        State(state): State<MockState>,
        AxumPath((_org, _model, _revision, file)): AxumPath<(String, String, String, String)>,
        headers: HeaderMap,
    ) -> Response<Body> {
        let current = state.active.fetch_add(1, Ordering::SeqCst) + 1;
        let mut observed = state.max_active.load(Ordering::SeqCst);
        while current > observed {
            match state.max_active.compare_exchange(
                observed,
                current,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => break,
                Err(next) => observed = next,
            }
        }
        let range = headers
            .get(axum::http::header::RANGE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        state
            .requests
            .lock()
            .expect("request log")
            .push(range.clone());
        tokio::time::sleep(Duration::from_millis(20)).await;

        let payload = state.files.get(&file).cloned().unwrap_or_default();
        let response = if state.fail_once.swap(false, Ordering::SeqCst) {
            Response::builder()
                .status(AxumStatusCode::SERVICE_UNAVAILABLE)
                .body(Body::from("try again"))
                .expect("mock failure response")
        } else if let Some(range) = range {
            let (start, end) = parse_test_range(&range).expect("test range");
            if !state.range_supported {
                Response::builder()
                    .status(AxumStatusCode::OK)
                    .header(axum::http::header::CONTENT_LENGTH, payload.len())
                    .body(Body::from(payload))
                    .expect("mock full response")
            } else {
                let end = end.min(payload.len().saturating_sub(1));
                let mut body = payload[start..=end].to_vec();
                if state.truncate && !body.is_empty() {
                    body.pop();
                }
                Response::builder()
                    .status(AxumStatusCode::PARTIAL_CONTENT)
                    .header(
                        axum::http::header::CONTENT_RANGE,
                        format!("bytes {start}-{end}/{}", payload.len()),
                    )
                    .header(
                        axum::http::header::CONTENT_LENGTH,
                        if state.truncate {
                            body.len()
                        } else {
                            end.saturating_sub(start).saturating_add(1)
                        },
                    )
                    .header(
                        axum::http::header::ETAG,
                        HeaderValue::from_static("test-etag"),
                    )
                    .body(Body::from(body))
                    .expect("mock range response")
            }
        } else {
            Response::builder()
                .status(AxumStatusCode::OK)
                .header(axum::http::header::CONTENT_LENGTH, payload.len())
                .header(
                    axum::http::header::ETAG,
                    HeaderValue::from_static("test-etag"),
                )
                .body(Body::from(payload))
                .expect("mock sequential response")
        };
        state.active.fetch_sub(1, Ordering::SeqCst);
        response
    }

    fn parse_test_range(range: &str) -> Option<(usize, usize)> {
        let range = range.strip_prefix("bytes=")?;
        let (start, end) = range.split_once('-')?;
        Some((start.parse().ok()?, end.parse().ok()?))
    }

    fn mock_state(payload: &[u8]) -> MockState {
        mock_state_with_files(HashMap::from([(
            "weights.bin".to_string(),
            payload.to_vec(),
        )]))
    }

    fn mock_state_with_files(files: HashMap<String, Vec<u8>>) -> MockState {
        MockState {
            files: Arc::new(files),
            active: Arc::new(AtomicUsize::new(0)),
            max_active: Arc::new(AtomicUsize::new(0)),
            requests: Arc::new(std::sync::Mutex::new(Vec::new())),
            fail_once: Arc::new(AtomicBool::new(false)),
            range_supported: true,
            truncate: false,
        }
    }

    fn test_file(size: usize) -> hi_ai::HfFileInfo {
        test_file_named("weights.bin", size)
    }

    fn test_file_named(path: &str, size: usize) -> hi_ai::HfFileInfo {
        hi_ai::HfFileInfo {
            path: path.to_string(),
            size: Some(size as u64),
        }
    }

    #[tokio::test]
    async fn downloads_multiple_ranges_concurrently() {
        let payload = b"0123456789abcdefghijklmn";
        let state = mock_state(payload);
        let max_active = state.max_active.clone();
        let requests = state.requests.clone();
        let server = start_mock_server(state).await;
        let client = hi_ai::HuggingFaceHubClient::new(&server.base_url, None);
        let repo = hi_ai::HfRepoRef::parse("org/model").unwrap();
        let dir = tempfile::tempdir().unwrap();
        let files = [test_file(payload.len())];
        let options = HfDownloadOptions {
            max_parallel_requests: 4,
            range_size: 4,
            max_retries: 1,
        };

        download_repo(&client, &repo, &files, dir.path(), options)
            .await
            .unwrap();

        assert_eq!(fs::read(dir.path().join("weights.bin")).unwrap(), payload);
        assert!(max_active.load(Ordering::SeqCst) >= 2);
        assert_eq!(requests.lock().unwrap().len(), 6);
        assert_eq!(download_progress_bytes(dir.path()), payload.len() as u64);
        assert_eq!(download_total_bytes(dir.path()), Some(payload.len() as u64));
        server.stop().await;
    }

    #[tokio::test]
    async fn downloads_independent_repository_files_under_one_shared_budget() {
        let first = b"abcdefgh";
        let second = b"ijklmnop";
        let state = mock_state_with_files(HashMap::from([
            ("weights.bin".to_string(), first.to_vec()),
            ("config.bin".to_string(), second.to_vec()),
        ]));
        let max_active = state.max_active.clone();
        let server = start_mock_server(state).await;
        let client = hi_ai::HuggingFaceHubClient::new(&server.base_url, None);
        let repo = hi_ai::HfRepoRef::parse("org/model").unwrap();
        let dir = tempfile::tempdir().unwrap();

        download_repo(
            &client,
            &repo,
            &[
                test_file_named("weights.bin", first.len()),
                test_file_named("config.bin", second.len()),
            ],
            dir.path(),
            HfDownloadOptions {
                max_parallel_requests: 2,
                range_size: 8,
                max_retries: 0,
            },
        )
        .await
        .unwrap();

        assert_eq!(fs::read(dir.path().join("weights.bin")).unwrap(), first);
        assert_eq!(fs::read(dir.path().join("config.bin")).unwrap(), second);
        assert!(max_active.load(Ordering::SeqCst) >= 2);
        server.stop().await;
    }

    #[tokio::test]
    async fn duplicate_downloads_for_one_cache_are_serialized() {
        let payload = b"0123456789abcdefghijklmn";
        let state = mock_state(payload);
        let requests = state.requests.clone();
        let server = start_mock_server(state).await;
        let client = hi_ai::HuggingFaceHubClient::new(&server.base_url, None);
        let repo = hi_ai::HfRepoRef::parse("org/model").unwrap();
        let dir = tempfile::tempdir().unwrap();
        let files = [test_file(payload.len())];
        let options = HfDownloadOptions {
            max_parallel_requests: 4,
            range_size: 4,
            max_retries: 0,
        };

        let first = download_repo(&client, &repo, &files, dir.path(), options.clone());
        let second = download_repo(&client, &repo, &files, dir.path(), options);
        let (first, second) = tokio::join!(first, second);
        first.unwrap();
        second.unwrap();

        assert_eq!(fs::read(dir.path().join("weights.bin")).unwrap(), payload);
        assert_eq!(requests.lock().unwrap().len(), 6);
        server.stop().await;
    }

    #[tokio::test]
    async fn resumes_only_incomplete_ranges_from_manifest() {
        let payload = b"0123456789abcdefghijklmn";
        let state = mock_state(payload);
        let requests = state.requests.clone();
        let server = start_mock_server(state).await;
        let client = hi_ai::HuggingFaceHubClient::new(&server.base_url, None);
        let repo = hi_ai::HfRepoRef::parse("org/model").unwrap();
        let dir = tempfile::tempdir().unwrap();
        let file = test_file(payload.len());
        let options = HfDownloadOptions {
            max_parallel_requests: 2,
            range_size: 4,
            max_retries: 1,
        };
        let paths = partial_paths(dir.path(), &file.path).unwrap();
        fs::create_dir_all(paths.destination.parent().unwrap()).unwrap();
        let mut manifest = DownloadManifest::new(&repo, &file, &options);
        manifest.completed_ranges[0] = true;
        manifest.recompute_completed_bytes();
        write_manifest(&paths.manifest, &manifest).unwrap();
        let part = File::create(&paths.part).unwrap();
        write_all_at(&part, &payload[..4], 0).unwrap();
        part.sync_all().unwrap();

        download_repo(&client, &repo, &[file], dir.path(), options)
            .await
            .unwrap();

        assert_eq!(fs::read(dir.path().join("weights.bin")).unwrap(), payload);
        let requests = requests.lock().unwrap();
        assert!(!requests.iter().flatten().any(|range| range == "bytes=0-3"));
        assert_eq!(requests.len(), 5);
        server.stop().await;
    }

    #[tokio::test]
    async fn falls_back_to_in_process_sequential_http_when_ranges_are_unsupported() {
        let payload = b"0123456789abcdefghijklmn";
        let mut state = mock_state(payload);
        state.range_supported = false;
        let requests = state.requests.clone();
        let server = start_mock_server(state).await;
        let client = hi_ai::HuggingFaceHubClient::new(&server.base_url, None);
        let repo = hi_ai::HfRepoRef::parse("org/model").unwrap();
        let dir = tempfile::tempdir().unwrap();

        download_repo(
            &client,
            &repo,
            &[test_file(payload.len())],
            dir.path(),
            HfDownloadOptions {
                max_parallel_requests: 4,
                range_size: 4,
                max_retries: 0,
            },
        )
        .await
        .unwrap();

        assert_eq!(fs::read(dir.path().join("weights.bin")).unwrap(), payload);
        let requests = requests.lock().unwrap();
        assert!(requests.iter().flatten().any(|range| range == "bytes=0-3"));
        assert!(requests.iter().any(Option::is_none));
        server.stop().await;
    }

    #[tokio::test]
    async fn retries_transient_range_failures() {
        let payload = b"01234567";
        let state = mock_state(payload);
        state.fail_once.store(true, Ordering::SeqCst);
        let requests = state.requests.clone();
        let server = start_mock_server(state).await;
        let client = hi_ai::HuggingFaceHubClient::new(&server.base_url, None);
        let repo = hi_ai::HfRepoRef::parse("org/model").unwrap();
        let dir = tempfile::tempdir().unwrap();

        download_repo(
            &client,
            &repo,
            &[test_file(payload.len())],
            dir.path(),
            HfDownloadOptions {
                max_parallel_requests: 1,
                range_size: 8,
                max_retries: 2,
            },
        )
        .await
        .unwrap();

        assert_eq!(requests.lock().unwrap().len(), 2);
        server.stop().await;
    }

    #[tokio::test]
    async fn rejects_truncated_ranges_without_exposing_final_file() {
        let payload = b"01234567";
        let mut state = mock_state(payload);
        state.truncate = true;
        let server = start_mock_server(state).await;
        let client = hi_ai::HuggingFaceHubClient::new(&server.base_url, None);
        let repo = hi_ai::HfRepoRef::parse("org/model").unwrap();
        let dir = tempfile::tempdir().unwrap();

        let result = download_repo(
            &client,
            &repo,
            &[test_file(payload.len())],
            dir.path(),
            HfDownloadOptions {
                max_parallel_requests: 1,
                range_size: 8,
                max_retries: 0,
            },
        )
        .await;

        assert!(result.is_err());
        assert!(!dir.path().join("weights.bin").exists());
        assert!(dir.path().join("weights.bin.hi-part.json").exists());
        server.stop().await;
    }

    #[tokio::test]
    async fn incompatible_manifest_is_discarded_and_legacy_partial_is_preserved() {
        let payload = b"01234567";
        let state = mock_state(payload);
        let server = start_mock_server(state).await;
        let client = hi_ai::HuggingFaceHubClient::new(&server.base_url, None);
        let repo = hi_ai::HfRepoRef::parse("org/model").unwrap();
        let dir = tempfile::tempdir().unwrap();
        let file = test_file(payload.len());
        let options = HfDownloadOptions {
            max_parallel_requests: 1,
            range_size: 4,
            max_retries: 0,
        };
        let paths = partial_paths(dir.path(), &file.path).unwrap();
        fs::create_dir_all(paths.destination.parent().unwrap()).unwrap();
        fs::write(&paths.part, b"legacy").unwrap();
        let incompatible = DownloadManifest {
            range_size: 2,
            ..DownloadManifest::new(&repo, &file, &options)
        };
        write_manifest(&paths.manifest, &incompatible).unwrap();

        download_repo(&client, &repo, &[file], dir.path(), options)
            .await
            .unwrap();

        assert!(dir.path().join("weights.bin.hi-part.legacy-part").exists());
        assert_eq!(fs::read(dir.path().join("weights.bin")).unwrap(), payload);
        server.stop().await;
    }

    #[test]
    fn progress_uses_manifest_bytes_not_partial_file_length() {
        let dir = tempfile::tempdir().unwrap();
        let repo = hi_ai::HfRepoRef::parse("org/model").unwrap();
        let file = test_file(16);
        let options = HfDownloadOptions {
            range_size: 4,
            ..Default::default()
        };
        let paths = partial_paths(dir.path(), &file.path).unwrap();
        fs::create_dir_all(paths.destination.parent().unwrap()).unwrap();
        let mut manifest = DownloadManifest::new(&repo, &file, &options);
        manifest.completed_ranges[0] = true;
        manifest.completed_ranges[2] = true;
        manifest.recompute_completed_bytes();
        write_manifest(&paths.manifest, &manifest).unwrap();
        fs::write(&paths.part, vec![0u8; 16]).unwrap();
        fs::write(dir.path().join("config.json"), b"{}{}").unwrap();

        assert_eq!(download_progress_bytes(dir.path()), 12);
    }

    #[test]
    fn progress_does_not_double_count_manifest_after_atomic_rename() {
        let dir = tempfile::tempdir().unwrap();
        let repo = hi_ai::HfRepoRef::parse("org/model").unwrap();
        let file = test_file(4);
        let options = HfDownloadOptions {
            range_size: 4,
            ..Default::default()
        };
        let paths = partial_paths(dir.path(), &file.path).unwrap();
        fs::write(&paths.destination, b"done").unwrap();
        let mut manifest = DownloadManifest::new(&repo, &file, &options);
        manifest.completed_ranges[0] = true;
        manifest.recompute_completed_bytes();
        write_manifest(&paths.manifest, &manifest).unwrap();

        assert_eq!(download_progress_bytes(dir.path()), 4);
        assert_eq!(download_total_bytes(dir.path()), Some(4));
    }
}
