use anyhow::{Context, Result, anyhow, bail};
use serde_json::json;
use std::io::Read;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::LazyLock;
use std::time::Duration;

// Local-inference commands are intentionally outside the core 0.2 runtime
// migration. Keep their detached processes isolated from agent-owned tool
// registries so they cannot be polled or killed through a workspace runtime.
static HF_BACKGROUND: LazyLock<crate::BackgroundRegistry> =
    LazyLock::new(crate::BackgroundRegistry::default);

fn hf_root() -> Result<PathBuf> {
    std::env::current_dir().map_err(Into::into)
}

fn spawn_hf_background(command: &str) -> Result<String> {
    let runner = crate::ProcessRunner::new(hf_root()?)?;
    HF_BACKGROUND.spawn(&runner, command)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WholeRepoMode {
    DeleteAfterEach,
    Keep,
}

#[derive(Default)]
pub struct HfCommandState {
    menu_author: Option<String>,
    menu: Vec<hi_ai::ModelCandidate>,
    last_files_repo: Option<String>,
    last_files: Vec<hi_ai::HfFileInfo>,
}

#[derive(Clone, Debug)]
pub enum HfCommandResult {
    Text(String),
    MlxReady(HfMlxRun),
}

impl HfCommandResult {
    pub fn into_text(self) -> String {
        match self {
            Self::Text(text) => text,
            Self::MlxReady(run) => run.message,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HfMlxRun {
    pub profile_name: String,
    pub repo: String,
    pub model_id: String,
    pub model_dir: PathBuf,
    pub base_url: String,
    pub process_id: String,
    pub host: String,
    pub port: u16,
    pub message: String,
}

#[derive(Clone, Debug)]
struct RepoFiles {
    repo: hi_ai::HfRepoRef,
    files: Vec<hi_ai::HfFileInfo>,
}

pub async fn handle_hf_command(arg: &str, state: &mut HfCommandState) -> Result<String> {
    Ok(handle_hf_command_result(arg, state).await?.into_text())
}

pub async fn handle_hf_command_result(
    arg: &str,
    state: &mut HfCommandState,
) -> Result<HfCommandResult> {
    let arg = arg.trim();
    let mut parts = arg.split_whitespace();
    let Some(subcommand) = parts.next() else {
        return Ok(HfCommandResult::Text(hf_usage()));
    };
    match subcommand {
        "search" => {
            let query = arg.strip_prefix("search").unwrap_or("").trim();
            if query.is_empty() {
                bail!("usage: /hf search <query>");
            }
            let client = hi_ai::HuggingFaceHubClient::from_env();
            let models = client.search_models(query, 10).await?;
            Ok(HfCommandResult::Text(format_hf_search(&models)))
        }
        "author" | "user" | "menu" => {
            let author = parts
                .next()
                .ok_or_else(|| anyhow!("usage: /hf menu <author> [limit]"))?;
            let limit = parts
                .next()
                .map(|s| s.parse::<usize>())
                .transpose()
                .map_err(|_| anyhow!("limit must be a positive integer"))?
                .unwrap_or(100);
            let client = hi_ai::HuggingFaceHubClient::from_env();
            let models = client.author_models(author, limit).await?;
            state.menu_author = Some(author.to_string());
            state.menu = models;
            state.last_files_repo = None;
            state.last_files.clear();
            Ok(HfCommandResult::Text(format_hf_author_menu(
                author,
                &state.menu,
            )))
        }
        "files" => {
            let repo_arg = parts
                .next()
                .ok_or_else(|| anyhow!("usage: /hf files <repo[@revision]|menu-number>"))?;
            let repo_source = state.resolve_repo(repo_arg)?;
            let client = hi_ai::HuggingFaceHubClient::from_env();
            let (repo, files) = fetch_files(&client, &repo_source).await?;
            state.last_files_repo = Some(repo.repo_id.clone());
            state.last_files = files;
            Ok(HfCommandResult::Text(format_hf_files(
                &repo,
                &state.last_files,
                Some(repo_arg),
            )))
        }
        "download" => {
            let repo_arg = parts.next().ok_or_else(|| anyhow!(download_usage()))?;
            let file_arg = parts.next();
            let output = parts.next();

            if let Some(mode) = whole_repo_mode(Some(repo_arg)) {
                let Some(author) = state.menu_author.clone() else {
                    bail!("usage: run /hf menu <author> first, then /hf download {repo_arg} [dir]");
                };
                if state.menu.is_empty() {
                    return Ok(HfCommandResult::Text(format!(
                        "No Hugging Face models are loaded for author {author}.\n"
                    )));
                }
                let client = hi_ai::HuggingFaceHubClient::from_env();
                return start_author_download(&client, &author, &state.menu, file_arg, mode)
                    .await
                    .map(HfCommandResult::Text);
            }

            if let Some(mode) = whole_repo_mode(file_arg)
                && !repo_arg.contains('/')
                && !is_positive_index(repo_arg)
            {
                let author = repo_arg;
                let client = hi_ai::HuggingFaceHubClient::from_env();
                let models = client.author_models(author, usize::MAX).await?;
                state.menu_author = Some(author.to_string());
                state.menu = models;
                state.last_files_repo = None;
                state.last_files.clear();
                if state.menu.is_empty() {
                    return Ok(HfCommandResult::Text(format!(
                        "No Hugging Face models found for author {author}.\n"
                    )));
                }
                return start_author_download(&client, author, &state.menu, output, mode)
                    .await
                    .map(HfCommandResult::Text);
            }

            if repo_arg.contains(':')
                && !is_positive_index(repo_arg)
                && whole_repo_mode(file_arg).is_none()
            {
                let out = run_download(repo_arg.to_string(), file_arg).await?;
                return Ok(HfCommandResult::Text(format!("{}\n", out.content)));
            }

            let repo_source = state.resolve_repo(repo_arg)?;
            let Some(file_arg) = file_arg else {
                if is_positive_index(repo_arg) {
                    let client = hi_ai::HuggingFaceHubClient::from_env();
                    let (repo, files) = fetch_files(&client, &repo_source).await?;
                    state.last_files_repo = Some(repo.repo_id.clone());
                    state.last_files = files;
                    return Ok(HfCommandResult::Text(format_hf_files(
                        &repo,
                        &state.last_files,
                        Some(repo_arg),
                    )));
                }
                bail!(download_usage());
            };

            if let Some(mode) = whole_repo_mode(Some(file_arg)) {
                let client = hi_ai::HuggingFaceHubClient::from_env();
                let (repo, files) = fetch_files(&client, &repo_source).await?;
                state.last_files_repo = Some(repo.repo_id.clone());
                state.last_files = files;
                return start_all_download(&client, &repo, &state.last_files, output, mode)
                    .map(HfCommandResult::Text);
            }

            let filename = if let Some(index) = parse_positive_index(file_arg) {
                let client = hi_ai::HuggingFaceHubClient::from_env();
                let repo_id = repo_id_for_source(&repo_source)?;
                if state.last_files_repo.as_deref() != Some(repo_id.as_str())
                    || state.last_files.is_empty()
                {
                    let (repo, files) = fetch_files(&client, &repo_source).await?;
                    state.last_files_repo = Some(repo.repo_id.clone());
                    state.last_files = files;
                }
                state
                    .last_files
                    .get(index - 1)
                    .map(|file| file.path.clone())
                    .ok_or_else(|| {
                        anyhow!(
                            "file number {index} is out of range for {repo_source}; use /hf files {repo_arg}"
                        )
                    })?
            } else {
                file_arg.to_string()
            };

            let source = format!("{repo_source}:{filename}");
            let out = run_download(source, output).await?;
            Ok(HfCommandResult::Text(format!("{}\n", out.content)))
        }
        "run" => start_mlx_run(arg.strip_prefix("run").unwrap_or("").trim(), state).await,
        _ => Ok(HfCommandResult::Text(hf_usage())),
    }
}

/// [`download_repo_keep_foreground`], but with every byte of downloader
/// output captured instead of written to the terminal. For frontends that
/// own the screen (the TUI runs in raw mode on the alternate screen): a
/// background model fetch must never paint over the UI. On failure the
/// captured tail is folded into the error for diagnosis.
pub async fn download_repo_keep_quiet(
    repo_source: &str,
    output_dir: impl AsRef<Path>,
) -> Result<()> {
    let client = hi_ai::HuggingFaceHubClient::from_env();
    let repo = hi_ai::HfRepoRef::parse(repo_source)?;
    let files = client.list_files(&repo).await?;
    if files.is_empty() {
        bail!("no files found in {}@{}", repo.repo_id, repo.revision);
    }
    let output_dir = output_dir.as_ref();
    repair_cached_repo_files(output_dir, &files)?;
    let urls = resolve_repo_file_urls(&client, &repo, &files).await?;
    let expected_bytes = files.iter().filter_map(|file| file.size).sum::<u64>();
    if let Some(free_bytes) = available_space_bytes(output_dir)
        && expected_bytes > 0
    {
        let existing_bytes = directory_file_bytes(output_dir);
        let required = expected_bytes.saturating_sub(existing_bytes);
        // Keep a modest buffer for temporary downloader files and the model's
        // metadata. This prevents a multi-GB MLX fetch from filling the disk
        // and leaving the user's workspace unusable.
        const HEADROOM: u64 = 512 * 1024 * 1024;
        if free_bytes < required.saturating_add(HEADROOM) {
            bail!(
                "not enough disk space for {} (needs about {:.1} GiB plus 512 MiB; only {:.1} GiB is free)",
                repo.repo_id,
                required as f64 / (1024.0 * 1024.0 * 1024.0),
                free_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
            );
        }
    }
    let command = all_download_command_with_urls(
        &repo,
        &files,
        &urls,
        output_dir,
        WholeRepoMode::Keep,
        None,
    )?;
    let output = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(&command)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await?;
    if !output.status.success() {
        let mut tail = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if tail.is_empty() {
            tail = String::from_utf8_lossy(&output.stdout).trim().to_string();
        }
        let tail: String = tail
            .chars()
            .rev()
            .take(600)
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        bail!(
            "download failed for {}@{} into {} ({}): {tail}",
            repo.repo_id,
            repo.revision,
            output_dir.display(),
            output.status
        );
    }
    Ok(())
}

pub async fn download_repo_keep_foreground(
    repo_source: &str,
    output_dir: impl AsRef<Path>,
) -> Result<String> {
    let client = hi_ai::HuggingFaceHubClient::from_env();
    let repo = hi_ai::HfRepoRef::parse(repo_source)?;
    let files = client.list_files(&repo).await?;
    if files.is_empty() {
        return Ok(format!(
            "No files found in {}@{}.\n",
            repo.repo_id, repo.revision
        ));
    }
    let output_dir = output_dir.as_ref();
    repair_cached_repo_files(output_dir, &files)?;
    let urls = resolve_repo_file_urls(&client, &repo, &files).await?;
    let command = all_download_command_with_urls(
        &repo,
        &files,
        &urls,
        output_dir,
        WholeRepoMode::Keep,
        None,
    )?;
    let status = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(&command)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .await?;
    if !status.success() {
        bail!(
            "download failed for {}@{} into {} with status {}",
            repo.repo_id,
            repo.revision,
            output_dir.display(),
            status
        );
    }
    Ok(format!(
        "Downloaded {} file(s) from {}@{} to {}\n",
        files.len(),
        repo.repo_id,
        repo.revision,
        output_dir.display()
    ))
}

fn hf_usage() -> String {
    "usage: /hf search <query> | /hf menu <author> [limit] | /hf files <repo|number> | /hf download <repo|number> <filename|file-number|--|--keep> [output] | /hf download <--|--keep> [dir] | /hf download <author> <--|--keep> [dir] | /hf run <repo|number> --mlx [--host 127.0.0.1] [--port 8080]\n".to_string()
}

fn download_usage() -> String {
    "usage: /hf download <repo[@revision]|menu-number> <filename|file-number|--|--keep> [output] | /hf download <--|--keep> [dir] | /hf download <author> <--|--keep> [dir]"
        .to_string()
}

async fn start_mlx_run(arg: &str, state: &HfCommandState) -> Result<HfCommandResult> {
    let mut parts = arg.split_whitespace();
    let repo_arg = parts
        .next()
        .ok_or_else(|| anyhow!("usage: /hf run <repo[@revision]|menu-number> --mlx"))?;
    let mut mlx = false;
    let mut host = "127.0.0.1".to_string();
    let mut port: Option<u16> = None;
    while let Some(part) = parts.next() {
        match part {
            "--mlx" => mlx = true,
            "--host" => {
                host = parts
                    .next()
                    .ok_or_else(|| anyhow!("--host requires a value"))?
                    .to_string();
            }
            "--port" => {
                let raw = parts
                    .next()
                    .ok_or_else(|| anyhow!("--port requires a value"))?;
                port = Some(
                    raw.parse::<u16>()
                        .map_err(|_| anyhow!("--port must be a number between 1 and 65535"))?,
                );
            }
            other => bail!("unknown /hf run option '{other}'"),
        }
    }
    if !mlx {
        bail!("usage: /hf run <repo[@revision]|menu-number> --mlx");
    }

    let repo_source = state.resolve_repo(repo_arg)?;
    let repo = hi_ai::HfRepoRef::parse(&repo_source)?;
    let model_dir = mlx_model_dir(&repo);
    let download_source = if repo.revision == "main" {
        repo.repo_id.clone()
    } else {
        format!("{}@{}", repo.repo_id, repo.revision)
    };
    if !mlx_model_present(&model_dir) {
        download_repo_keep_quiet(&download_source, &model_dir).await?;
        if !mlx_model_present(&model_dir) {
            bail!(
                "downloaded {} but its MLX weights are still incomplete under {}",
                download_source,
                model_dir.display()
            );
        }
    }

    let port = match port {
        Some(port) => port,
        None => find_available_port(8080)?,
    };
    let model_id = if repo.revision == "main" {
        repo.repo_id.clone()
    } else {
        format!("{}@{}", repo.repo_id, repo.revision)
    };
    let profile_name = format!("mlx-{}", safe_path(&repo.repo_id).to_ascii_lowercase());
    let base_url = format!("http://{host}:{port}/v1");
    let sidecar = find_hi_local_executable();
    let args = vec![
        "serve".to_string(),
        model_dir.to_string_lossy().into_owned(),
        "--backend".to_string(),
        "mlx".to_string(),
        "--host".to_string(),
        host.clone(),
        "--port".to_string(),
        port.to_string(),
        "--model-id".to_string(),
        model_id.clone(),
    ];
    let process_id = crate::spawn_local_server(&sidecar, &args)?;
    if let Err(err) =
        crate::await_local_server_health(&process_id, &host, port, Duration::from_secs(600)).await
    {
        bail!("hi-local did not become healthy at {base_url}: {err}");
    }
    if let Err(error) = crate::verify_local_server(&base_url, &model_id).await {
        crate::stop_local_server(&process_id);
        return Err(error).context("verifying the local MLX runtime");
    }
    let message = format!(
        "Started hi-local MLX for {model_id}.\n→ {base_url}\nManaged process `{process_id}`.\n"
    );
    Ok(HfCommandResult::MlxReady(HfMlxRun {
        profile_name,
        repo: repo.repo_id,
        model_id,
        model_dir,
        base_url,
        process_id,
        host,
        port,
        message,
    }))
}

fn mlx_model_dir(repo: &hi_ai::HfRepoRef) -> PathBuf {
    let root = std::env::var_os("HI_MLX_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".hi").join("models"));
    let mut name = safe_path(&repo.repo_id);
    if repo.revision != "main" {
        name.push('@');
        name.push_str(&safe_path(&repo.revision));
    }
    root.join(name)
}

/// Whether a cached MLX directory contains usable metadata and safetensors.
/// A Hugging Face `/resolve` response can be a small redirect body when a
/// downloader is pointed at the unresolved URL; checking only file existence
/// would incorrectly treat those text bodies as a downloaded model.
pub fn mlx_model_present(dir: &Path) -> bool {
    if has_aria2_remnants(dir) || !valid_json_object_file(&dir.join("config.json")) {
        return false;
    }
    let index = dir.join("model.safetensors.index.json");
    if index.is_file() {
        let Some(files) = indexed_weight_files(&index) else {
            return false;
        };
        return files
            .iter()
            .all(|file| valid_safetensors_file(&dir.join(file)));
    }
    std::fs::read_dir(dir).is_ok_and(|entries| {
        entries.flatten().any(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|ext| ext == "safetensors")
                && valid_safetensors_file(&entry.path())
        })
    })
}

fn has_aria2_remnants(dir: &Path) -> bool {
    std::fs::read_dir(dir).is_ok_and(|entries| {
        entries
            .flatten()
            .any(|entry| entry.path().extension().is_some_and(|ext| ext == "aria2"))
    })
}

fn valid_json_object_file(path: &Path) -> bool {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .is_some_and(|value| value.is_object())
}

fn indexed_weight_files(path: &Path) -> Option<Vec<String>> {
    let raw = std::fs::read_to_string(path).ok()?;
    let value = serde_json::from_str::<serde_json::Value>(&raw).ok()?;
    let map = value.get("weight_map")?.as_object()?;
    let mut files = map
        .values()
        .filter_map(|name| name.as_str().map(str::to_string))
        .collect::<Vec<_>>();
    if files.is_empty() {
        return None;
    }
    files.sort();
    files.dedup();
    Some(files)
}

fn valid_safetensors_file(path: &Path) -> bool {
    const MAX_HEADER_BYTES: u64 = 128 * 1024 * 1024;
    let Ok(metadata) = std::fs::metadata(path) else {
        return false;
    };
    if !metadata.is_file() || metadata.len() < 8 {
        return false;
    }
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    let mut length = [0u8; 8];
    if file.read_exact(&mut length).is_err() {
        return false;
    }
    let header_bytes = u64::from_le_bytes(length);
    if header_bytes == 0
        || header_bytes > MAX_HEADER_BYTES
        || header_bytes > metadata.len().saturating_sub(8)
    {
        return false;
    }
    let mut header = vec![0u8; header_bytes as usize];
    file.read_exact(&mut header).is_ok_and(|_| {
        serde_json::from_slice::<serde_json::Value>(&header).is_ok_and(|value| value.is_object())
    })
}

/// Remove only listed repo files that are clearly stale or corrupt, allowing
/// the next resumable download to start from a clean file instead of appending
/// real bytes to a saved redirect/error body.
fn repair_cached_repo_files(output_dir: &Path, files: &[hi_ai::HfFileInfo]) -> Result<()> {
    for file in files {
        let path = output_dir.join(&file.path);
        if !path.is_file() || cached_file_is_valid(&path, file) {
            continue;
        }
        std::fs::remove_file(&path)
            .with_context(|| format!("removing invalid cached file {}", path.display()))?;
        let control = PathBuf::from(format!("{}.aria2", path.display()));
        let _ = std::fs::remove_file(control);
    }
    Ok(())
}

fn cached_file_is_valid(path: &Path, file: &hi_ai::HfFileInfo) -> bool {
    if file
        .size
        .is_some_and(|expected| std::fs::metadata(path).is_ok_and(|meta| meta.len() != expected))
    {
        return false;
    }
    match path.file_name().and_then(|name| name.to_str()) {
        Some("config.json") => valid_json_object_file(path),
        Some("model.safetensors.index.json") => indexed_weight_files(path).is_some(),
        _ if path.extension().is_some_and(|ext| ext == "safetensors") => {
            valid_safetensors_file(path)
        }
        _ => true,
    }
}

async fn resolve_repo_file_urls(
    client: &hi_ai::HuggingFaceHubClient,
    repo: &hi_ai::HfRepoRef,
    files: &[hi_ai::HfFileInfo],
) -> Result<Vec<String>> {
    let mut urls = Vec::with_capacity(files.len());
    for file in files {
        let raw = client.resolve_file_url(&repo.clone().with_filename(file.path.clone()))?;
        let resolved = crate::web::resolve_download_redirects(&raw)
            .await
            .with_context(|| format!("resolving Hugging Face download URL for {}", file.path))?;
        urls.push(resolved);
    }
    Ok(urls)
}

#[cfg(test)]
fn health_ready(body: &serde_json::Value) -> bool {
    body.get("ready").and_then(serde_json::Value::as_bool) == Some(true)
}

fn find_available_port(start: u16) -> Result<u16> {
    for port in start..=u16::MAX {
        if TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return Ok(port);
        }
    }
    bail!("no available localhost port found starting at {start}")
}

fn find_hi_local_executable() -> PathBuf {
    if let Some(path) = std::env::var_os("HI_LOCAL_BIN") {
        return PathBuf::from(path);
    }
    if let Ok(current) = std::env::current_exe()
        && let Some(dir) = current.parent()
    {
        let sibling = dir.join(format!("hi-local{}", std::env::consts::EXE_SUFFIX));
        if sibling.exists() {
            return sibling;
        }
    }
    PathBuf::from("hi-local")
}

async fn run_download(source: String, output: Option<&str>) -> Result<crate::ToolOutcome> {
    let mut args = json!({ "source": source });
    if let Some(output) = output {
        args["output"] = serde_json::Value::String(output.to_string());
    }
    crate::web::run_web_download_in(&hf_root()?, &HF_BACKGROUND, &args.to_string()).await
}

async fn fetch_files(
    client: &hi_ai::HuggingFaceHubClient,
    repo_source: &str,
) -> Result<(hi_ai::HfRepoRef, Vec<hi_ai::HfFileInfo>)> {
    let repo = hi_ai::HfRepoRef::parse(repo_source)?;
    let files = client.list_files(&repo).await?;
    Ok((repo, files))
}

fn start_all_download(
    client: &hi_ai::HuggingFaceHubClient,
    repo: &hi_ai::HfRepoRef,
    files: &[hi_ai::HfFileInfo],
    output_dir: Option<&str>,
    mode: WholeRepoMode,
) -> Result<String> {
    if files.is_empty() {
        return Ok(format!(
            "No files found in {}@{}.\n",
            repo.repo_id, repo.revision
        ));
    }
    let dir = output_dir.map(PathBuf::from).unwrap_or_else(|| {
        std::env::temp_dir().join(format!("hi-hf-{}", safe_path(&repo.repo_id)))
    });
    let command = all_download_command(client, repo, files, &dir, mode)?;
    let id = spawn_hf_background(&command)?;
    let mode_text = match mode {
        WholeRepoMode::DeleteAfterEach => {
            "Each file is deleted after it finishes, so disk use stays bounded to one artifact plus aria2c metadata."
        }
        WholeRepoMode::Keep => {
            "Files are kept under the destination directory, preserving nested Hugging Face paths."
        }
    };
    Ok(format!(
        "Downloading all {} file(s) from {}@{} sequentially.\n→ {}\n{mode_text}\nStarted download ({id}). Use bash_output with id {id} for progress; bash_kill with id {id} to stop.\n",
        files.len(),
        repo.repo_id,
        repo.revision,
        dir.display()
    ))
}

async fn start_author_download(
    client: &hi_ai::HuggingFaceHubClient,
    author: &str,
    models: &[hi_ai::ModelCandidate],
    output_dir: Option<&str>,
    mode: WholeRepoMode,
) -> Result<String> {
    let repos = fetch_author_files(client, models).await?;
    let total_files = repos.iter().map(|repo| repo.files.len()).sum::<usize>();
    if total_files == 0 {
        return Ok(format!(
            "No files found for author {author} across {} repo(s).\n",
            repos.len()
        ));
    }
    let dir = output_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join(format!("hi-hf-{}-all", safe_path(author))));
    let command = all_author_download_command(client, author, &repos, &dir, mode)?;
    let id = spawn_hf_background(&command)?;
    let mode_text = match mode {
        WholeRepoMode::DeleteAfterEach => {
            "Each file is deleted after it finishes, so disk use stays bounded to one artifact plus aria2c metadata."
        }
        WholeRepoMode::Keep => {
            "Files are kept under the destination directory, preserving each repo and nested Hugging Face paths."
        }
    };
    Ok(format!(
        "Downloading all {total_files} file(s) from {} repo(s) in /hf menu {author} sequentially.\n→ {}\n{mode_text}\nStarted download ({id}). Use bash_output with id {id} for progress; bash_kill with id {id} to stop.\n",
        repos.len(),
        dir.display()
    ))
}

async fn fetch_author_files(
    client: &hi_ai::HuggingFaceHubClient,
    models: &[hi_ai::ModelCandidate],
) -> Result<Vec<RepoFiles>> {
    let mut repos = Vec::with_capacity(models.len());
    for model in models {
        let repo = hi_ai::HfRepoRef::parse(&model.id)?;
        let files = client.list_files(&repo).await?;
        repos.push(RepoFiles { repo, files });
    }
    Ok(repos)
}

fn all_download_command(
    client: &hi_ai::HuggingFaceHubClient,
    repo: &hi_ai::HfRepoRef,
    files: &[hi_ai::HfFileInfo],
    output_dir: &Path,
    mode: WholeRepoMode,
) -> Result<String> {
    all_download_command_with_availability(client, repo, files, output_dir, mode, None)
}

fn all_download_command_with_availability(
    client: &hi_ai::HuggingFaceHubClient,
    repo: &hi_ai::HfRepoRef,
    files: &[hi_ai::HfFileInfo],
    output_dir: &Path,
    mode: WholeRepoMode,
    aria2c_available: Option<bool>,
) -> Result<String> {
    let urls = files
        .iter()
        .map(|file| client.resolve_file_url(&repo.clone().with_filename(file.path.clone())))
        .collect::<Result<Vec<_>>>()?;
    all_download_command_with_urls(repo, files, &urls, output_dir, mode, aria2c_available)
}

fn all_download_command_with_urls(
    repo: &hi_ai::HfRepoRef,
    files: &[hi_ai::HfFileInfo],
    urls: &[String],
    output_dir: &Path,
    mode: WholeRepoMode,
    aria2c_available: Option<bool>,
) -> Result<String> {
    if files.len() != urls.len() {
        bail!(
            "download URL count ({}) does not match file count ({})",
            urls.len(),
            files.len()
        );
    }
    let dir = output_dir.to_string_lossy();
    let mut command = format!(
        "set -u\nmkdir -p {dir}\nprintf '%s\\n' {start}\n",
        dir = crate::web::shell_quote(&dir),
        start = crate::web::shell_quote(&format!(
            "starting {} file(s) from {}@{}",
            files.len(),
            repo.repo_id,
            repo.revision
        )),
    );
    for (idx, file) in files.iter().enumerate() {
        let output = match mode {
            WholeRepoMode::DeleteAfterEach => {
                output_dir.join(format!("{:05}.{}", idx + 1, safe_path(&file.path)))
            }
            WholeRepoMode::Keep => output_dir.join(&file.path),
        };
        append_download_step(
            &mut command,
            &urls[idx],
            &output,
            format!("{} / {} {}", idx + 1, files.len(), file.path),
            format!("ok {} {}", idx + 1, file.path),
            format!("failed {} {}", idx + 1, file.path),
            mode,
            aria2c_available,
        )?;
    }
    command.push_str(&format!(
        "printf '%s\\n' {done}\n",
        done = crate::web::shell_quote(&format!(
            "completed {} file(s) from {}@{}",
            files.len(),
            repo.repo_id,
            repo.revision
        )),
    ));
    Ok(command)
}

fn all_author_download_command(
    client: &hi_ai::HuggingFaceHubClient,
    author: &str,
    repos: &[RepoFiles],
    output_dir: &Path,
    mode: WholeRepoMode,
) -> Result<String> {
    let repo_count = repos.len();
    let total_files = repos.iter().map(|repo| repo.files.len()).sum::<usize>();
    let dir = output_dir.to_string_lossy();
    let mut command = format!(
        "set -u\nmkdir -p {dir}\nprintf '%s\\n' {start}\n",
        dir = crate::web::shell_quote(&dir),
        start = crate::web::shell_quote(&format!(
            "starting {total_files} file(s) from {repo_count} repo(s) for {author}"
        )),
    );
    for (repo_idx, repo_files) in repos.iter().enumerate() {
        command.push_str(&format!(
            "printf '%s\\n' {repo_start}\n",
            repo_start = crate::web::shell_quote(&format!(
                "repo {} / {} {}@{} ({} file(s))",
                repo_idx + 1,
                repo_count,
                repo_files.repo.repo_id,
                repo_files.repo.revision,
                repo_files.files.len()
            )),
        ));
        for (file_idx, file) in repo_files.files.iter().enumerate() {
            let output = match mode {
                WholeRepoMode::DeleteAfterEach => output_dir
                    .join(safe_path(&repo_files.repo.repo_id))
                    .join(format!("{:05}.{}", file_idx + 1, safe_path(&file.path))),
                WholeRepoMode::Keep => output_dir.join(&repo_files.repo.repo_id).join(&file.path),
            };
            let url = client
                .resolve_file_url(&repo_files.repo.clone().with_filename(file.path.clone()))?;
            append_download_step(
                &mut command,
                &url,
                &output,
                format!(
                    "repo {} / {} file {} / {} {}:{}",
                    repo_idx + 1,
                    repo_count,
                    file_idx + 1,
                    repo_files.files.len(),
                    repo_files.repo.repo_id,
                    file.path
                ),
                format!("ok {} {}", repo_files.repo.repo_id, file.path),
                format!("failed {} {}", repo_files.repo.repo_id, file.path),
                mode,
                None,
            )?;
        }
    }
    command.push_str(&format!(
        "printf '%s\\n' {done}\n",
        done = crate::web::shell_quote(&format!(
            "completed {total_files} file(s) from {repo_count} repo(s) for {author}"
        )),
    ));
    Ok(command)
}

#[allow(clippy::too_many_arguments)]
fn append_download_step(
    command: &mut String,
    url: &str,
    output: &Path,
    progress: String,
    ok: String,
    failed: String,
    mode: WholeRepoMode,
    aria2c_available: Option<bool>,
) -> Result<()> {
    let output = output.to_string_lossy().to_string();
    let download = match aria2c_available {
        Some(available) => crate::web::download_command_with_availability(url, &output, available),
        None => crate::web::download_command(url, &output),
    };
    let cleanup = match mode {
        WholeRepoMode::DeleteAfterEach => {
            format!(
                "rm -f {} {}; ",
                crate::web::shell_quote(&output),
                crate::web::shell_quote(&format!("{output}.aria2"))
            )
        }
        WholeRepoMode::Keep => String::new(),
    };
    command.push_str(&format!(
        "printf '%s\\n' {progress}\n\
         mkdir -p {parent}\n\
         if {download}; then \
         bytes=$(wc -c < {out}); \
         printf '%s ' {ok}; printf '%s\\n' \"${{bytes}} bytes\"; \
         {cleanup}\
         else \
         code=$?; \
         printf '%s ' {failed}; printf '%s\\n' \"exit ${{code}}\"; \
         rm -f {out} {aria}; \
         exit ${{code}}; \
         fi\n",
        progress = crate::web::shell_quote(&progress),
        parent = crate::web::shell_quote(
            Path::new(&output)
                .parent()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|| ".".to_string())
                .as_str()
        ),
        download = download,
        out = crate::web::shell_quote(&output),
        aria = crate::web::shell_quote(&format!("{output}.aria2")),
        cleanup = cleanup,
        ok = crate::web::shell_quote(&ok),
        failed = crate::web::shell_quote(&failed),
    ));
    Ok(())
}

fn repo_id_for_source(source: &str) -> Result<String> {
    Ok(hi_ai::HfRepoRef::parse(source)?.repo_id)
}

impl HfCommandState {
    fn resolve_repo(&self, input: &str) -> Result<String> {
        let Some(index) = parse_positive_index(input) else {
            return Ok(input.to_string());
        };
        let Some(model) = self.menu.get(index - 1) else {
            let context = self
                .menu_author
                .as_deref()
                .map(|author| format!(" from /hf menu {author}"))
                .unwrap_or_default();
            bail!("menu number {index} is out of range{context}");
        };
        Ok(model.id.clone())
    }
}

fn parse_positive_index(input: &str) -> Option<usize> {
    if input.is_empty() || !input.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    input.parse::<usize>().ok().filter(|n| *n > 0)
}

fn is_positive_index(input: &str) -> bool {
    parse_positive_index(input).is_some()
}

fn whole_repo_mode(input: Option<&str>) -> Option<WholeRepoMode> {
    match input {
        Some("--" | "--all" | "all") => Some(WholeRepoMode::DeleteAfterEach),
        Some("--keep" | "keep") => Some(WholeRepoMode::Keep),
        _ => None,
    }
}

pub fn safe_path(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for c in input.chars() {
        if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    let out = out.trim_matches('_');
    if out.is_empty() {
        "download".to_string()
    } else {
        out.chars().take(160).collect()
    }
}

fn directory_file_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .filter_map(|entry| entry.metadata().ok())
                .filter(|metadata| metadata.is_file())
                .map(|metadata| metadata.len())
                .sum()
        })
        .unwrap_or(0)
}

#[cfg(unix)]
pub fn available_space_bytes(path: &Path) -> Option<u64> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let mut existing = path;
    while !existing.exists() {
        existing = existing.parent()?;
    }
    let c_path = CString::new(existing.as_os_str().as_bytes()).ok()?;
    let mut stats = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: `c_path` is NUL-free and `stats` points to writable storage of
    // the type required by statvfs. The return code is checked before reading.
    let result = unsafe { libc::statvfs(c_path.as_ptr(), stats.as_mut_ptr()) };
    if result != 0 {
        return None;
    }
    let stats = unsafe { stats.assume_init() };
    Some((stats.f_bavail as u64).saturating_mul(stats.f_frsize as u64))
}

#[cfg(not(unix))]
pub fn available_space_bytes(_path: &Path) -> Option<u64> {
    None
}

fn format_hf_search(models: &[hi_ai::ModelCandidate]) -> String {
    if models.is_empty() {
        return "No Hugging Face models found.\n".to_string();
    }
    let mut out = String::from("Hugging Face models:\n");
    for model in models {
        let tags = model
            .tags
            .iter()
            .take(4)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        let downloads = model
            .downloads
            .map(|n| n.to_string())
            .unwrap_or_else(|| "-".into());
        let likes = model
            .likes
            .map(|n| n.to_string())
            .unwrap_or_else(|| "-".into());
        let runnable = if model.runnable {
            "runnable"
        } else {
            "not runnable by provider"
        };
        out.push_str(&format!(
            "  {}  [downloads: {downloads}, likes: {likes}]  {runnable}\n",
            model.id
        ));
        if !tags.is_empty() {
            out.push_str(&format!("    tags: {tags}\n"));
        }
    }
    out
}

fn format_hf_author_menu(author: &str, models: &[hi_ai::ModelCandidate]) -> String {
    if models.is_empty() {
        return format!("No Hugging Face models found for author {author}.\n");
    }
    let mut out = format!("Hugging Face download menu for {author}:\n");
    for (idx, model) in models.iter().enumerate() {
        let downloads = model
            .downloads
            .map(|n| n.to_string())
            .unwrap_or_else(|| "-".into());
        let likes = model
            .likes
            .map(|n| n.to_string())
            .unwrap_or_else(|| "-".into());
        let updated = model.updated_at.as_deref().unwrap_or("-");
        let tags = model
            .tags
            .iter()
            .take(4)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        out.push_str(&format!(
            "  {}. {}  [files: {}, downloads: {downloads}, likes: {likes}, updated: {updated}]\n",
            idx + 1,
            model.id,
            model.files.len()
        ));
        if !tags.is_empty() {
            out.push_str(&format!("     tags: {tags}\n"));
        }
        out.push_str(&format!("     files: /hf files {}\n", idx + 1));
    }
    out.push_str(
        "\nUse /hf files <number> to inspect artifacts, /hf download <number> <file-number|filename> [output] for one file, /hf download <number> -- [dir] to validate and delete every file in one repo, /hf download -- [dir] to validate every listed repo for this author, /hf download <author> -- [dir] to fetch an author and validate it directly, or /hf download --keep [dir] to keep every listed repo.\n",
    );
    out
}

fn format_hf_files(
    repo: &hi_ai::HfRepoRef,
    files: &[hi_ai::HfFileInfo],
    repo_hint: Option<&str>,
) -> String {
    if files.is_empty() {
        return format!("No files found in {}@{}.\n", repo.repo_id, repo.revision);
    }
    let mut out = format!("Files in {}@{}:\n", repo.repo_id, repo.revision);
    for (idx, file) in files.iter().enumerate() {
        let size = file.size.map(human_size).unwrap_or_default();
        out.push_str(&format!("  {}. {}  {size}\n", idx + 1, file.path));
    }
    if let Some(repo_hint) = repo_hint {
        out.push_str(&format!(
            "\nUse /hf download {repo_hint} <file-number|filename> [output], /hf download {repo_hint} -- [dir] to validate and delete every file sequentially, or /hf download {repo_hint} --keep [dir] to keep the full model.\n"
        ));
    }
    out
}

fn human_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlx_model_present_rejects_huggingface_redirect_bodies() {
        let dir = std::env::temp_dir().join(format!("hi-mlx-cache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            "Temporary Redirect. Redirecting to /api/resolve-cache/…",
        )
        .unwrap();
        std::fs::write(dir.join("model.safetensors"), "Temporary Redirect").unwrap();
        assert!(!mlx_model_present(&dir));

        std::fs::write(dir.join("config.json"), r#"{"architectures":["Test"]}"#).unwrap();
        let header = b"{}";
        let mut shard = (header.len() as u64).to_le_bytes().to_vec();
        shard.extend_from_slice(header);
        std::fs::write(dir.join("model.safetensors"), shard).unwrap();
        assert!(mlx_model_present(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolves_menu_numbers_to_model_ids() {
        let state = HfCommandState {
            menu_author: Some("pipenetwork".to_string()),
            menu: vec![hi_ai::ModelCandidate {
                id: "pipenetwork/model".to_string(),
                source: hi_ai::ModelSource::HuggingFace,
                runnable: false,
                downloadable: true,
                tags: Vec::new(),
                files: Vec::new(),
                downloads: None,
                likes: None,
                updated_at: None,
                context_window: None,
                max_output_tokens: None,
                price: None,
                capabilities: Vec::new(),
            }],
            last_files_repo: None,
            last_files: Vec::new(),
        };

        assert_eq!(state.resolve_repo("1").unwrap(), "pipenetwork/model");
        assert!(state.resolve_repo("2").is_err());
        assert_eq!(state.resolve_repo("org/model").unwrap(), "org/model");
    }

    #[tokio::test]
    async fn author_download_requires_active_menu() {
        let mut state = HfCommandState::default();

        let err = handle_hf_command("download --", &mut state)
            .await
            .unwrap_err();

        assert!(err.to_string().contains("run /hf menu <author> first"));
    }

    #[test]
    fn all_download_command_downloads_and_deletes_each_file() {
        let client = hi_ai::HuggingFaceHubClient::new("https://huggingface.co", None);
        let repo = hi_ai::HfRepoRef::parse("org/model").unwrap();
        let files = vec![
            hi_ai::HfFileInfo {
                path: ".gitattributes".to_string(),
                size: Some(10),
            },
            hi_ai::HfFileInfo {
                path: "nested/model.gguf".to_string(),
                size: Some(20),
            },
        ];

        let command = all_download_command(
            &client,
            &repo,
            &files,
            Path::new("/tmp/hi-hf-test"),
            WholeRepoMode::DeleteAfterEach,
        )
        .unwrap();

        assert!(command.contains("starting 2 file(s) from org/model@main"));
        assert!(command.contains("https://huggingface.co/org/model/resolve/main/.gitattributes"));
        assert!(command.contains("nested/model.gguf"));
        assert!(command.contains("AI_AGENT: hi"));
        assert!(command.contains("hi/"));
        assert!(command.contains("rm -f"));
        assert!(command.contains("completed 2 file(s) from org/model@main"));
    }

    #[test]
    fn all_author_download_command_downloads_every_repo_and_deletes_each_file() {
        let client = hi_ai::HuggingFaceHubClient::new("https://huggingface.co", None);
        let repos = vec![
            RepoFiles {
                repo: hi_ai::HfRepoRef::parse("pipe/one").unwrap(),
                files: vec![hi_ai::HfFileInfo {
                    path: "model.gguf".to_string(),
                    size: Some(10),
                }],
            },
            RepoFiles {
                repo: hi_ai::HfRepoRef::parse("pipe/two").unwrap(),
                files: vec![hi_ai::HfFileInfo {
                    path: "nested/model.safetensors".to_string(),
                    size: Some(20),
                }],
            },
        ];

        let command = all_author_download_command(
            &client,
            "pipenetwork",
            &repos,
            Path::new("/tmp/hi-hf-pipenetwork-all"),
            WholeRepoMode::DeleteAfterEach,
        )
        .unwrap();

        assert!(command.contains("starting 2 file(s) from 2 repo(s) for pipenetwork"));
        assert!(command.contains("repo 1 / 2 pipe/one@main (1 file(s))"));
        assert!(command.contains("repo 2 / 2 pipe/two@main (1 file(s))"));
        assert!(command.contains("https://huggingface.co/pipe/one/resolve/main/model.gguf"));
        assert!(
            command
                .contains("https://huggingface.co/pipe/two/resolve/main/nested/model.safetensors")
        );
        assert!(command.contains("/tmp/hi-hf-pipenetwork-all/pipe_one/00001.model.gguf"));
        assert!(
            command.contains("/tmp/hi-hf-pipenetwork-all/pipe_two/00001.nested_model.safetensors")
        );
        assert!(command.contains("AI_AGENT: hi"));
        assert!(command.contains("rm -f"));
        assert!(command.contains("completed 2 file(s) from 2 repo(s) for pipenetwork"));
    }

    #[test]
    fn keep_download_command_preserves_nested_paths_without_success_cleanup() {
        let client = hi_ai::HuggingFaceHubClient::new("https://huggingface.co", None);
        let repo = hi_ai::HfRepoRef::parse("org/model").unwrap();
        let files = vec![hi_ai::HfFileInfo {
            path: "nested/model.gguf".to_string(),
            size: Some(20),
        }];

        let command = all_download_command_with_availability(
            &client,
            &repo,
            &files,
            Path::new("/tmp/hi-hf-keep"),
            WholeRepoMode::Keep,
            Some(true),
        )
        .unwrap();

        assert!(command.contains("mkdir -p /tmp/hi-hf-keep/nested"));
        assert!(command.contains("-d /tmp/hi-hf-keep/nested -o model.gguf"));
        assert_eq!(
            command
                .matches("rm -f /tmp/hi-hf-keep/nested/model.gguf")
                .count(),
            1
        );
    }

    #[test]
    fn health_requires_ready_true() {
        assert!(health_ready(&json!({"status": "ok", "ready": true})));
        assert!(!health_ready(&json!({"status": "ok", "ready": false})));
        assert!(!health_ready(&json!({"status": "ok"})));
    }
}
