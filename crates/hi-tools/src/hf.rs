use anyhow::{Result, anyhow, bail};
use serde_json::json;
use std::path::{Path, PathBuf};

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

pub async fn handle_hf_command(arg: &str, state: &mut HfCommandState) -> Result<String> {
    let arg = arg.trim();
    let mut parts = arg.split_whitespace();
    let Some(subcommand) = parts.next() else {
        return Ok(hf_usage());
    };
    match subcommand {
        "search" => {
            let query = arg.strip_prefix("search").unwrap_or("").trim();
            if query.is_empty() {
                bail!("usage: /hf search <query>");
            }
            let client = hi_ai::HuggingFaceHubClient::from_env();
            let models = client.search_models(query, 10).await?;
            Ok(format_hf_search(&models))
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
            Ok(format_hf_author_menu(author, &state.menu))
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
            Ok(format_hf_files(&repo, &state.last_files, Some(repo_arg)))
        }
        "download" => {
            let repo_arg = parts.next().ok_or_else(|| anyhow!(download_usage()))?;
            let file_arg = parts.next();
            let output = parts.next();

            if repo_arg.contains(':')
                && !is_positive_index(repo_arg)
                && whole_repo_mode(file_arg).is_none()
            {
                let out = run_download(repo_arg.to_string(), file_arg).await?;
                return Ok(format!("{}\n", out.content));
            }

            let repo_source = state.resolve_repo(repo_arg)?;
            let Some(file_arg) = file_arg else {
                if is_positive_index(repo_arg) {
                    let client = hi_ai::HuggingFaceHubClient::from_env();
                    let (repo, files) = fetch_files(&client, &repo_source).await?;
                    state.last_files_repo = Some(repo.repo_id.clone());
                    state.last_files = files;
                    return Ok(format_hf_files(&repo, &state.last_files, Some(repo_arg)));
                }
                bail!(download_usage());
            };

            if let Some(mode) = whole_repo_mode(Some(file_arg)) {
                let client = hi_ai::HuggingFaceHubClient::from_env();
                let (repo, files) = fetch_files(&client, &repo_source).await?;
                state.last_files_repo = Some(repo.repo_id.clone());
                state.last_files = files;
                return start_all_download(&client, &repo, &state.last_files, output, mode);
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
            Ok(format!("{}\n", out.content))
        }
        _ => Ok(hf_usage()),
    }
}

fn hf_usage() -> String {
    "usage: /hf search <query> | /hf menu <author> [limit] | /hf files <repo|number> | /hf download <repo|number> <filename|file-number> [output]\n".to_string()
}

fn download_usage() -> String {
    "usage: /hf download <repo[@revision]|menu-number> <filename|file-number|--|--keep> [output]"
        .to_string()
}

async fn run_download(source: String, output: Option<&str>) -> Result<crate::ToolOutput> {
    let mut args = json!({ "source": source });
    if let Some(output) = output {
        args["output"] = serde_json::Value::String(output.to_string());
    }
    crate::web::run_web_download(&args.to_string()).await
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
    let id = crate::background::spawn(&command)?;
    let mode_text = match mode {
        WholeRepoMode::DeleteAfterEach => {
            "Each file is deleted after it finishes, so disk use stays bounded to one artifact plus aria2c metadata."
        }
        WholeRepoMode::Keep => {
            "Files are kept under the destination directory, preserving nested Hugging Face paths."
        }
    };
    Ok(format!(
        "Downloading all {} file(s) from {}@{} sequentially.\n→ {}\n{mode_text}\nStarted background process `{id}`. Poll progress with `bash_output`, stop with `bash_kill`.\n",
        files.len(),
        repo.repo_id,
        repo.revision,
        dir.display()
    ))
}

fn all_download_command(
    client: &hi_ai::HuggingFaceHubClient,
    repo: &hi_ai::HfRepoRef,
    files: &[hi_ai::HfFileInfo],
    output_dir: &Path,
    mode: WholeRepoMode,
) -> Result<String> {
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
        let output = output.to_string_lossy().to_string();
        let url = client.resolve_file_url(&repo.clone().with_filename(file.path.clone()))?;
        let download = crate::web::download_command(&url, &output);
        let progress = format!("{} / {} {}", idx + 1, files.len(), file.path);
        let ok = format!("ok {} {}", idx + 1, file.path);
        let failed = format!("failed {} {}", idx + 1, file.path);
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

fn safe_path(input: &str) -> String {
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
        "\nUse /hf files <number> to inspect artifacts, /hf download <number> <file-number|filename> [output] for one file, /hf download <number> -- [dir] to validate and delete every file sequentially, or /hf download <number> --keep [dir] to keep the full model.\n",
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
    fn keep_download_command_preserves_nested_paths_without_success_cleanup() {
        let client = hi_ai::HuggingFaceHubClient::new("https://huggingface.co", None);
        let repo = hi_ai::HfRepoRef::parse("org/model").unwrap();
        let files = vec![hi_ai::HfFileInfo {
            path: "nested/model.gguf".to_string(),
            size: Some(20),
        }];

        let command = all_download_command(
            &client,
            &repo,
            &files,
            Path::new("/tmp/hi-hf-keep"),
            WholeRepoMode::Keep,
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
}
