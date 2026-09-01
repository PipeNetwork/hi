//! CLI orchestration helpers extracted from `main` so the binary stays a thin
//! dispatcher: best-of-N, sync config, MCP/HF side commands.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use hi_agent::VerifyStage;
use hi_ai::PipeMcpClient;
use hi_tools::McpBackend;

use crate::bestof;
use crate::config::{Cli, Config, Settings};
use crate::provider::provider_label;
use crate::report::pipeline_command;
use crate::session;
use crate::sync;

/// Run `--best-of N` in isolated worktrees; returns whether a candidate completed.
#[allow(
    clippy::too_many_arguments,
    reason = "orchestration receives the resolved CLI, workspace, quality, and reporting inputs"
)]
pub(crate) fn run_best_of(
    cli: &Cli,
    settings: &Settings,
    workspace_root: &Path,
    state_root: &Path,
    verify_stages: &[VerifyStage],
    quality_max_verify_repairs: u32,
    prompt: &str,
    report_path: Option<&Path>,
) -> Result<bool> {
    let judge = bestof::effective_judge(cli.judge.as_deref(), !verify_stages.is_empty());
    let verify = pipeline_command(verify_stages).unwrap_or_default();
    if judge != hi_research::JudgeChoice::Model && verify.trim().is_empty() {
        anyhow::bail!("--best-of requires a resolved verification pipeline");
    }
    if !hi_tools::worktree::in_git_repo(workspace_root) {
        anyhow::bail!("--best-of requires a git repository");
    }
    let exe = std::env::current_exe().context("locating the hi executable")?;
    bestof::run(&bestof::BestOf {
        exe: &exe,
        provider: provider_label(settings.provider),
        model: &settings.model,
        base_url: &settings.base_url,
        api_key: &settings.api_key,
        verify: &verify,
        prompt,
        candidates: cli.best_of,
        max_steps: cli.max_steps,
        max_verify: quality_max_verify_repairs,
        workspace_root,
        state_root,
        report: report_path,
        targets: None,
        max_concurrency: cli.best_of as usize,
        apply: true,
        fuzz: None,
        expected_workspace_digest: None,
        judge,
        research_id: None,
        snippet_block: String::new(),
    })
}

/// Build remote-session sync credentials.
///
/// Precedence (first non-empty wins):
/// 1. `HI_SYNC_BASE_URL` / `HI_SYNC_API_KEY` env
/// 2. config file `[sync]` section
/// 3. CLI `--base-url` / `--api-key` when present
/// 4. resolved provider `settings` (profile defaults)
pub(crate) fn build_sync_config(settings: &Settings, cli: &Cli, file: &Config) -> sync::SyncConfig {
    let sync_section = file.sync.as_ref();
    let endpoint_candidates = [
        (
            std::env::var("HI_SYNC_BASE_URL").ok(),
            SyncEndpointSource::Environment,
        ),
        (
            sync_section.and_then(|section| section.base_url.clone()),
            SyncEndpointSource::Config,
        ),
        (cli.base_url.clone(), SyncEndpointSource::Cli),
        (
            Some(settings.base_url.clone()),
            SyncEndpointSource::Provider,
        ),
    ];
    let (base_url, endpoint_source) = endpoint_candidates
        .into_iter()
        .find_map(|(candidate, source)| {
            candidate
                .filter(|value| !value.trim().is_empty())
                .map(|value| (value.trim_end_matches('/').to_string(), source))
        })
        .unwrap_or_default();
    // The API key is attached to every sync request, so the base URL must not
    // be able to redirect it onto a plaintext or non-HTTP endpoint. Only
    // https (or loopback http for local dev/tests) is allowed; anything else
    // disables sync rather than leaking the credential.
    let base_url = if base_url.is_empty() || sync_base_url_is_safe(&base_url) {
        base_url
    } else {
        eprintln!(
            "warning: sync base_url '{base_url}' is not https (or loopback http); \
             disabling remote sync to avoid exposing the API key"
        );
        String::new()
    };
    let file_api_key = sync_section.and_then(sync_section_api_key);
    let api_key = if base_url.is_empty() {
        String::new()
    } else {
        resolve_sync_api_key(
            endpoint_source,
            &base_url,
            &settings.base_url,
            &settings.api_key,
            std::env::var("HI_SYNC_API_KEY").ok(),
            file_api_key,
            cli.api_key.clone(),
        )
    };
    let machine_id = session::machine_id();
    let cwd_digest = Some(session::cwd_digest());
    sync::SyncConfig {
        base_url,
        api_key,
        machine_id,
        cwd_digest,
    }
}

fn sync_section_api_key(section: &crate::config::SyncSection) -> Option<String> {
    let literal = section.api_key.clone().filter(|key| !key.is_empty());
    if literal.is_some() {
        return literal;
    }
    if section.project_local && section.api_key_env.is_some() {
        eprintln!(
            "warning: ignoring project-local [sync] api_key_env; repository config cannot \
             read credentials from your environment"
        );
        return None;
    }
    section
        .api_key_env
        .as_deref()
        .and_then(|env_var| std::env::var(env_var).ok())
        .filter(|key| !key.is_empty())
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum SyncEndpointSource {
    Environment,
    Config,
    Cli,
    #[default]
    Provider,
}

fn resolve_sync_api_key(
    source: SyncEndpointSource,
    base_url: &str,
    provider_base_url: &str,
    provider_api_key: &str,
    sync_env_key: Option<String>,
    config_key: Option<String>,
    cli_key: Option<String>,
) -> String {
    let source_paired = match source {
        SyncEndpointSource::Environment => sync_env_key,
        SyncEndpointSource::Config => config_key,
        SyncEndpointSource::Cli => cli_key,
        SyncEndpointSource::Provider => Some(provider_api_key.to_string()),
    };
    first_nonempty(&[source_paired])
        .or_else(|| {
            // Reusing the resolved provider credential is safe only when sync
            // is staying on the same authenticated origin.
            hi_provider_config::same_endpoint_origin(base_url, provider_base_url)
                .then(|| provider_api_key.trim().to_string())
                .filter(|key| !key.is_empty())
        })
        .unwrap_or_default()
}

/// First non-empty string in precedence order (env → file → cli → settings).
fn first_nonempty(candidates: &[Option<String>]) -> Option<String> {
    candidates
        .iter()
        .filter_map(|c| c.as_ref())
        .map(|s| s.trim())
        .find(|s| !s.is_empty())
        .map(|s| s.to_string())
}

/// True when a base URL is safe to receive a credential: `https`, or
/// loopback `http` (127.0.0.1 / localhost / [::1]) for local dev and tests.
/// Anything else (plaintext remote http, non-HTTP schemes, malformed URLs)
/// is rejected so a misconfigured or attacker-controlled endpoint cannot
/// harvest the credential.
pub(crate) fn sync_base_url_is_safe(url: &str) -> bool {
    let url = url.trim();
    if let Some(rest) = url.strip_prefix("https://") {
        return !rest.is_empty();
    }
    let Some(rest) = url.strip_prefix("http://") else {
        return false;
    };
    sync_host_is_loopback(rest)
}

/// Extract the host from a URL authority/path tail and report loopback.
fn sync_host_is_loopback(rest: &str) -> bool {
    // Authority ends at the first path/query/fragment delimiter.
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    // Drop any userinfo.
    let authority = authority.rsplit('@').next().unwrap_or_default();
    // Bracketed IPv6 literal: host is inside the brackets; anything after the
    // closing bracket (e.g. `:port`) is not part of the host.
    let host = if let Some(after_open) = authority.strip_prefix('[') {
        after_open.split(']').next().unwrap_or_default()
    } else {
        // Bare host or host:port.
        authority.split(':').next().unwrap_or_default()
    };
    let host = host.trim();
    host.eq_ignore_ascii_case("localhost") || host == "127.0.0.1" || host == "::1"
}

pub(crate) async fn run_mcp_cli(args: &[String]) -> Result<()> {
    let workspace = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let file = crate::config::load_config(None).unwrap_or_default();
    let policy = file.mcp_import.to_policy();
    let cli = crate::config::Cli::parse_from(["hi"]);
    let settings = crate::config::resolve(&cli, &file).ok();
    let pipe_attach = settings.as_ref().and_then(|settings| {
        crate::mcp_host::decide_pipe_attach(
            settings.mcp_pipe_enabled,
            settings.mcp_url.as_deref(),
            &settings.api_key,
            settings.mcp_pipe_allow.clone(),
        )
        .ok()
    });
    let policies = file.mcp.server_allowlists();
    match args.first().map(String::as_str) {
        None | Some("status") | Some("list") => {
            match crate::mcp_host::connect_workspace_mcp_with_policies(
                &workspace,
                &policy,
                pipe_attach.as_ref(),
                &policies,
            )
            .await
            {
                (Some(host), _) => print!("{}", host.workspace_status().await),
                (None, _) => print!(
                    "workspace MCP: (none)\n  add `.hi/mcp/*.json` or `.mcp.json`\n  `hi mcp pipe` inspects the provider mcp_url\n  first-party Pipe attaches when mcp_url + API key are set\n  `/mcp add <name> --stdio … | --http <url>` writes `.hi/mcp/<name>.json`\n"
                ),
            }
            Ok(())
        }
        Some("pipe") => {
            let settings = settings.ok_or_else(|| anyhow!("could not resolve settings"))?;
            let Some(url) = settings.mcp_url.as_deref() else {
                return Err(anyhow!("no MCP URL configured for this provider"));
            };
            let report = mcp_inspect(url, &settings.api_key, &settings.model).await?;
            print!("{report}");
            Ok(())
        }
        Some("test") => {
            let name = args
                .get(1)
                .map(String::as_str)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| anyhow!("usage: hi mcp test <name>"))?;
            let report = crate::mcp_host::test_workspace_mcp(
                &workspace,
                &policy,
                pipe_attach.as_ref(),
                name,
            )
            .await?;
            print!("{report}");
            Ok(())
        }
        Some("add") => {
            let rest = args[1..].join(" ");
            let (Some(host), _) = crate::mcp_host::connect_workspace_mcp_with_policies(
                &workspace,
                &policy,
                pipe_attach.as_ref(),
                &policies,
            )
            .await
            else {
                return Err(anyhow!("could not open an MCP host for this workspace"));
            };
            print!("{}", host.workspace_admin(&format!("add {rest}")).await?);
            Ok(())
        }
        Some("serve") => crate::mcp_serve::run().await,
        Some(other) => Err(anyhow!(
            "unknown mcp command '{other}'\nusage: hi mcp [status|pipe|test <name>|add <name> --stdio|--http|serve]"
        )),
    }
}

pub(crate) async fn run_hf_cli(args: &[String]) -> Result<()> {
    if args.is_empty() {
        print!(
            "{}",
            hi_tools::handle_hf_command("help", &mut hi_tools::HfCommandState::default()).await?
        );
        return Ok(());
    }
    if args.first().map(String::as_str) == Some("download")
        && args
            .get(2)
            .map(String::as_str)
            .is_some_and(|arg| matches!(arg, "--keep" | "keep"))
    {
        let repo = args
            .get(1)
            .ok_or_else(|| anyhow!("usage: hi hf download <repo[@revision]> --keep <dir>"))?;
        let dir = args
            .get(3)
            .ok_or_else(|| anyhow!("usage: hi hf download <repo[@revision]> --keep <dir>"))?;
        print!(
            "{}",
            hi_tools::download_repo_keep_foreground(repo, dir).await?
        );
        return Ok(());
    }

    let mut state = hi_tools::HfCommandState::default();
    match hi_tools::handle_hf_command_result(&args.join(" "), &mut state).await? {
        hi_tools::HfCommandResult::Text(text) => print!("{text}"),
        hi_tools::HfCommandResult::MlxReady(run) => print!("{}", run.message),
    }
    Ok(())
}

/// Build the MCP inspection report (server, tools, model count, current model)
/// as a plain-text block. Shared by the `hi mcp` one-shot and the REPL `/mcp`
/// command so their output can't drift.
pub(crate) async fn mcp_inspect(url: &str, api_key: &str, current_model: &str) -> Result<String> {
    let client = PipeMcpClient::new(url, api_key);
    let (server, protocol) = client.server_info().await?;
    let tools = client.tools_list().await?;
    let models = client.list_models().await?;
    let mut out = String::new();
    out.push_str(&format!("mcp_url:  {url}\n"));
    out.push_str(&format!("server:   {server}\n"));
    out.push_str(&format!("protocol: {protocol}\n"));
    out.push_str("tools:\n");
    for tool in tools {
        let title = tool.title.as_deref().unwrap_or("");
        if title.is_empty() {
            out.push_str(&format!("  {}\n", tool.name));
        } else {
            out.push_str(&format!("  {}  - {}\n", tool.name, title));
        }
    }
    out.push_str(&format!("models:   {}\n", models.len()));
    if let Some(model) = models.iter().find(|m| m.id == current_model) {
        let provider = model.provider_label.as_deref().unwrap_or("Pipe");
        out.push_str(&format!("current:  {} · {}\n", model.id, provider));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_nonempty_prefers_earlier_rungs() {
        assert_eq!(
            first_nonempty(&[
                None,
                Some(String::new()),
                Some("cli".into()),
                Some("settings".into()),
            ])
            .as_deref(),
            Some("cli")
        );
    }

    #[test]
    fn sync_base_url_accepts_https_and_loopback() {
        assert!(sync_base_url_is_safe("https://api.pipenetwork.ai/v1"));
        assert!(sync_base_url_is_safe("https://example.com"));
        assert!(sync_base_url_is_safe("http://127.0.0.1:8080/v1"));
        assert!(sync_base_url_is_safe("http://localhost:3000"));
        assert!(sync_base_url_is_safe("http://LOCALHOST:3000"));
        assert!(sync_base_url_is_safe("http://[::1]:3000"));
    }

    #[test]
    fn sync_base_url_rejects_plaintext_remote_and_non_http() {
        assert!(!sync_base_url_is_safe("http://evil.example"));
        assert!(!sync_base_url_is_safe("http://169.254.169.254/latest"));
        assert!(!sync_base_url_is_safe("ftp://example.com"));
        assert!(!sync_base_url_is_safe("file:///etc/passwd"));
        assert!(!sync_base_url_is_safe("https://"));
        assert!(!sync_base_url_is_safe("example.com"));
        assert!(!sync_base_url_is_safe(""));
    }

    #[test]
    fn config_sync_endpoint_cannot_consume_ambient_keys_from_other_sources() {
        let key = resolve_sync_api_key(
            SyncEndpointSource::Config,
            "https://attacker.example/v1",
            "https://openrouter.ai/api/v1",
            "resolved-provider-key",
            Some("ambient-sync-key".into()),
            None,
            Some("cli-provider-key".into()),
        );
        assert!(key.is_empty());

        let paired = resolve_sync_api_key(
            SyncEndpointSource::Config,
            "https://gateway.example/v1",
            "https://openrouter.ai/api/v1",
            "resolved-provider-key",
            Some("ambient-sync-key".into()),
            Some("profile-paired-sync-key".into()),
            Some("cli-provider-key".into()),
        );
        assert_eq!(paired, "profile-paired-sync-key");
    }

    #[test]
    fn sync_can_reuse_provider_key_only_on_the_same_origin() {
        let key = resolve_sync_api_key(
            SyncEndpointSource::Config,
            "https://openrouter.ai/sync/v1",
            "https://openrouter.ai/api/v1",
            "resolved-provider-key",
            None,
            None,
            None,
        );
        assert_eq!(key, "resolved-provider-key");
    }

    #[test]
    fn project_sync_section_cannot_read_environment_but_can_use_literal_key() {
        let blocked = crate::config::SyncSection {
            project_local: true,
            api_key_env: Some("PATH".into()),
            ..Default::default()
        };
        assert_eq!(sync_section_api_key(&blocked), None);

        let literal = crate::config::SyncSection {
            project_local: true,
            api_key: Some("repository-test-key".into()),
            api_key_env: Some("PATH".into()),
            ..Default::default()
        };
        assert_eq!(
            sync_section_api_key(&literal).as_deref(),
            Some("repository-test-key")
        );
    }
}
