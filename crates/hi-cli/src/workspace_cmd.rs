//! Preactivation workspace recovery commands.
//!
//! This dispatcher deliberately depends only on local PipeFS cache APIs and
//! authority configuration. It runs before session, provider, or remote
//! workspace startup so a broken activation path cannot hide recovery data.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail, ensure};
use clap::{Args, Parser, Subcommand, error::ErrorKind};
use serde::Serialize;

mod journal_recovery;
mod local_recovery;
mod migration;
mod recovery_service;

use recovery_service::{RecoveryInspection, RecoveryService, for_authority as recovery_service};

const AUTHORITY_HELP: &str = "\
Omit --session to inspect or disposition restart evidence for the current
local workspace without sync credentials. PipeFS recovery caches are scoped to
an authenticated sync authority. Set both
HI_SYNC_BASE_URL and HI_SYNC_API_KEY, or configure both values in a trusted
[sync] section. These commands inspect local recovery state and do not start a
provider unless the selected command must acquire a writer lease. Status,
recover list/inspect/export, and import preview stay local; retry, takeover,
detach, and confirmed import contact the configured authority.";
const OUTPUT_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Parser)]
#[command(
    name = "hi workspace",
    about = "Inspect and salvage local or PipeFS recovery state",
    after_help = AUTHORITY_HELP
)]
struct WorkspaceCli {
    /// Read sync authority configuration from this file.
    #[arg(long, global = true, value_name = "PATH")]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: WorkspaceCommand,
}

#[derive(Debug, Subcommand)]
enum WorkspaceCommand {
    /// Show local PipeFS recovery and remote-probe indicators.
    Status(SessionOutputArgs),
    /// Inspect, export, or permanently discard a recovery cache.
    Recover {
        #[command(subcommand)]
        command: RecoveryCommand,
    },
    /// Explicitly replace the current remote writer lease.
    Takeover(RequiredSessionOutputArgs),
    /// Disable PipeFS authority only when no local recovery evidence exists.
    Detach(migration::DetachArgs),
    /// Preview or confirm an explicit launch-directory import.
    Import(migration::ImportArgs),
    /// Materialize a verified remote revision into a fresh local directory.
    Export(RemoteExportArgs),
}

#[derive(Debug, Subcommand)]
enum RecoveryCommand {
    /// List recovery caches for one session and authenticated authority.
    List(SessionOutputArgs),
    /// Inspect one exact recovery cache.
    Inspect(InspectArgs),
    /// Reconcile and publish one retained recovery cache.
    Retry(InspectArgs),
    /// Export a cache as a fresh deterministic `.tar.zst` archive.
    Export(ExportArgs),
    /// Permanently discard one exact recovery cache.
    Discard(DiscardArgs),
}

#[derive(Debug, Args)]
struct SessionOutputArgs {
    /// Canonical remote session ID; omit for the current local workspace.
    #[arg(long, value_name = "ID")]
    session: Option<String>,
    /// Emit stable machine-readable JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct RequiredSessionOutputArgs {
    #[arg(long, value_name = "ID")]
    session: String,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct InspectArgs {
    /// Exact recovery ID from `recover list`.
    #[arg(value_name = "RECOVERY_ID")]
    recovery_id: String,
    /// Canonical remote session ID; omit for the current local workspace.
    #[arg(long, value_name = "ID")]
    session: Option<String>,
    /// Emit stable machine-readable JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct ExportArgs {
    /// Exact recovery ID from `recover list`.
    #[arg(value_name = "RECOVERY_ID")]
    recovery_id: String,
    /// Canonical remote session ID; omit to receive the local in-place guidance.
    #[arg(long, value_name = "ID")]
    session: Option<String>,
    /// New archive path. The destination must not already exist.
    #[arg(long, value_name = "PATH")]
    to: PathBuf,
}

#[derive(Debug, Args)]
struct RemoteExportArgs {
    /// Canonical remote session ID.
    #[arg(long, value_name = "ID")]
    session: String,
    /// Fresh destination directory; it must not already exist.
    #[arg(long, value_name = "NEW_PATH")]
    to: PathBuf,
    /// `HEAD` (default) or an exact revision UUID from the restore chain.
    #[arg(long, value_name = "HEAD|UUID")]
    revision: Option<String>,
}

#[derive(Debug, Args)]
struct DiscardArgs {
    /// Exact recovery ID from `recover list`.
    #[arg(value_name = "RECOVERY_ID")]
    recovery_id: String,
    /// Canonical remote session ID; omit for the current local workspace.
    #[arg(long, value_name = "ID")]
    session: Option<String>,
    /// Content digest shown by `recover inspect`.
    #[arg(long, value_name = "DIGEST")]
    confirm: String,
    /// Acknowledge external writers are stopped and accept all current bytes.
    #[arg(long)]
    accept_current_bytes: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum AuthoritySource {
    Environment,
    Config,
}

struct CacheAuthority {
    client: hi_pipefs::PipeFsClient,
    scope: hi_pipefs::PipeFsCacheScope,
    source: AuthoritySource,
    machine_id: Option<String>,
    sync_config: crate::sync::SyncConfig,
}

#[derive(Debug, Serialize)]
struct RecoveryCacheView {
    id: String,
    confirmation_digest: Option<String>,
    path: String,
    workspace_root: Option<String>,
    phase: Option<hi_pipefs::WorkspacePhase>,
    logical_size_bytes: u64,
    pending_archive_bytes: u64,
    last_error: Option<String>,
}

impl From<hi_pipefs::PipeFsRecoveryCache> for RecoveryCacheView {
    fn from(cache: hi_pipefs::PipeFsRecoveryCache) -> Self {
        Self {
            id: cache.id,
            confirmation_digest: cache.confirmation_digest,
            path: cache.path.display().to_string(),
            workspace_root: cache.workspace_root.map(|path| path.display().to_string()),
            phase: cache.phase,
            logical_size_bytes: cache.logical_size_bytes,
            pending_archive_bytes: cache.pending_archive_bytes,
            last_error: cache.last_error,
        }
    }
}

#[derive(Debug, Serialize)]
struct RecoveryListView {
    schema_version: u16,
    session_id: String,
    authority_source: AuthoritySource,
    recovery_caches: Vec<RecoveryCacheView>,
    journal_recoveries: Vec<journal_recovery::JournalRecoveryView>,
}

#[derive(Debug, Serialize)]
struct RecoveryInspectView {
    schema_version: u16,
    session_id: String,
    authority_source: AuthoritySource,
    recovery_cache: RecoveryCacheView,
}

#[derive(Debug, Serialize)]
struct LocalStatusView {
    schema_version: u16,
    session_id: String,
    authority_source: AuthoritySource,
    remote_probe_required: bool,
    recovery_required: bool,
    recovery_caches: Vec<RecoveryCacheView>,
    journal_recoveries: Vec<journal_recovery::JournalRecoveryView>,
}

/// Run `hi workspace ...` before ordinary CLI/session initialization.
pub(crate) async fn run_cli(args: &[String]) -> Result<()> {
    let cli = match WorkspaceCli::try_parse_from(
        std::iter::once("hi workspace").chain(args.iter().map(String::as_str)),
    ) {
        Ok(cli) => cli,
        Err(error)
            if matches!(
                error.kind(),
                ErrorKind::DisplayHelp | ErrorKind::DisplayVersion
            ) =>
        {
            print!("{error}");
            return Ok(());
        }
        Err(error) => return Err(anyhow!(error)),
    };

    let Some(session_id) = command_session_id(&cli.command) else {
        return local_recovery::run(cli.command).await;
    };
    validate_session(session_id)?;
    let authority = resolve_cache_authority(cli.config.as_deref())?;
    match cli.command {
        WorkspaceCommand::Status(args) => run_status(&authority, args),
        WorkspaceCommand::Recover { command } => match command {
            RecoveryCommand::List(args) => run_list(&authority, args),
            RecoveryCommand::Inspect(args) => run_inspect(&authority, args),
            RecoveryCommand::Retry(args) => run_retry(&authority, args).await,
            RecoveryCommand::Export(args) => run_export(&authority, args).await,
            RecoveryCommand::Discard(args) => run_discard(&authority, args).await,
        },
        WorkspaceCommand::Takeover(args) => migration::run_takeover(&authority, args).await,
        WorkspaceCommand::Detach(args) => migration::run_detach(&authority, args).await,
        WorkspaceCommand::Import(args) => migration::run_import(&authority, args).await,
        WorkspaceCommand::Export(args) => run_remote_export(&authority, args).await,
    }
}

fn run_status(authority: &CacheAuthority, args: SessionOutputArgs) -> Result<()> {
    let session = args.session.expect("PipeFS dispatch requires a session");
    let mut recovery_required = hi_pipefs::local_recovery_required(&authority.scope, &session);
    let remote_probe_required =
        hi_pipefs::local_state_requires_remote_probe(&authority.scope, &session);
    let inventory = recovery_service(authority, &session).inventory()?;
    recovery_required |= !inventory.journal_recoveries.is_empty();
    let caches = inventory
        .caches
        .into_iter()
        .map(RecoveryCacheView::from)
        .collect::<Vec<_>>();
    let view = LocalStatusView {
        schema_version: OUTPUT_SCHEMA_VERSION,
        session_id: session,
        authority_source: authority.source,
        remote_probe_required,
        recovery_required,
        recovery_caches: caches,
        journal_recoveries: inventory.journal_recoveries,
    };
    if args.json {
        print_json(&view)
    } else {
        println!("PipeFS local status for session {}", view.session_id);
        println!(
            "authority: {}",
            authority_source_label(view.authority_source)
        );
        println!(
            "remote probe required: {}",
            yes_no(view.remote_probe_required)
        );
        println!("recovery required: {}", yes_no(view.recovery_required));
        println!("recovery caches: {}", view.recovery_caches.len());
        for cache in &view.recovery_caches {
            println!("- {}", recovery_cache_summary(cache));
        }
        for recovery in &view.journal_recoveries {
            journal_recovery::print_summary(recovery);
        }
        Ok(())
    }
}

fn run_list(authority: &CacheAuthority, args: SessionOutputArgs) -> Result<()> {
    let session = args.session.expect("PipeFS dispatch requires a session");
    let inventory = recovery_service(authority, &session).inventory()?;
    let caches = inventory
        .caches
        .into_iter()
        .map(RecoveryCacheView::from)
        .collect::<Vec<_>>();
    let view = RecoveryListView {
        schema_version: OUTPUT_SCHEMA_VERSION,
        session_id: session,
        authority_source: authority.source,
        recovery_caches: caches,
        journal_recoveries: inventory.journal_recoveries,
    };
    if args.json {
        print_json(&view)
    } else if view.recovery_caches.is_empty() && view.journal_recoveries.is_empty() {
        println!(
            "PipeFS: no recovery caches for session {} ({})",
            view.session_id,
            authority_source_label(view.authority_source)
        );
        Ok(())
    } else {
        println!("PipeFS recovery caches for session {}:", view.session_id);
        for cache in &view.recovery_caches {
            println!("- {}", recovery_cache_summary(cache));
        }
        for recovery in &view.journal_recoveries {
            journal_recovery::print_summary(recovery);
        }
        Ok(())
    }
}

fn run_inspect(authority: &CacheAuthority, args: InspectArgs) -> Result<()> {
    let session = args.session.expect("PipeFS dispatch requires a session");
    match recovery_service(authority, &session).inspect(&args.recovery_id)? {
        RecoveryInspection::Journal(recovery) => {
            if args.json {
                return print_json(&recovery);
            }
            journal_recovery::print_detail(&recovery, &session);
            Ok(())
        }
        RecoveryInspection::Cache(cache) => {
            let view = RecoveryInspectView {
                schema_version: OUTPUT_SCHEMA_VERSION,
                session_id: session,
                authority_source: authority.source,
                recovery_cache: RecoveryCacheView::from(cache),
            };
            if args.json {
                print_json(&view)
            } else {
                println!(
                    "{}",
                    format_recovery_cache(
                        &view.recovery_cache,
                        Some((&view.session_id, view.authority_source))
                    )
                );
                Ok(())
            }
        }
    }
}

async fn run_retry(authority: &CacheAuthority, args: InspectArgs) -> Result<()> {
    let session = args.session.expect("PipeFS dispatch requires a session");
    let receipt = recovery_service(authority, &session)
        .retry(&args.recovery_id)
        .await?;
    if args.json {
        #[derive(Serialize)]
        struct RecoveryView<'a> {
            schema_version: u16,
            recovery_id: &'a str,
            revision_id: Option<String>,
        }
        return print_json(&RecoveryView {
            schema_version: OUTPUT_SCHEMA_VERSION,
            recovery_id: &receipt.requested_id,
            revision_id: receipt.revision_id.map(|id| id.to_string()),
        });
    }
    println!(
        "PipeFS recovery {} reconciled at revision {}",
        receipt.requested_id,
        receipt
            .revision_id
            .map_or_else(|| "empty".into(), |id| id.to_string())
    );
    Ok(())
}

async fn run_export(authority: &CacheAuthority, args: ExportArgs) -> Result<()> {
    let session = args.session.expect("PipeFS dispatch requires a session");
    let exported = recovery_service(authority, &session)
        .export(&args.recovery_id, &args.to)
        .await?;
    println!(
        "PipeFS recovery cache exported to {}; source cache retained",
        exported.display()
    );
    Ok(())
}

async fn run_remote_export(authority: &CacheAuthority, args: RemoteExportArgs) -> Result<()> {
    let revision = parse_remote_revision(args.revision.as_deref())?;
    let receipt =
        hi_pipefs::export_remote_workspace(&authority.client, &args.session, revision, &args.to)
            .await?;
    println!(
        "PipeFS revision {} exported to {} ({} entries, {} bytes)",
        receipt
            .revision_id
            .map_or_else(|| "empty".to_string(), |value| value.to_string()),
        receipt.destination.display(),
        receipt.entry_count,
        receipt.logical_size_bytes
    );
    Ok(())
}

fn parse_remote_revision(value: Option<&str>) -> Result<Option<uuid::Uuid>> {
    match value.map(str::trim) {
        None | Some("") | Some("HEAD") => Ok(None),
        Some(value) => uuid::Uuid::parse_str(value)
            .map(Some)
            .with_context(|| format!("invalid PipeFS revision {value:?}")),
    }
}

async fn run_discard(authority: &CacheAuthority, args: DiscardArgs) -> Result<()> {
    let session = args.session.expect("PipeFS dispatch requires a session");
    recovery_service(authority, &session)
        .discard(&args.recovery_id, &args.confirm)
        .await?;
    println!(
        "PipeFS recovery {} permanently discarded with its entire owning cache; it cannot be recovered by hi",
        args.recovery_id
    );
    Ok(())
}

const RECOVERY_USAGE: &str = "usage: /pipefs recover list | inspect <recovery-id> | retry <recovery-id> | export <recovery-id> <destination.tar.zst> | discard <recovery-id> --confirm <whole-cache-digest>";

enum InteractiveRecoveryCommand<'a> {
    List,
    Inspect(&'a str),
    Retry(&'a str),
    Export {
        recovery_id: &'a str,
        to: &'a str,
    },
    Discard {
        recovery_id: &'a str,
        confirm: &'a str,
    },
}

fn parse_interactive_recovery(argument: &str) -> Result<InteractiveRecoveryCommand<'_>> {
    let mut parts = argument.trim().splitn(2, char::is_whitespace);
    let operation = parts.next().unwrap_or("").to_ascii_lowercase();
    let remainder = parts.next().unwrap_or("").trim();
    match operation.as_str() {
        "list" if remainder.is_empty() => Ok(InteractiveRecoveryCommand::List),
        "inspect" | "retry"
            if !remainder.is_empty() && !remainder.contains(char::is_whitespace) =>
        {
            Ok(if operation == "inspect" {
                InteractiveRecoveryCommand::Inspect(remainder)
            } else {
                InteractiveRecoveryCommand::Retry(remainder)
            })
        }
        "export" => {
            let mut values = remainder.splitn(2, char::is_whitespace);
            let recovery_id = values.next().unwrap_or("");
            let to = values.next().unwrap_or("").trim();
            ensure!(!recovery_id.is_empty() && !to.is_empty(), RECOVERY_USAGE);
            Ok(InteractiveRecoveryCommand::Export { recovery_id, to })
        }
        "discard" => {
            let values = remainder.split_whitespace().collect::<Vec<_>>();
            ensure!(
                values.len() == 3 && values[1] == "--confirm",
                RECOVERY_USAGE
            );
            Ok(InteractiveRecoveryCommand::Discard {
                recovery_id: values[0],
                confirm: values[2],
            })
        }
        _ => bail!(RECOVERY_USAGE),
    }
}

/// Interactive compatibility adapter over the same preactivation recovery
/// service. Stable journal recovery IDs are preferred; generation cache IDs
/// remain accepted as legacy aliases and always resolve to the whole cache.
pub(crate) async fn run_pipefs_recovery_alias(
    client: &hi_pipefs::PipeFsClient,
    scope: &hi_pipefs::PipeFsCacheScope,
    session_id: &str,
    sync_config: &crate::sync::SyncConfig,
    active: bool,
    argument: &str,
) -> Result<String> {
    validate_session(session_id)?;
    let service = RecoveryService::new(
        client,
        scope,
        session_id,
        sync_config.machine_id.as_deref(),
        sync_config,
    );
    match parse_interactive_recovery(argument)? {
        InteractiveRecoveryCommand::List => {
            let inventory = service.inventory()?;
            if inventory.caches.is_empty() && inventory.journal_recoveries.is_empty() {
                return Ok("PipeFS: no recovery caches for this session".to_owned());
            }
            let mut output = format!("PipeFS recovery caches for session {session_id}:\n");
            for cache in inventory.caches {
                output.push_str("- ");
                output.push_str(&recovery_cache_summary(&RecoveryCacheView::from(cache)));
                output.push('\n');
            }
            for recovery in inventory.journal_recoveries {
                output.push_str(&journal_recovery::format_summary(&recovery));
                output.push('\n');
            }
            Ok(output)
        }
        InteractiveRecoveryCommand::Inspect(recovery_id) => match service.inspect(recovery_id)? {
            RecoveryInspection::Journal(recovery) => {
                Ok(journal_recovery::format_detail(&recovery, session_id))
            }
            RecoveryInspection::Cache(cache) => {
                Ok(format_recovery_cache(&RecoveryCacheView::from(cache), None))
            }
        },
        InteractiveRecoveryCommand::Retry(recovery_id) => {
            ensure!(
                !active,
                "turn PipeFS off before retrying a retained recovery cache; use /pipefs retry for the active workspace"
            );
            let receipt = service.retry(recovery_id).await?;
            Ok(format!(
                "PipeFS recovery {} reconciled at revision {}",
                receipt.requested_id,
                receipt
                    .revision_id
                    .map_or_else(|| "empty".into(), |id| id.to_string())
            ))
        }
        InteractiveRecoveryCommand::Export { recovery_id, to } => {
            ensure!(!active, "turn PipeFS off before exporting a recovery cache");
            let exported = service.export(recovery_id, Path::new(to)).await?;
            Ok(format!(
                "PipeFS recovery cache exported to {}; source cache retained",
                exported.display()
            ))
        }
        InteractiveRecoveryCommand::Discard {
            recovery_id,
            confirm,
        } => {
            ensure!(
                !active,
                "turn PipeFS off before discarding a recovery cache"
            );
            service.discard(recovery_id, confirm).await?;
            Ok(format!(
                "PipeFS recovery {recovery_id} discarded with its entire owning cache"
            ))
        }
    }
}

fn command_session_id(command: &WorkspaceCommand) -> Option<&str> {
    match command {
        WorkspaceCommand::Status(args) => args.session.as_deref(),
        WorkspaceCommand::Recover { command } => match command {
            RecoveryCommand::List(args) => args.session.as_deref(),
            RecoveryCommand::Inspect(args) => args.session.as_deref(),
            RecoveryCommand::Retry(args) => args.session.as_deref(),
            RecoveryCommand::Export(args) => args.session.as_deref(),
            RecoveryCommand::Discard(args) => args.session.as_deref(),
        },
        WorkspaceCommand::Takeover(args) => Some(&args.session),
        WorkspaceCommand::Detach(args) => Some(&args.session),
        WorkspaceCommand::Import(args) => Some(&args.session),
        WorkspaceCommand::Export(args) => Some(&args.session),
    }
}

fn validate_session(session_id: &str) -> Result<()> {
    crate::sync::validate_session_id(session_id).context("validating the PipeFS recovery session")
}

fn resolve_cache_authority(config_path: Option<&Path>) -> Result<CacheAuthority> {
    let env_base = std::env::var("HI_SYNC_BASE_URL").ok();
    let env_key = std::env::var("HI_SYNC_API_KEY").ok();
    if nonempty(env_base.as_deref()).is_some() || nonempty(env_key.as_deref()).is_some() {
        let (base_url, api_key) = require_pair(
            env_base.as_deref(),
            env_key.as_deref(),
            "HI_SYNC_BASE_URL and HI_SYNC_API_KEY",
        )?;
        return build_cache_authority(base_url, api_key, AuthoritySource::Environment);
    }

    let config = crate::config::load_config(config_path).with_context(|| {
        config_path.map_or_else(
            || "loading sync authority configuration".to_string(),
            |path| {
                format!(
                    "loading sync authority configuration from {}",
                    path.display()
                )
            },
        )
    })?;
    let section = config.sync.as_ref().ok_or_else(|| {
        anyhow!(
            "PipeFS cache authority is not configured; set both HI_SYNC_BASE_URL and HI_SYNC_API_KEY, or add a trusted [sync] base_url and credential"
        )
    })?;
    let (base_url, api_key) =
        authority_from_sync_section(section, |name| std::env::var(name).ok())?;
    build_cache_authority(&base_url, &api_key, AuthoritySource::Config)
}

fn authority_from_sync_section(
    section: &crate::config::SyncSection,
    environment: impl Fn(&str) -> Option<String>,
) -> Result<(String, String)> {
    let base_url = nonempty(section.base_url.as_deref()).ok_or_else(|| {
        anyhow!(
            "[sync] has no base_url; set both HI_SYNC_BASE_URL and HI_SYNC_API_KEY for preactivation recovery"
        )
    })?.to_string();
    let referenced_key = section
        .api_key_ref
        .as_deref()
        .map(|reference| {
            crate::config::resolve_credential_reference(
                reference,
                section.project_local,
                section.project_local,
            )
        })
        .transpose()?;
    let literal_key = nonempty(section.api_key.as_deref()).map(str::to_string);
    let api_key = if let Some(key) = referenced_key.or(literal_key) {
        key
    } else if let Some(name) = nonempty(section.api_key_env.as_deref()) {
        ensure!(
            !section.project_local,
            "project-local [sync] api_key_env is not accepted for recovery; set HI_SYNC_BASE_URL and HI_SYNC_API_KEY explicitly"
        );
        environment(name)
            .and_then(|value| nonempty(Some(&value)).map(str::to_string))
            .ok_or_else(|| anyhow!("[sync] credential environment variable {name} is not set"))?
    } else {
        bail!(
            "[sync] has no credential; set api_key_ref (legacy api_key/api_key_env are still readable), or set HI_SYNC_BASE_URL and HI_SYNC_API_KEY"
        );
    };
    Ok((base_url, api_key))
}

fn build_cache_authority(
    base_url: &str,
    api_key: &str,
    source: AuthoritySource,
) -> Result<CacheAuthority> {
    let client =
        hi_pipefs::PipeFsClient::new(hi_pipefs::PipeFsClientConfig::new(base_url, api_key))
            .map_err(anyhow::Error::from)
            .context("validating the PipeFS cache authority")?;
    let machine_id = crate::session::machine_id();
    Ok(CacheAuthority {
        scope: client.cache_scope(),
        client,
        source,
        machine_id: machine_id.clone(),
        sync_config: crate::sync::SyncConfig {
            base_url: base_url.to_owned(),
            api_key: api_key.to_owned(),
            machine_id,
            cwd_digest: None,
        },
    })
}

fn require_pair<'a>(
    base_url: Option<&'a str>,
    api_key: Option<&'a str>,
    label: &str,
) -> Result<(&'a str, &'a str)> {
    match (nonempty(base_url), nonempty(api_key)) {
        (Some(base_url), Some(api_key)) => Ok((base_url, api_key)),
        _ => bail!("{label} must both be set and non-empty for preactivation recovery"),
    }
}

fn nonempty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

fn print_json(value: &impl Serialize) -> Result<()> {
    println!(
        "{}",
        serde_json::to_string_pretty(value).context("encoding workspace recovery JSON")?
    );
    Ok(())
}

fn recovery_cache_summary(cache: &RecoveryCacheView) -> String {
    let mut summary = format!(
        "{}: phase {}, {} logical bytes, {} pending archive bytes; whole-cache confirmation {}",
        cache.id,
        phase_label(cache.phase),
        cache.logical_size_bytes,
        cache.pending_archive_bytes,
        cache
            .confirmation_digest
            .as_deref()
            .unwrap_or("unavailable")
    );
    if let Some(error) = &cache.last_error {
        summary.push_str("; warning: ");
        summary.push_str(error);
    }
    summary
}

fn format_recovery_cache(
    cache: &RecoveryCacheView,
    preactivation: Option<(&str, AuthoritySource)>,
) -> String {
    let mut output = format!("PipeFS recovery cache {}\n", cache.id);
    if let Some((session_id, source)) = preactivation {
        output.push_str(&format!(
            "session: {session_id}\nauthority: {}\n",
            authority_source_label(source)
        ));
    }
    output.push_str(&format!(
        "path: {}\nwhole-cache discard confirmation: {}\nphase: {}\nworkspace: {}\nlogical bytes: {}\npending archive bytes: {}\nlast error: {}",
        cache.path,
        cache.confirmation_digest.as_deref().unwrap_or("unavailable"),
        phase_label(cache.phase),
        cache.workspace_root.as_deref().unwrap_or("unavailable"),
        cache.logical_size_bytes,
        cache.pending_archive_bytes,
        cache.last_error.as_deref().unwrap_or("none")
    ));
    output
}

fn phase_label(phase: Option<hi_pipefs::WorkspacePhase>) -> String {
    phase.map_or_else(
        || "unknown".to_string(),
        |phase| {
            serde_json::to_value(phase)
                .ok()
                .and_then(|value| value.as_str().map(str::to_string))
                .unwrap_or_else(|| "unknown".to_string())
        },
    )
}

fn authority_source_label(source: AuthoritySource) -> &'static str {
    match source {
        AuthoritySource::Environment => "HI_SYNC_* environment",
        AuthoritySource::Config => "trusted [sync] configuration",
    }
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

#[cfg(test)]
#[path = "workspace_cmd_tests.rs"]
mod tests;
