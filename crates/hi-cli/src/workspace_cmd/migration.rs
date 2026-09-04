use std::path::PathBuf;

use anyhow::{Result, ensure};
use clap::Args;
use serde::Serialize;

use super::{CacheAuthority, RequiredSessionOutputArgs, print_json};

#[derive(Debug, Args)]
pub(super) struct DetachArgs {
    /// Canonical remote session ID.
    #[arg(long, value_name = "ID")]
    pub(super) session: String,
    /// Required acknowledgement that detach is allowed only from clean state.
    #[arg(long)]
    if_clean: bool,
}

#[derive(Debug, Args)]
pub(super) struct ImportArgs {
    /// Canonical remote session ID.
    #[arg(long, value_name = "ID")]
    pub(super) session: String,
    /// Local directory to scan; it is never imported without confirmation.
    #[arg(long = "from", value_name = "PATH")]
    source: PathBuf,
    /// Print the content-bound preview without modifying remote state.
    #[arg(long, conflicts_with = "confirm")]
    preview: bool,
    /// Preview digest authorizing import of exactly the rescanned bytes.
    #[arg(long, value_name = "DIGEST", conflicts_with = "preview")]
    confirm: Option<String>,
    /// Emit stable machine-readable JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Serialize)]
struct LeaseView<'a> {
    schema_version: u16,
    session_id: &'a str,
    generation: u64,
    expires_at_unix: u64,
}

pub(super) async fn run_takeover(
    authority: &CacheAuthority,
    args: RequiredSessionOutputArgs,
) -> Result<()> {
    let receipt = authority
        .client
        .acquire_writer_lease(&args.session, machine_id(authority)?, true)
        .await?;
    let view = LeaseView {
        schema_version: 1,
        session_id: &args.session,
        generation: receipt.lease.generation,
        expires_at_unix: receipt.expires_at_unix,
    };
    if args.json {
        print_json(&view)
    } else {
        println!(
            "PipeFS writer lease taken over for session {} at generation {}",
            args.session, receipt.lease.generation
        );
        Ok(())
    }
}

pub(super) async fn run_detach(authority: &CacheAuthority, args: DetachArgs) -> Result<()> {
    ensure!(args.if_clean, "PipeFS detach requires --if-clean");
    let receipt = authority
        .client
        .acquire_writer_lease(&args.session, machine_id(authority)?, false)
        .await?;
    hi_pipefs::detach_if_clean(
        &authority.client,
        &authority.scope,
        &args.session,
        &receipt.lease,
    )
    .await?;
    println!("PipeFS detached cleanly for session {}", args.session);
    Ok(())
}

pub(super) async fn run_import(authority: &CacheAuthority, args: ImportArgs) -> Result<()> {
    ensure!(
        args.preview ^ args.confirm.is_some(),
        "choose exactly one of --preview or --confirm DIGEST"
    );
    let preview = hi_pipefs::preview_import(&args.source)?;
    if args.preview {
        if args.json {
            return print_json(&preview);
        }
        println!("PipeFS import preview for {}", preview.source.display());
        println!("scanner: {}", preview.scanner_version);
        println!("digest: {}", preview.confirmation_digest);
        println!(
            "entries: {}, bytes: {}",
            preview.entry_count, preview.byte_count
        );
        println!("exclusions: {}", preview.exclusions.len());
        println!("unsupported entries: {}", preview.unsupported_entries.len());
        return Ok(());
    }

    let confirmation = args.confirm.expect("validated above");
    let remote = authority.client.state(&args.session).await?;
    ensure!(
        remote.current_head.is_none(),
        "PipeFS import is allowed only when the remote workspace has no head"
    );
    let lease = authority
        .client
        .acquire_writer_lease(&args.session, machine_id(authority)?, false)
        .await?;
    let receipt = hi_pipefs::import_workspace(
        &authority.client,
        lease.lease,
        &args.session,
        &args.source,
        &confirmation,
    )
    .await?;
    println!(
        "PipeFS imported {} entries ({} bytes) as revision {}",
        receipt.preview.entry_count, receipt.preview.byte_count, receipt.revision_id
    );
    Ok(())
}

fn machine_id(authority: &CacheAuthority) -> Result<&str> {
    authority
        .machine_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!("a stable sync machine identity is required; set HI_SYNC_MACHINE_ID")
        })
}
