//! Pipenetwork x402: local Solana keypair payments and paste-sig fallback.

use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use hi_ai::{
    AutoX402Confirmer, PersistableToken, StaticToken, TokenSource, X402_CREDIT_TOKEN_PREFIX,
    X402_PROVIDER_ID, X402Confirmer, X402PaymentRequirements, X402QuoteSummary, X402Settler,
    credit_token_source,
};

use crate::config::{Config, Settings, X402Section, save_config_to};

static CONFIRMER: Mutex<Option<Arc<dyn X402Confirmer>>> = Mutex::new(None);

pub fn set_confirmer(confirmer: Arc<dyn X402Confirmer>) {
    *CONFIRMER.lock().expect("x402 confirmer lock") = Some(confirmer);
}

pub fn clear_confirmer() {
    *CONFIRMER.lock().expect("x402 confirmer lock") = None;
}

pub struct StdioX402Confirmer;

#[async_trait]
impl X402Confirmer for StdioX402Confirmer {
    async fn confirm(&self, quote: &X402QuoteSummary) -> Result<bool> {
        if !io::stdin().is_terminal() {
            bail!("x402 payment needs confirmation; pass --yes or set HI_X402_AUTO_CONFIRM=1");
        }
        eprintln!("{}", quote.prompt_text());
        eprint!("Pay this quote? [y/N] ");
        io::stderr().flush().ok();
        let mut line = String::new();
        io::stdin()
            .read_line(&mut line)
            .context("reading x402 confirmation")?;
        let answer = line.trim().to_ascii_lowercase();
        Ok(matches!(answer.as_str(), "y" | "yes"))
    }

    async fn prompt_signature(&self) -> Result<String> {
        if !io::stdin().is_terminal() {
            bail!(
                "paste an x402 Solana signature in an interactive session, or set HI_X402_KEYPAIR"
            );
        }
        eprint!("Paste Solana signature: ");
        io::stderr().flush().ok();
        let mut line = String::new();
        io::stdin()
            .read_line(&mut line)
            .context("reading x402 signature")?;
        let signature = line.trim().to_string();
        anyhow::ensure!(!signature.is_empty(), "empty signature");
        Ok(signature)
    }
}

pub struct CliX402Settler {
    payer: Option<hi_x402::KeypairPayer>,
    auto_confirm: bool,
}

#[async_trait]
impl X402Settler for CliX402Settler {
    async fn settle(&self, requirements: &X402PaymentRequirements) -> Result<String> {
        let confirmer = if self.auto_confirm {
            Arc::new(AutoX402Confirmer) as Arc<dyn X402Confirmer>
        } else if let Some(confirmer) = CONFIRMER.lock().ok().and_then(|guard| guard.clone()) {
            confirmer
        } else {
            Arc::new(StdioX402Confirmer)
        };
        let summary = hi_ai::quote_summary(requirements)?;
        if !confirmer.confirm(&summary).await? {
            bail!("x402 payment declined");
        }
        if let Some(payer) = &self.payer {
            let timeout = Duration::from_secs(requirements.max_timeout_seconds.max(15));
            return payer.pay(requirements, timeout).await;
        }
        confirmer.prompt_signature().await
    }
}

pub fn pipenetwork_token_source(settings: &Settings) -> Arc<dyn TokenSource> {
    let key = settings.api_key.trim();
    if key.starts_with("pk_live_")
        || (!key.is_empty() && !key.starts_with(X402_CREDIT_TOKEN_PREFIX))
    {
        return Arc::new(StaticToken(settings.api_key.clone()));
    }
    if key.starts_with(X402_CREDIT_TOKEN_PREFIX) {
        return Arc::new(PersistableToken::persisting(
            X402_PROVIDER_ID,
            settings.api_key.clone(),
        ));
    }
    Arc::new(credit_token_source())
}

pub fn is_pairing_key(settings: &Settings) -> bool {
    credential_is_pairing_key(&settings.api_key)
}

pub fn credential_is_pairing_key(key: &str) -> bool {
    let key = key.trim();
    !key.is_empty() && !key.starts_with(X402_CREDIT_TOKEN_PREFIX)
}

pub fn build_settler(settings: &Settings) -> Option<Arc<dyn X402Settler>> {
    if is_pairing_key(settings) {
        return None;
    }
    let payer = settings.x402.keypair.as_ref().and_then(|path| {
        match hi_x402::KeypairPayer::from_file(path, &settings.x402.rpc) {
            Ok(payer) => Some(payer),
            Err(error) => {
                eprintln!(
                    "warning: could not load HI_X402_KEYPAIR {}: {error:#}",
                    path.display()
                );
                None
            }
        }
    });
    if payer.is_none() && !settings.x402.paste_sig {
        return None;
    }
    Some(Arc::new(CliX402Settler {
        payer,
        auto_confirm: settings.x402.auto_confirm,
    }))
}

pub fn login(config: &mut Config, config_path: Option<&Path>) -> Result<()> {
    if hi_ai::pipenetwork_auth::has_credential() {
        println!(
            "a pipenetwork pairing key is already stored; x402 will not run until \
             `/logout pipenetwork`. `/login x402` still records the Solana keypair."
        );
    }
    let env_path = std::env::var("HI_X402_KEYPAIR")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .map(PathBuf::from);
    let path = env_path.clone().or_else(|| {
        config
            .x402
            .as_ref()
            .and_then(|section| section.keypair.clone())
    });
    if let Some(path) = path.as_ref() {
        let payer = hi_x402::KeypairPayer::from_file(path, hi_x402::DEFAULT_RPC_URL)
            .with_context(|| format!("loading {}", path.display()))?;
        println!(
            "x402 ready (pubkey {}). First turn quotes USDC, then stores a credit token.",
            payer.pubkey()
        );
        println!(
            "Run /provider pipenetwork to use it. Cap is $1.00 unless HI_X402_MAX_USD is set."
        );
    } else {
        println!(
            "x402 enabled without a local keypair — paste a Solana signature when quoted \
             (plain REPL), or set HI_X402_KEYPAIR to sign in-process."
        );
    }
    let section = config.x402.get_or_insert_with(X402Section::default);
    section.enabled = Some(true);
    if env_path.is_none()
        && let Some(path) = path
    {
        section.keypair = Some(path);
    }
    if let Some(config_path) = config_path {
        save_config_to(config, config_path)
            .with_context(|| format!("saving x402 config to {}", config_path.display()))?;
    }
    Ok(())
}

pub fn logout() -> Result<()> {
    hi_ai::x402_logout()
}
