//! In-process Solana USDC transfer used by hi's x402 hop.
//!
//! Builds an idempotent destination ATA (if needed), an SPL token
//! `transferChecked` of the quote's exact minor units, and an `spl-memo`
//! instruction carrying `extra.memo`. Live RPC is skipped in unit tests.

use std::path::Path;
use std::str::FromStr;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use hi_ai::{X402_USDC_MINT_MAINNET, X402PaymentRequirements, X402Settler};
use solana_client::nonblocking::rpc_client::RpcClient;
use solana_sdk::commitment_config::CommitmentConfig;
use solana_sdk::instruction::Instruction;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::{Keypair, Signer};
use solana_sdk::transaction::Transaction;
use spl_associated_token_account::get_associated_token_address;
use spl_associated_token_account::instruction::create_associated_token_account_idempotent;
use spl_token::ID as TOKEN_PROGRAM_ID;

pub const DEFAULT_RPC_URL: &str = "https://api.mainnet-beta.solana.com";
pub const USDC_DECIMALS: u8 = 6;
/// spl-memo program id (`MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr`).
pub const MEMO_PROGRAM_ID: &str = "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr";

pub fn load_keypair(path: &Path) -> Result<Keypair> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading Solana keypair {}", path.display()))?;
    let bytes: Vec<u8> = serde_json::from_str(text.trim())
        .with_context(|| format!("Solana keypair {} is not a JSON byte array", path.display()))?;
    Keypair::try_from(bytes.as_slice())
        .map_err(|error| anyhow::anyhow!("invalid Solana keypair {}: {error}", path.display()))
}

pub struct KeypairPayer {
    keypair: Keypair,
    rpc_url: String,
}

impl KeypairPayer {
    pub fn from_file(path: &Path, rpc_url: impl Into<String>) -> Result<Self> {
        Ok(Self {
            keypair: load_keypair(path)?,
            rpc_url: rpc_url.into(),
        })
    }

    pub fn pubkey(&self) -> String {
        self.keypair.pubkey().to_string()
    }

    pub async fn pay(
        &self,
        requirements: &X402PaymentRequirements,
        timeout: Duration,
    ) -> Result<String> {
        let instructions = payment_instructions(&self.keypair.pubkey(), requirements)?;
        let client =
            RpcClient::new_with_commitment(self.rpc_url.clone(), CommitmentConfig::confirmed());
        let send = async {
            let blockhash = client
                .get_latest_blockhash()
                .await
                .context("fetching Solana blockhash")?;
            let tx = Transaction::new_signed_with_payer(
                &instructions,
                Some(&self.keypair.pubkey()),
                &[&self.keypair],
                blockhash,
            );
            let signature = client
                .send_and_confirm_transaction(&tx)
                .await
                .context("sending Solana USDC transfer")?;
            Ok::<String, anyhow::Error>(signature.to_string())
        };
        match tokio::time::timeout(timeout, send).await {
            Ok(result) => result,
            Err(_) => bail!("Solana confirmation timed out after {}s", timeout.as_secs()),
        }
    }
}

#[async_trait]
impl X402Settler for KeypairPayer {
    async fn settle(&self, requirements: &X402PaymentRequirements) -> Result<String> {
        let timeout = Duration::from_secs(requirements.max_timeout_seconds.max(15));
        self.pay(requirements, timeout).await
    }
}

pub fn payment_instructions(
    payer: &Pubkey,
    requirements: &X402PaymentRequirements,
) -> Result<Vec<Instruction>> {
    let mint = parse_pubkey(&requirements.asset, "mint")?;
    let expected_mint =
        Pubkey::from_str(X402_USDC_MINT_MAINNET).expect("USDC mint constant is a valid pubkey");
    if mint != expected_mint {
        bail!("refusing to transfer mint {mint}; x402 v1 only pays USDC {X402_USDC_MINT_MAINNET}");
    }
    let pay_to = parse_pubkey(&requirements.pay_to, "payTo")?;
    let amount: u64 = requirements
        .amount
        .parse()
        .with_context(|| format!("x402 amount {:?} is not a u64", requirements.amount))?;
    let memo = requirements
        .memo()
        .or(requirements.quote_id())
        .context("x402 quote is missing extra.memo")?;
    let memo = if memo.starts_with("x402_") {
        memo.to_string()
    } else {
        format!("x402_{memo}")
    };
    Ok(usdc_transfer_instructions(
        *payer, pay_to, mint, amount, &memo,
    ))
}

pub fn usdc_transfer_instructions(
    payer: Pubkey,
    pay_to: Pubkey,
    mint: Pubkey,
    amount: u64,
    memo: &str,
) -> Vec<Instruction> {
    let source_ata = get_associated_token_address(&payer, &mint);
    let dest_ata = get_associated_token_address(&pay_to, &mint);
    let create_dest =
        create_associated_token_account_idempotent(&payer, &pay_to, &mint, &TOKEN_PROGRAM_ID);
    let transfer = transfer_checked(payer, source_ata, mint, dest_ata, amount, USDC_DECIMALS);
    let memo_ix = spl_memo::build_memo(memo.as_bytes(), &[&payer]);
    vec![create_dest, transfer, memo_ix]
}

fn transfer_checked(
    authority: Pubkey,
    source: Pubkey,
    mint: Pubkey,
    destination: Pubkey,
    amount: u64,
    decimals: u8,
) -> Instruction {
    spl_token::instruction::transfer_checked(
        &TOKEN_PROGRAM_ID,
        &source,
        &mint,
        &destination,
        &authority,
        &[],
        amount,
        decimals,
    )
    .expect("transferChecked instruction is well-formed")
}

fn parse_pubkey(value: &str, label: &str) -> Result<Pubkey> {
    Pubkey::from_str(value.trim()).with_context(|| format!("invalid x402 {label} pubkey"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_ai::{X402_SCHEME_EXACT, X402_SOLANA_MAINNET, X402PaymentRequirements};

    fn quote(amount: &str) -> X402PaymentRequirements {
        X402PaymentRequirements {
            scheme: X402_SCHEME_EXACT.to_string(),
            network: X402_SOLANA_MAINNET.to_string(),
            amount: amount.to_string(),
            asset: X402_USDC_MINT_MAINNET.to_string(),
            pay_to: "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM".to_string(),
            max_timeout_seconds: 60,
            extra: Some(serde_json::json!({
                "memo": "x402_quote1",
                "quoteId": "quote1"
            })),
        }
    }

    #[test]
    fn instructions_target_usdc_ata_exact_amount_and_memo() {
        let payer = Pubkey::new_unique();
        let requirements = quote("20000");
        let instructions = payment_instructions(&payer, &requirements).unwrap();
        assert_eq!(instructions.len(), 3);
        let mint = Pubkey::from_str(X402_USDC_MINT_MAINNET).unwrap();
        let pay_to = Pubkey::from_str(&requirements.pay_to).unwrap();
        let dest_ata = get_associated_token_address(&pay_to, &mint);
        let source_ata = get_associated_token_address(&payer, &mint);

        assert_eq!(instructions[0].program_id, spl_associated_token_account::ID);

        let transfer = &instructions[1];
        assert_eq!(transfer.program_id, TOKEN_PROGRAM_ID);
        let dest_listed = transfer
            .accounts
            .iter()
            .any(|account| account.pubkey == dest_ata);
        let source_listed = transfer
            .accounts
            .iter()
            .any(|account| account.pubkey == source_ata);
        assert!(source_listed, "transfer must spend the payer USDC ATA");
        assert!(dest_listed, "transfer must credit the payTo USDC ATA");
        // spl-token transferChecked: tag 12, amount LE u64, decimals u8
        assert_eq!(transfer.data[0], 12);
        assert_eq!(&transfer.data[1..9], 20000u64.to_le_bytes());
        assert_eq!(transfer.data[9], USDC_DECIMALS);

        let memo = &instructions[2];
        assert_eq!(memo.program_id, Pubkey::from_str(MEMO_PROGRAM_ID).unwrap());
        assert_eq!(memo.data, b"x402_quote1");
    }

    #[test]
    fn refuses_a_non_usdc_mint() {
        let mut requirements = quote("20000");
        requirements.asset = "So11111111111111111111111111111111111111112".into();
        let error = payment_instructions(&Pubkey::new_unique(), &requirements)
            .unwrap_err()
            .to_string();
        assert!(error.contains("USDC"), "{error}");
    }

    #[test]
    fn load_keypair_rejects_non_array_json() {
        let dir = std::env::temp_dir().join(format!(
            "hi-x402-kp-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("id.json");
        std::fs::write(&path, "{\"secret\":true}").unwrap();
        let error = load_keypair(&path).unwrap_err().to_string();
        assert!(error.contains("JSON byte array"), "{error}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
