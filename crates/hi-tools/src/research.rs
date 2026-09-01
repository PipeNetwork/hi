//! Pipe `research` / `research_read` tools. Thin HTTP client of `/v1/research`.

use anyhow::{Result, bail};
use serde::Deserialize;

use crate::ToolOutcome;

const NOT_CONFIGURED: &str = "Research is not configured (set PIPENETWORK_API_KEY or `/login pipenetwork`). \
This is not a reason to guess URLs with web_fetch.";

pub async fn run_research(arguments: &str) -> Result<ToolOutcome> {
    #[derive(Deserialize)]
    struct Args {
        query: String,
        #[serde(default)]
        research_id: Option<String>,
    }
    let args: Args = serde_json::from_str(arguments).unwrap_or_else(|_| Args {
        query: String::new(),
        research_id: None,
    });
    if let Some(existing) = args
        .research_id
        .as_deref()
        .filter(|id| !id.trim().is_empty())
    {
        return Ok(ToolOutcome::plain(format!(
            "research_id {existing} is already bound for this race. Use research_read for a page, or answer from the shared snippets already in the prompt."
        )));
    }
    if args.query.trim().is_empty() {
        bail!("research needs a non-empty `query`");
    }
    if std::env::var("HI_RESEARCH_SNIPPETS_INJECTED")
        .ok()
        .is_some_and(|value| value == "1")
    {
        return Ok(ToolOutcome::plain(
            "Shared research snippets are already in this prompt. Do not call research() again. Use research_read for a full page.".into(),
        ));
    }
    let client = match hi_research::ResearchClient::from_process_defaults() {
        Ok(client) => client,
        Err(error) if error.is_fail_open() => {
            return Ok(ToolOutcome::plain(NOT_CONFIGURED.into()));
        }
        Err(error) => bail!("{error}"),
    };
    match client.research(args.query.trim()).await {
        Ok(response) => {
            let mut body = format!(
                "research_id={}\nqueries={}\nUse research_read with research_id and page_id for full pages.\n\n{}",
                response.research_id,
                response.queries.join(" | "),
                response.snippet_block()
            );
            if body.trim().is_empty() {
                body = format!("research_id={} returned no snippets.", response.research_id);
            }
            Ok(ToolOutcome::plain(body))
        }
        Err(error) if error.is_fail_open() => Ok(ToolOutcome::plain(format!(
            "Research unavailable ({error}). Do not guess URLs with web_fetch."
        ))),
        Err(error) => bail!("{error}"),
    }
}

pub async fn run_research_read(arguments: &str) -> Result<ToolOutcome> {
    #[derive(Deserialize)]
    struct Args {
        research_id: String,
        page_id: String,
    }
    let args: Args = serde_json::from_str(arguments).unwrap_or_else(|_| Args {
        research_id: String::new(),
        page_id: String::new(),
    });
    if args.research_id.trim().is_empty() || args.page_id.trim().is_empty() {
        bail!("research_read needs `research_id` and `page_id`");
    }
    let client = match hi_research::ResearchClient::from_process_defaults() {
        Ok(client) => client,
        Err(error) if error.is_fail_open() => {
            return Ok(ToolOutcome::plain(NOT_CONFIGURED.into()));
        }
        Err(error) => bail!("{error}"),
    };
    match client
        .read_page(args.research_id.trim(), args.page_id.trim())
        .await
    {
        Ok(page) => Ok(ToolOutcome::plain(format!(
            "research_id={} page_id={} url={}\n# {}\n\n{}",
            page.research_id, page.page_id, page.url, page.title, page.markdown
        ))),
        Err(error) if error.is_fail_open() => Ok(ToolOutcome::plain(format!(
            "Research page unavailable ({error})."
        ))),
        Err(error) => bail!("{error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn missing_key_returns_configured_message() {
        unsafe {
            std::env::remove_var("PIPENETWORK_API_KEY");
        }
        let out = run_research(r#"{"query":"zig http"}"#).await.unwrap();
        assert!(out.content.contains("not configured"));
        assert!(!out.content.contains("web_fetch of https://"));
    }

    #[tokio::test]
    async fn empty_query_rejected() {
        let out = run_research(r#"{"query":""}"#).await;
        assert!(out.is_err());
    }
}
