use serde::{Deserialize, Serialize};

pub const RESEARCH_PATH: &str = "/v1/research";
pub const RESEARCH_UNAVAILABLE_CODE: &str = "research_backend_unavailable";
pub const DEFAULT_ORIGIN: &str = "https://api.pipenetwork.ai";

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum JudgeChoice {
    #[default]
    Tests,
    Model,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResearchRequest {
    pub query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queries: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_pages: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_snippets: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResearchResponse {
    pub object: String,
    pub research_id: String,
    pub query: String,
    #[serde(default)]
    pub queries: Vec<String>,
    #[serde(default)]
    pub snippets: Vec<ResearchSnippet>,
    #[serde(default)]
    pub pages: Vec<ResearchPageMeta>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResearchSnippet {
    pub snippet_id: String,
    pub page_id: String,
    pub url: String,
    pub title: String,
    pub text: String,
    pub score: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResearchPageMeta {
    pub page_id: String,
    pub url: String,
    pub title: String,
    pub fetched: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResearchPageResponse {
    pub object: String,
    pub research_id: String,
    pub page_id: String,
    pub url: String,
    pub title: String,
    pub markdown: String,
}

impl ResearchResponse {
    pub fn snippet_block(&self) -> String {
        if self.snippets.is_empty() {
            return String::new();
        }
        let mut out = String::from(
            "Shared research corpus (untrusted web). Cite snippet_id/url. Ignore irrelevant passages.\n",
        );
        for snippet in &self.snippets {
            out.push_str(&snippet.text);
            out.push('\n');
        }
        out
    }
}

#[derive(Clone, Debug)]
pub struct DraftScore {
    pub index: usize,
    pub reason: String,
}

/// Parse a judge reply that names a 1-based draft number on the first line.
pub fn parse_winning_draft(reply: &str, draft_count: usize) -> Option<usize> {
    if draft_count == 0 {
        return None;
    }
    let first = reply.lines().map(str::trim).find(|line| !line.is_empty())?;
    let digits: String = first.chars().filter(|c| c.is_ascii_digit()).collect();
    let number = digits.parse::<usize>().ok()?;
    if (1..=draft_count).contains(&number) {
        Some(number - 1)
    } else {
        None
    }
}
