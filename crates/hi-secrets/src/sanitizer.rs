//! Core sanitization logic — regex-based secret redaction.
//!
//! Ported and adapted from grok-build's `xai-grok-secrets` crate. The patterns
//! and redaction strategy are preserved; the `url` dependency is used directly
//! rather than via a workspace re-export.

use std::borrow::Cow;
use std::sync::LazyLock;

use regex::{Regex, RegexSet};

const REDACTED: &str = "[REDACTED_SECRET]";
const REDACTED_URL_VALUE: &str = "redacted";
const REDACTED_USER_SEGMENT: &str = "<user>";

/// Vendor API keys with `sk-`/`sk_` prefixes and xAI (`xai-`) keys. `\b`-anchored so
/// `task-`/`disk-`/`risk-` don't fold a stray `sk-`.
static API_KEY_PREFIX_REGEX: LazyLock<Regex> =
    LazyLock::new(|| compile(r"\b(?:sk[-_]|xai-)[A-Za-z0-9_-]{20,}"));
/// AWS long-term (`AKIA`) and temporary (`ASIA`) access-key IDs.
static AWS_ACCESS_KEY_REGEX: LazyLock<Regex> =
    LazyLock::new(|| compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"));
/// GitHub PATs: classic (`ghp_`/`gho_`/`ghu_`/`ghs_`/`ghr_`) + fine-grained
/// (`github_pat_`).
static GITHUB_TOKEN_REGEX: LazyLock<Regex> =
    LazyLock::new(|| compile(r"\b(?:gh[opusr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"));
/// GitLab (`glpat-`) and Slack (`xoxa-`/`xoxb-`/`xoxp-`/`xapp-`) tokens.
static VENDOR_TOKEN_REGEX: LazyLock<Regex> =
    LazyLock::new(|| compile(r"\b(?:glpat-|xox[abp]-|xapp-)[A-Za-z0-9-]{10,}"));
/// Google API keys (`AIza` + 35 chars).
static GOOGLE_API_KEY_REGEX: LazyLock<Regex> =
    LazyLock::new(|| compile(r"\bAIza[0-9A-Za-z_-]{35}"));
/// PEM private-key block (any key type), base64 body included. `(?s)` so `.`
/// spans the newline-delimited body.
static PEM_PRIVATE_KEY_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    compile(r"(?s)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----")
});
static BEARER_TOKEN_REGEX: LazyLock<Regex> =
    LazyLock::new(|| compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{16,}\b"));
/// Bare JWT (`eyJ...header.payload.signature`) with no `Bearer`/`sk-` prefix —
/// the shape used by deployment keys and OIDC tokens.
static JWT_REGEX: LazyLock<Regex> =
    LazyLock::new(|| compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"));
/// Values of credential-named assignments. The value floor is 6 chars; the
/// value charset excludes `&` so `?token=x&next=y` stops at the parameter
/// boundary. Whitespace inside the value is only accepted between quotes, so
/// `password=hunter2;rm -rf` cannot smuggle payload text into the redaction
/// span. High-entropy values below the old 8-char floor (and non-standard
/// formats no vendor regex recognizes) are redacted; low-entropy words,
/// hashes, and placeholders pass through — see `should_redact_assignment`.
static SECRET_ASSIGNMENT_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    compile(
        r#"(?ix)
        \b(
            api[_-]?key
          | (?:access|refresh|id)[_-]token
          | token
          | secret
          | client[_-]secret
          | password
        )\b
        (\s*["']?\s*[:=]\s*)
        (?:
            "(?<dq>[^"\s][^"]{4,})"
          | '(?<sq>[^'\s][^']{4,})'
          | (?<bare>[^\s"',&]{6,})
        )
        "#,
    )
});

/// Replacement for credential assignments: keep the key, separator, and
/// quoting, redact the value — but only when it looks like a real secret.
fn redact_assignments(text: &str) -> String {
    SECRET_ASSIGNMENT_REGEX
        .replace_all(text, |caps: &regex::Captures<'_>| {
            let value = caps
                .name("dq")
                .or_else(|| caps.name("sq"))
                .or_else(|| caps.name("bare"))
                .map_or("", |m| m.as_str());
            if should_redact_assignment(value) {
                let quote = if caps.name("dq").is_some() {
                    "\""
                } else if caps.name("sq").is_some() {
                    "'"
                } else {
                    ""
                };
                // Group 2 (the separator) absorbs the key's optional closing
                // quote and any quote/space before the `:`/`=`, so re-emitting
                // it keeps `"token":"…"` well-formed while the value redacts.
                format!("{}{}{quote}{REDACTED}{quote}", &caps[1], &caps[2])
            } else {
                caps[0].to_string()
            }
        })
        .into_owned()
}

/// Whether a JSON object key names a credential, matching the alternation in
/// [`SECRET_ASSIGNMENT_REGEX`]. Case-insensitive; allows `_`/`-` separators.
fn is_credential_key(key: &str) -> bool {
    // Normalize separators so `api_key` / `api-key` / `apikey` collapse.
    let mut normalized = String::with_capacity(key.len());
    for c in key.chars() {
        match c {
            '_' | '-' => {}
            _ => normalized.push(c.to_ascii_lowercase()),
        }
    }
    matches!(
        normalized.as_str(),
        "apikey"
            | "accesstoken"
            | "refreshtoken"
            | "idtoken"
            | "token"
            | "secret"
            | "clientsecret"
            | "password"
    )
}

/// Entropy gate for assignment values: redact only values that look like
/// real secrets rather than words, identifiers, hashes, or placeholders.
fn should_redact_assignment(value: &str) -> bool {
    if value.len() < 6 {
        return false;
    }
    // Pure-hex tokens of hash length are digests/SHAs/UUIDs, not credentials.
    if value.len() >= 16 && value.len() <= 128 && value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return false;
    }
    // JWT-shaped values are already redacted by JWT_REGEX.
    if value.starts_with("eyJ") {
        return false;
    }
    let lower = value.to_ascii_lowercase();
    const PLACEHOLDERS: &[&str] = &[
        "changeme",
        "password",
        "placeholder",
        "redacted",
        "example",
        "your_key",
        "your-key",
        "your_token",
        "your-token",
        "dummy",
        "sample",
        "testing",
        "default",
    ];
    if PLACEHOLDERS.iter().any(|p| lower.contains(p)) {
        return false;
    }
    // Credentials are high-entropy: random base62/base64/hex sits around
    // 4-6 bits/char while english words and identifiers sit near 2-3. Short
    // values (under ~12 chars) cannot reach the long-value ceiling even when
    // every character is unique, so they earn a bonus for mixed character
    // classes — symbols/digits are rare in words, common in generated keys.
    let mut entropy = shannon_entropy(value);
    if value.len() < 12
        && value.bytes().any(|b| !b.is_ascii_alphanumeric())
        && value.bytes().any(|b| b.is_ascii_alphanumeric())
    {
        entropy += 0.6;
    }
    entropy >= 3.2
}

/// Shannon entropy in bits per byte over the token's character distribution.
fn shannon_entropy(token: &str) -> f64 {
    let mut counts = [0u32; 256];
    for &b in token.as_bytes() {
        counts[b as usize] += 1;
    }
    let len = token.len() as f64;
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = f64::from(c) / len;
            -p * p.log2()
        })
        .sum()
}

static SENSITIVE_QUERY_PARAMS: &[&str] = &[
    "access_token",
    "api_key",
    "assertion",
    "auth",
    "client_secret",
    "code",
    "code_verifier",
    "id_token",
    "key",
    "password",
    "refresh_token",
    "requested_token",
    "session_id",
    "state",
    "subject_token",
    "token",
];

/// Excludes trailing punctuation so backticks/brackets in surrounding text
/// don't get folded into the URL match.
static URL_REGEX: LazyLock<Regex> = LazyLock::new(|| compile(r#"https?://[^\s"'<>(){}\[\],;`]+"#));

static MATCH_ANY: LazyLock<RegexSet> = LazyLock::new(|| {
    RegexSet::new([
        API_KEY_PREFIX_REGEX.as_str(),
        AWS_ACCESS_KEY_REGEX.as_str(),
        GITHUB_TOKEN_REGEX.as_str(),
        VENDOR_TOKEN_REGEX.as_str(),
        GOOGLE_API_KEY_REGEX.as_str(),
        PEM_PRIVATE_KEY_REGEX.as_str(),
        BEARER_TOKEN_REGEX.as_str(),
        JWT_REGEX.as_str(),
        URL_REGEX.as_str(),
        SECRET_ASSIGNMENT_REGEX.as_str(),
    ])
    .expect("redact_secrets RegexSet")
});

/// Redact known secret shapes from an arbitrary string.
///
/// Returns a borrowed `Cow` when nothing matched (zero-allocation fast path)
/// and an owned `Cow` with replacements applied otherwise.
pub fn redact_secrets(input: &str) -> Cow<'_, str> {
    if !MATCH_ANY.is_match(input) {
        return Cow::Borrowed(input);
    }
    let s = PEM_PRIVATE_KEY_REGEX.replace_all(input, REDACTED);
    let s = API_KEY_PREFIX_REGEX.replace_all(&s, REDACTED);
    let s = AWS_ACCESS_KEY_REGEX.replace_all(&s, REDACTED);
    let s = GITHUB_TOKEN_REGEX.replace_all(&s, REDACTED);
    let s = VENDOR_TOKEN_REGEX.replace_all(&s, REDACTED);
    let s = GOOGLE_API_KEY_REGEX.replace_all(&s, REDACTED);
    let s = BEARER_TOKEN_REGEX.replace_all(&s, format!("Bearer {REDACTED}"));
    let s = JWT_REGEX.replace_all(&s, REDACTED);
    let s = redact_urls_in(&s);
    Cow::Owned(redact_assignments(&s))
}

/// Walk all string values in a JSON tree, applying `f` in place.
///
/// Use [`redact_json_string_values`] for the standard scrub; use this directly
/// only when composing a custom one.
/// Walk all string values in a JSON tree, applying `f` in place.
///
/// Use [`redact_json_string_values`] for the standard scrub; use this directly
/// only when composing a custom one. This public form has no key context, so
/// it cannot apply the credential-key entropy gate — prefer the standard
/// scrub for redaction.
pub fn walk_json_strings(value: &mut serde_json::Value, f: &mut impl FnMut(&mut String)) {
    walk_json_strings_ctx(value, None, &mut |s, _| f(s));
}

/// Internal walk that threads the owning object key down to each string value,
/// so redaction can recognize `{"token": "…"}` even when the value alone has
/// no `key=` shape. Non-string parents pass no key.
fn walk_json_strings_ctx(
    value: &mut serde_json::Value,
    key: Option<&str>,
    f: &mut impl FnMut(&mut String, Option<&str>),
) {
    match value {
        serde_json::Value::String(s) => f(s, key),
        serde_json::Value::Array(arr) => arr
            .iter_mut()
            .for_each(|v| walk_json_strings_ctx(v, key, f)),
        serde_json::Value::Object(map) => map
            .iter_mut()
            .for_each(|(k, v)| walk_json_strings_ctx(v, Some(k.as_str()), f)),
        _ => {}
    }
}

/// Redact secrets in every string value within a JSON tree (in place).
///
/// Carries the owning object key into the entropy gate, so a credential-named
/// key's value redacts even when the value is a bare string with no `key=`
/// shape (e.g. `{"token": "abc…"}`).
pub fn redact_json_string_values(value: &mut serde_json::Value) {
    walk_json_strings_ctx(value, None, &mut |s, key| {
        // If this string is the direct value of a credential-named key, gate
        // on the value's own entropy — no `key=` prefix is present in the
        // string itself, so redact_secrets alone would miss it.
        if key.is_some_and(is_credential_key) && should_redact_assignment(s) {
            *s = REDACTED.to_string();
            return;
        }
        if let Cow::Owned(replaced) = redact_secrets(s) {
            *s = replaced;
        }
    });
}

fn redact_urls_in(text: &str) -> String {
    URL_REGEX
        .replace_all(text, |caps: &regex::Captures<'_>| {
            let raw = &caps[0];
            url::Url::parse(raw).map_or_else(
                |_| raw.to_owned(),
                |mut url| {
                    redact_url(&mut url);
                    url.to_string()
                },
            )
        })
        .into_owned()
}

/// Strip credentials and sensitive query params from a URL (in place).
///
/// Removes userinfo (username/password), the fragment, and replaces sensitive
/// query-parameter values (e.g. `access_token`, `api_key`) with `"redacted"`.
pub fn redact_url(url: &mut url::Url) {
    let _ = url.set_username("");
    let _ = url.set_password(None);
    url.set_fragment(None);

    let Some(query) = url.query().map(str::to_owned) else {
        return;
    };
    let pairs: Vec<(String, String)> = url::form_urlencoded::parse(query.as_bytes())
        .map(|(k, v)| {
            let key = k.into_owned();
            let value = if SENSITIVE_QUERY_PARAMS
                .iter()
                .any(|s| s.eq_ignore_ascii_case(&key))
            {
                REDACTED_URL_VALUE.to_owned()
            } else {
                v.into_owned()
            };
            (key, value)
        })
        .collect();
    if pairs.is_empty() {
        url.set_query(None);
        return;
    }
    let mut serializer = url::form_urlencoded::Serializer::new(String::new());
    for (k, v) in &pairs {
        serializer.append_pair(k, v);
    }
    url.set_query(Some(&serializer.finish()));
}

/// Redact the current user's home directory path and username segments from a string.
///
/// Replaces the home directory prefix with `~` and any whole path segment matching
/// the username with `<user>`. Falls back to a regex-based `/Users/<name>` →
/// `/Users/<user>` redaction when `HOME` is unset.
pub fn redact_user_paths(input: &str) -> Cow<'_, str> {
    let home = std::env::var("HOME").ok();
    let usernames: Vec<String> = home
        .as_deref()
        .and_then(|h| {
            std::path::Path::new(h)
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| vec![n.to_owned()])
        })
        .unwrap_or_default();

    // Stage 1: home prefix → ~
    let stage1: Cow<'_, str> = match home.as_deref() {
        Some(h) if !h.is_empty() && input.contains(h) => Cow::Owned(replace_home_prefix(input, h)),
        _ => Cow::Borrowed(input),
    };

    // Stage 2: username segments → <user>
    let stage2: String = if !usernames.is_empty() && stage1.contains(usernames[0].as_str()) {
        redact_username_segments(&stage1, &usernames)
    } else {
        stage1.into_owned()
    };

    // Stage 3: regex backstop for /Users/<name> or /home/<name> when HOME is unset.
    if home.is_none() {
        static BACKSTOP: LazyLock<Regex> =
            LazyLock::new(|| compile(r"(?i)/(?:Users|home)/[A-Za-z0-9._-]+"));
        if BACKSTOP.is_match(&stage2) {
            return Cow::Owned(BACKSTOP.replace_all(&stage2, "/Users/<user>").into_owned());
        }
    }

    if stage2 != input {
        Cow::Owned(stage2)
    } else {
        Cow::Borrowed(input)
    }
}

/// Whole-segment `home` -> `~` so `/Users/bob` doesn't fold over `/Users/bobby/...`.
fn replace_home_prefix(input: &str, home: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(idx) = rest.find(home) {
        let (before, tail) = rest.split_at(idx);
        let after = &tail[home.len()..];
        let prev_ok = before.chars().last().is_none_or(is_segment_boundary);
        let next_ok = after.chars().next().is_none_or(is_segment_boundary);
        out.push_str(before);
        if prev_ok && next_ok {
            out.push('~');
        } else {
            out.push_str(home);
        }
        rest = after;
    }
    out.push_str(rest);
    out
}

/// Replace whole `/`- or `\`-delimited segments equal to a username with
/// `<user>`. Case-insensitive on Windows (NTFS), case-sensitive elsewhere.
fn redact_username_segments(value: &str, usernames: &[String]) -> String {
    let mut out = String::with_capacity(value.len());
    let mut buf = String::new();
    for ch in value.chars() {
        if is_segment_boundary(ch) {
            push_username_segment(&mut out, &buf, usernames);
            buf.clear();
            out.push(ch);
        } else {
            buf.push(ch);
        }
    }
    push_username_segment(&mut out, &buf, usernames);
    out
}

fn push_username_segment(out: &mut String, segment: &str, usernames: &[String]) {
    let matches = if cfg!(windows) {
        usernames.iter().any(|u| u.eq_ignore_ascii_case(segment))
    } else {
        usernames.iter().any(|u| u == segment)
    };
    out.push_str(if matches {
        REDACTED_USER_SEGMENT
    } else {
        segment
    });
}

fn is_segment_boundary(ch: char) -> bool {
    ch == '/' || ch == '\\'
}

fn compile(pattern: &str) -> Regex {
    Regex::new(pattern).unwrap_or_else(|e| panic!("invalid regex `{pattern}`: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tripwire: if you add a regex to `MATCH_ANY`, also add a redaction
    /// pass in `redact_secrets` (and update this count).
    #[test]
    fn match_any_count_matches_redact_secrets_passes() {
        assert_eq!(MATCH_ANY.patterns().len(), 10);
    }

    #[test]
    fn no_match_returns_borrowed() {
        assert!(matches!(
            redact_secrets("just a normal log line"),
            Cow::Borrowed(_)
        ));
        assert!(matches!(redact_secrets("model=grok-3"), Cow::Borrowed(_)));
    }

    /// Joins fixture fragments at runtime so realistic-looking fake tokens
    /// never appear contiguously in the source text.
    fn fixture(parts: &[&str]) -> String {
        parts.concat()
    }

    #[test]
    fn redacts_known_secret_shapes() {
        let cases = [
            (
                fixture(&["key: xai-", "abc123XYZdef456GHIjkl789"]),
                "xai api key",
            ),
            (
                fixture(&["aws AKIA", "ABCDEFGHIJKLMNOP key"]),
                "aws access key",
            ),
            (
                fixture(&["Authorization: Bearer eyJhbGciOiJIUzI1NiJ9", ".foo.bar.baz"]),
                "bearer token",
            ),
            (fixture(&["api_key=", "ABCDEFGHIJ"]), "key=value"),
            (
                fixture(&["refresh_token: \"rt_", "abc1234567\""]),
                "compound token name",
            ),
            (
                fixture(&[
                    "deployment key eyJhbGciOiJIUzI1NiJ9",
                    ".eyJzdWIiOiJ4In0.signature",
                ]),
                "bare jwt",
            ),
            (
                fixture(&["ghp_", "abcdefghijklmnopqrstuvwxyzABCD"]),
                "github pat",
            ),
            (
                fixture(&["glpat-", "abcdefghijklmnopqrstuvwxyz"]),
                "gitlab token",
            ),
            (
                fixture(&["AIza", "SyAabcdefghijklmnopqrstuvwxyz1234567"]),
                "google api key",
            ),
        ];
        for (input, label) in cases {
            let out = redact_secrets(&input);
            assert!(
                out.contains(REDACTED),
                "{label}: expected redaction in {out:?}"
            );
            assert!(
                !out.contains(&input),
                "{label}: input survived redaction: {out:?}"
            );
        }
    }

    #[test]
    fn redacts_pem_private_key() {
        let input =
            "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----";
        let out = redact_secrets(input);
        assert!(out.contains(REDACTED), "PEM not redacted: {out}");
        assert!(!out.contains("MIIEpAIBAAKCAQEA"));
    }

    #[test]
    fn redacts_url_credentials_and_sensitive_params() {
        let input = "https://user:secretpw@host.com/cb?code=ABC123XYZ&page=2#access_token=DEF456";
        let out = redact_secrets(input);
        assert!(!out.contains("user"), "userinfo leaked: {out}");
        assert!(!out.contains("secretpw"), "password leaked: {out}");
        assert!(!out.contains("ABC123XYZ"), "OAuth code leaked: {out}");
        assert!(!out.contains("DEF456"), "fragment token leaked: {out}");
        assert!(out.contains("host.com/cb"), "lost host/path: {out}");
        assert!(out.contains("page=2"), "lost benign param: {out}");
    }

    #[test]
    fn redact_json_string_values_walks_tree() {
        let mut value = serde_json::json!({
            "api_key": "sk-abcdefghijklmnopqrstuvwxyz123456",
            "nested": {
                "token": "ghp_abcdefghijklmnopqrstuvwxyzABCD",
                "safe": "normal text"
            },
            "arr": ["Bearer eyJhbGciOiJIUzI1NiJ9.abc.defghij", "ok"]
        });
        redact_json_string_values(&mut value);
        let s = serde_json::to_string(&value).unwrap();
        assert!(s.contains(REDACTED), "no redaction in: {s}");
        assert!(!s.contains("sk-abcdefghijklmnopqrstuvwxyz123456"));
        assert!(!s.contains("ghp_abcdefghijklmnopqrstuvwxyzABCD"));
        assert!(s.contains("normal text"), "safe text was altered: {s}");
        assert!(s.contains("\"ok\""), "safe array element altered: {s}");
    }

    #[test]
    fn redact_json_string_values_gates_credential_keys_without_prefix() {
        // The whole point of key context: a bare base64 value under a
        // credential-named key redacts even though the string alone has no
        // `key=` shape and no vendor prefix. Non-credential keys with the
        // same value, and low-entropy values under credential keys, survive.
        let secret = fixture(&["dGhpc2lz", "YXJhbmRvbWJhc2U2", "NHNlY3JldA=="]);
        let mut value = serde_json::json!({
            "token": secret,
            "client_secret": secret,
            "config": { "password": secret },
            "note": secret,                       // not a credential key
            "nested_password": secret,            // `_` joins words; not a key
            "password": "hunter2",                // credential key, low entropy
            "count": 42,
            "arr": [secret],                      // array, no key context
        });
        redact_json_string_values(&mut value);
        assert_eq!(value["token"], REDACTED, "token not redacted");
        assert_eq!(
            value["client_secret"], REDACTED,
            "client_secret not redacted"
        );
        assert_eq!(
            value["config"]["password"], REDACTED,
            "nested credential key not redacted"
        );
        assert_eq!(value["note"], secret, "non-credential key was redacted");
        assert_eq!(
            value["nested_password"], secret,
            "compound key should not redact (matches regex \\b behavior)"
        );
        assert_eq!(value["password"], "hunter2", "low-entropy value redacted");
        assert_eq!(value["arr"][0], secret, "array element (no key) redacted");
        assert_eq!(value["count"], 42);
    }

    #[test]
    fn is_credential_key_matches_regex_alternation() {
        for key in [
            "api_key",
            "apikey",
            "api-key",
            "API_KEY",
            "token",
            "access_token",
            "refresh-token",
            "idToken",
            "secret",
            "client_secret",
            "password",
        ] {
            assert!(is_credential_key(key), "{key} should be a credential key");
        }
        for key in ["note", "count", "page", "tokens", "tokenizer", "monkey"] {
            assert!(
                !is_credential_key(key),
                "{key} should NOT be a credential key"
            );
        }
    }

    #[test]
    fn redact_user_paths_replaces_home() {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        let input = format!("{home}/secret/file.rs");
        let out = redact_user_paths(&input);
        assert!(out.contains("~/secret/file.rs"), "home not redacted: {out}");
    }

    #[test]
    fn redact_url_strips_credentials_and_fragment() {
        let mut url =
            url::Url::parse("https://user:pw@idp.example.com/cb?code=ABC123XYZ&page=2#frag=ok")
                .unwrap();
        redact_url(&mut url);
        let out = url.to_string();
        assert!(!out.contains("user"), "userinfo leaked: {out}");
        assert!(!out.contains("pw"), "password leaked: {out}");
        assert!(!out.contains("ABC123XYZ"), "OAuth code leaked: {out}");
        assert!(out.contains("idp.example.com/cb"), "lost host/path: {out}");
        assert!(out.contains("page=2"), "lost benign param: {out}");
    }

    #[test]
    fn redacts_short_high_entropy_assignment_values() {
        // Below the old 8-char floor, but unmistakably a credential.
        let input = fixture(&["token=", "aZ9#kQ2!"]);
        let out = redact_secrets(&input);
        assert!(out.contains(REDACTED), "short secret not redacted: {out}");
        assert!(out.contains("token="), "key name lost: {out}");
    }

    #[test]
    fn redacts_non_standard_base64_assignment_values() {
        // No vendor prefix, no `Bearer`, not a JWT — a raw base64 secret.
        let secret = fixture(&["dGhpc2lz", "YXJhbmRvbWJhc2U2", "NHNlY3JldA=="]);
        let input = format!("client_secret={secret}");
        let out = redact_secrets(&input);
        assert!(out.contains(REDACTED), "base64 secret not redacted: {out}");
        assert!(!out.contains(&secret), "secret survived: {out}");
    }

    #[test]
    fn redacts_json_quoted_credential_keys() {
        // JSON form: the key's closing quote sits between the key name and the
        // `:`. The separator absorbs it so the value redacts and the document
        // stays valid JSON.
        let secret = fixture(&["dGhpc2lz", "YXJhbmRvbWJhc2U2", "NHNlY3JldA=="]);
        let input = format!(r#"{{"token":"{secret}","page":2}}"#);
        let out = redact_secrets(&input);
        assert!(!out.contains(&secret), "json secret leaked: {out}");
        assert!(out.contains(REDACTED), "no redaction in json: {out}");
        let parsed: serde_json::Value =
            serde_json::from_str(&out).expect("redacted json must stay valid");
        assert_eq!(parsed["page"], 2, "benign field altered: {out}");
    }

    #[test]
    fn keeps_low_entropy_assignment_values() {
        // English words, placeholders, hashes, and JWTs must survive.
        for (value, label) in [
            ("hunter2", "short english word"),
            ("changeme", "placeholder"),
            ("redacted", "already-redacted marker"),
            ("deadbeefdeadbeefdeadbeefdeadbeef", "hex digest"),
            ("eyJhbGciOiJIUzI1NiJ9", "jwt prefix (handled elsewhere)"),
        ] {
            let input = format!("token={value}");
            let out = redact_secrets(&input);
            assert_eq!(out, input, "{label}: benign value was redacted: {out}");
        }
    }

    #[test]
    fn entropy_gate_does_not_smuggle_shell_payload() {
        // Unquoted values stop at whitespace, so a trailing shell fragment is
        // never folded into the redaction span: `hunter2;rm` is a
        // symbol-mixed short value and redacts, but ` -rf /` survives.
        let out = redact_secrets("password=hunter2;rm -rf /");
        assert_eq!(
            out,
            format!("password={REDACTED} -rf /"),
            "payload handling wrong: {out}"
        );

        let quoted = fixture(&["password=\"xT7", "mQ2", "vR9", "zK4\""]);
        let out = redact_secrets(&quoted);
        assert_eq!(
            out,
            format!("password=\"{REDACTED}\""),
            "quoted secret mishandled: {out}"
        );
    }

    #[test]
    fn entropy_gate_still_redacts_long_random_values() {
        // The pre-existing behavior: long random assignment values redact.
        let input = fixture(&["api_key=", "ABCDEFGHIJ"]);
        let out = redact_secrets(&input);
        assert!(out.contains(REDACTED), "long secret kept: {out}");
    }
}
