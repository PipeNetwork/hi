//! Thin Chrome DevTools Protocol client over `ws://` (local debugging only).

use std::net::{IpAddr, SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};
use futures_util::{Sink, SinkExt, Stream, StreamExt};
use serde_json::{Value, json};
use tokio_tungstenite::tungstenite::{Error as WsError, Message};
use url::{Host, Url};

use crate::parser::{BrowserCommand, ClickTarget};
use crate::ssrf::{
    BrowserPolicy, check_navigation_url, check_resolved_ips, check_url_with_dns,
    resolve_and_check_host_ips,
};
use crate::{BrowserExecResult, BrowserImage};

const MAX_CDP_DISCOVERY_BYTES: usize = 64 * 1024;
const MAX_INTERCEPTED_HEADERS_BYTES: usize = 128 * 1024;
const MAX_INTERCEPTED_REQUEST_BYTES: usize = 1024 * 1024;
const MAX_INTERCEPTED_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

pub async fn run(
    mode: &str,
    commands: &[BrowserCommand],
    policy: BrowserPolicy,
) -> Result<BrowserExecResult> {
    match mode {
        "live" => bail!(
            "live browser mode is disabled: an already-running target can issue requests before \
             hi installs its fail-closed request guard; use headless mode"
        ),
        "extension" => bail!(
            "extension browser mode is disabled because an existing target cannot be guarded \
             before it starts network activity; use headless mode"
        ),
        "headless" | "dedicated" => run_owned(commands, policy).await,
        other => bail!("unknown browser mode '{other}' (expected headless or dedicated)"),
    }
}

async fn run_owned(
    commands: &[BrowserCommand],
    policy: BrowserPolicy,
) -> Result<BrowserExecResult> {
    let chrome = find_chrome().context("Chrome/Chromium not found (set HI_BROWSER_BIN)")?;
    let profile = tempfile_profile()?;
    // Chrome itself never gets a route to an origin. Every browser request is
    // paused and fulfilled by the checked transport below; anything that
    // somehow bypasses Fetch interception is trapped in this owned listener.
    let proxy_sink = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
        .context("binding fail-closed browser proxy sink")?;
    let proxy_port = proxy_sink.local_addr()?.port();
    let mut child = launch_chrome(&chrome, profile.path(), proxy_port)?;
    let result = async {
        let port = wait_devtools_port(profile.path()).await?;
        let ws = websocket_from_endpoint(&format!("http://127.0.0.1:{port}"), policy).await?;
        drive(ws, commands, policy).await
    }
    .await;
    let _ = child.kill();
    let _ = child.wait();
    result
}

fn find_chrome() -> Option<PathBuf> {
    if let Ok(bin) = std::env::var("HI_BROWSER_BIN") {
        let path = PathBuf::from(bin);
        if path.is_file() {
            return Some(path);
        }
    }
    let candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/usr/local/bin/chromium",
    ];
    candidates
        .into_iter()
        .map(PathBuf::from)
        .find(|p| p.is_file())
}

fn tempfile_profile() -> Result<tempfile::TempDir> {
    let mut builder = tempfile::Builder::new();
    builder.prefix("hi-browser-");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        builder.permissions(std::fs::Permissions::from_mode(0o700));
    }
    builder.tempdir().context("creating private Chrome profile")
}

fn launch_chrome(bin: &Path, profile: &Path, proxy_port: u16) -> Result<Child> {
    let security_args = chrome_security_args(proxy_port);
    Command::new(bin)
        .args([
            "--headless=new",
            "--block-new-web-contents",
            "--disable-background-networking",
            "--disable-component-update",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-extensions",
            "--disable-sync",
            "--disable-quic",
            "--remote-debugging-port=0",
        ])
        .args(security_args)
        .arg(format!("--user-data-dir={}", profile.display()))
        .arg("about:blank")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("launching {}", bin.display()))
}

fn chrome_security_args(proxy_port: u16) -> [String; 4] {
    [
        format!("--proxy-server=http://127.0.0.1:{proxy_port}"),
        // Chromium implicitly bypasses proxies for loopback unless this token
        // is present. A missed interception must not become a localhost SSRF.
        "--proxy-bypass-list=<-loopback>".to_string(),
        // The guarded reqwest transport resolves names; Chrome must not race a
        // second resolution (or speculatively resolve an untrusted hostname).
        "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1".to_string(),
        // Fetch interception does not mediate WebRTC's UDP sockets. Force that
        // transport onto the proxy route too, where the owned sink fails closed.
        "--force-webrtc-ip-handling-policy=disable_non_proxied_udp".to_string(),
    ]
}

async fn wait_devtools_port(profile: &Path) -> Result<u16> {
    let path = profile.join("DevToolsActivePort");
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        if let Ok(text) = std::fs::read_to_string(&path)
            && let Some(line) = text.lines().next()
            && let Ok(port) = line.trim().parse::<u16>()
        {
            return Ok(port);
        }
        if tokio::time::Instant::now() > deadline {
            bail!(
                "Chrome did not publish DevToolsActivePort in {}",
                path.display()
            );
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

async fn websocket_from_endpoint(endpoint: &str, policy: BrowserPolicy) -> Result<CdpConn> {
    let endpoint = validate_loopback_cdp_url(endpoint, false)?;
    let ws_url = if matches!(endpoint.scheme(), "ws" | "wss") {
        endpoint
    } else {
        let mut discovery = endpoint;
        discovery.set_path("/json/version");
        discovery.set_query(None);
        discovery.set_fragment(None);
        let response = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(5))
            .build()?
            .get(discovery.clone())
            .send()
            .await
            .with_context(|| format!("GET {discovery}"))?
            .error_for_status()?;
        if response
            .content_length()
            .is_some_and(|size| size > MAX_CDP_DISCOVERY_BYTES as u64)
        {
            bail!("CDP /json/version response is too large");
        }
        let body = response.bytes().await?;
        if body.len() > MAX_CDP_DISCOVERY_BYTES {
            bail!("CDP /json/version response is too large");
        }
        let value: Value = serde_json::from_slice(&body).context("parsing /json/version")?;
        let ws_url = value
            .get("webSocketDebuggerUrl")
            .and_then(Value::as_str)
            .context("webSocketDebuggerUrl missing from /json/version")?;
        validate_loopback_cdp_url(ws_url, true)?
    };
    let ws_url = validate_loopback_cdp_url(ws_url.as_str(), true)?;
    CdpConn::connect(ws_url.as_str(), policy).await
}

fn validate_loopback_cdp_url(raw: &str, require_browser_socket: bool) -> Result<Url> {
    let url = Url::parse(raw).with_context(|| format!("invalid CDP endpoint '{raw}'"))?;
    if !matches!(url.scheme(), "http" | "https" | "ws" | "wss") {
        bail!("CDP endpoint must use http(s) or ws(s)");
    }
    let loopback = match url.host() {
        Some(Host::Ipv4(ip)) => ip.is_loopback(),
        Some(Host::Ipv6(ip)) => ip.is_loopback(),
        _ => false,
    };
    if !loopback {
        bail!("CDP endpoint must use a literal loopback address");
    }
    if require_browser_socket && !matches!(url.scheme(), "ws" | "wss") {
        bail!("CDP browser endpoint must use ws:// or wss://");
    }
    if require_browser_socket
        && matches!(url.scheme(), "ws" | "wss")
        && !url.path().starts_with("/devtools/browser/")
    {
        bail!("CDP endpoint must be a browser websocket, not a page target");
    }
    Ok(url)
}

/// Local Chrome DevTools is `ws://` only. Avoid naming `MaybeTlsStream` so
/// this compiles with tokio-tungstenite's `connect` feature and no TLS.
struct CdpConn {
    write: Box<dyn Sink<Message, Error = WsError> + Unpin + Send>,
    read: Box<dyn Stream<Item = Result<Message, WsError>> + Unpin + Send>,
    next_id: u64,
    policy: BrowserPolicy,
    main_session: Option<String>,
}

impl CdpConn {
    async fn connect(url: &str, policy: BrowserPolicy) -> Result<Self> {
        let (ws, _) = tokio_tungstenite::connect_async(url)
            .await
            .with_context(|| format!("connecting to {url}"))?;
        let (write, read) = ws.split();
        Ok(Self {
            write: Box::new(write),
            read: Box::new(read),
            next_id: 1,
            policy,
            main_session: None,
        })
    }

    async fn prepare_primary_page(&mut self) -> Result<String> {
        let targets = self.call_root("Target.getTargets", json!({})).await?;
        let target_id = targets
            .get("targetInfos")
            .and_then(Value::as_array)
            .and_then(|targets| {
                targets
                    .iter()
                    .find(|target| {
                        target.get("type").and_then(Value::as_str) == Some("page")
                            && target.get("url").and_then(Value::as_str) == Some("about:blank")
                    })
                    .or_else(|| {
                        targets.iter().find(|target| {
                            target.get("type").and_then(Value::as_str) == Some("page")
                        })
                    })
            })
            .and_then(|target| target.get("targetId"))
            .and_then(Value::as_str)
            .context("Chrome did not expose its initial page target")?
            .to_string();
        let attached = self
            .call_root(
                "Target.attachToTarget",
                json!({"targetId": target_id, "flatten": true}),
            )
            .await?;
        let session = attached
            .get("sessionId")
            .and_then(Value::as_str)
            .context("Target.attachToTarget did not return a sessionId")?
            .to_string();
        self.main_session = Some(session.clone());

        // These calls are mandatory. Continuing after any failure would give
        // the page an unguarded path to the network.
        self.call_session(
            &session,
            "Fetch.enable",
            json!({"patterns": [{"urlPattern": "*", "requestStage": "Request"}]}),
        )
        .await?;
        self.call_session(
            &session,
            "Network.setCacheDisabled",
            json!({"cacheDisabled": true}),
        )
        .await?;
        self.call_session(
            &session,
            "Network.setBypassServiceWorker",
            json!({"bypass": true}),
        )
        .await?;
        self.call_session(
            &session,
            "Target.setAutoAttach",
            json!({
                "autoAttach": true,
                "waitForDebuggerOnStart": true,
                "flatten": true
            }),
        )
        .await?;
        self.call_session(&session, "Page.enable", json!({}))
            .await?;
        self.call_session(&session, "Runtime.enable", json!({}))
            .await?;
        self.call_session(&session, "Accessibility.enable", json!({}))
            .await?;
        Ok(session)
    }

    async fn call_root(&mut self, method: &str, params: Value) -> Result<Value> {
        self.call(None, method, params).await
    }

    async fn call_session(&mut self, session: &str, method: &str, params: Value) -> Result<Value> {
        self.call(Some(session), method, params).await
    }

    async fn call(&mut self, session: Option<&str>, method: &str, params: Value) -> Result<Value> {
        let id = self.send_command(session, method, params).await?;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
        loop {
            let now = tokio::time::Instant::now();
            if now >= deadline {
                bail!("CDP {method} timed out");
            }
            let value = self.read_value(deadline - now).await?;
            if value.get("id").and_then(Value::as_u64) == Some(id) {
                if let Some(err) = value.get("error") {
                    bail!("CDP {method}: {err}");
                }
                return Ok(value.get("result").cloned().unwrap_or(Value::Null));
            }
            self.handle_message(&value).await?;
        }
    }

    async fn send_command(
        &mut self,
        session: Option<&str>,
        method: &str,
        params: Value,
    ) -> Result<u64> {
        let id = self.next_id;
        self.next_id += 1;
        let mut payload = json!({"id": id, "method": method, "params": params});
        if let Some(session) = session {
            payload["sessionId"] = Value::String(session.to_string());
        }
        self.write
            .send(Message::Text(payload.to_string().into()))
            .await
            .context("sending CDP command")?;
        Ok(id)
    }

    async fn read_value(&mut self, duration: Duration) -> Result<Value> {
        let deadline = tokio::time::Instant::now() + duration;
        loop {
            let now = tokio::time::Instant::now();
            if now >= deadline {
                bail!("timed out waiting for a CDP frame");
            }
            let next = tokio::time::timeout(deadline - now, self.read.next())
                .await
                .map_err(|_| anyhow::anyhow!("timed out waiting for a CDP frame"))?;
            let Some(frame) = next else {
                bail!("CDP websocket closed");
            };
            match frame.context("reading CDP frame")? {
                Message::Text(text) => {
                    return serde_json::from_str(&text).context("parsing CDP JSON");
                }
                Message::Ping(body) => {
                    self.write
                        .send(Message::Pong(body))
                        .await
                        .context("replying to CDP ping")?;
                }
                Message::Close(_) => bail!("CDP websocket closed"),
                _ => {}
            }
        }
    }

    async fn pump_for(&mut self, duration: Duration) -> Result<()> {
        let deadline = tokio::time::Instant::now() + duration;
        loop {
            let now = tokio::time::Instant::now();
            if now >= deadline {
                return Ok(());
            }
            match tokio::time::timeout(
                deadline - now,
                self.read_value(Duration::from_secs(24 * 60 * 60)),
            )
            .await
            {
                Ok(Ok(value)) => self.handle_message(&value).await?,
                Ok(Err(err)) => return Err(err),
                Err(_) => return Ok(()),
            }
        }
    }

    async fn handle_message(&mut self, value: &Value) -> Result<()> {
        if let Some(error) = value.get("error") {
            bail!("asynchronous CDP command failed: {error}");
        }
        match value.get("method").and_then(Value::as_str) {
            Some("Fetch.requestPaused") => self.handle_paused_request(value).await,
            // Child pages, workers and out-of-process frames are deliberately
            // never resumed. The primary session's Fetch guard handles normal
            // subframes; auto-attached targets stay paused and cannot race it.
            Some("Target.attachedToTarget") => Ok(()),
            _ => Ok(()),
        }
    }

    async fn handle_paused_request(&mut self, event: &Value) -> Result<()> {
        let (session, request_id, request) =
            paused_request_fields(event, self.main_session.as_deref().unwrap_or_default())?;
        let session = session.to_string();
        let request_id = request_id.to_string();
        match guarded_fetch(request, self.policy).await {
            Ok(response) => {
                self.send_command(
                    Some(&session),
                    "Fetch.fulfillRequest",
                    json!({
                        "requestId": request_id,
                        "responseCode": response.status,
                        "responseHeaders": response.headers,
                        "body": response.body_base64,
                    }),
                )
                .await?;
                Ok(())
            }
            Err(error) => {
                let _ = self
                    .send_command(
                        Some(&session),
                        "Fetch.failRequest",
                        json!({"requestId": request_id, "errorReason": "BlockedByClient"}),
                    )
                    .await;
                Err(error.context("blocked browser request before network access"))
            }
        }
    }
}

fn paused_request_fields<'a>(
    event: &'a Value,
    main_session: &str,
) -> Result<(&'a str, &'a str, &'a Value)> {
    let session = event
        .get("sessionId")
        .and_then(Value::as_str)
        .context("Fetch.requestPaused event is missing sessionId")?;
    if session != main_session {
        bail!("blocked request from a secondary browser target");
    }
    let params = event
        .get("params")
        .context("Fetch.requestPaused event is missing params")?;
    let request_id = params
        .get("requestId")
        .and_then(Value::as_str)
        .context("Fetch.requestPaused event is missing requestId")?;
    if matches!(
        params.get("resourceType").and_then(Value::as_str),
        Some("WebSocket" | "EventSource" | "WebTransport")
    ) {
        bail!("streaming browser transports are blocked by the request guard");
    }
    let request = params
        .get("request")
        .context("Fetch.requestPaused event is missing request")?;
    Ok((session, request_id, request))
}

#[derive(Debug)]
struct GuardedResponse {
    status: u16,
    headers: Vec<Value>,
    body_base64: String,
}

async fn guarded_fetch(request: &Value, policy: BrowserPolicy) -> Result<GuardedResponse> {
    let raw_url = request
        .get("url")
        .and_then(Value::as_str)
        .context("paused request is missing its URL")?;
    check_navigation_url(raw_url, policy)?;
    let url = Url::parse(raw_url).with_context(|| format!("invalid request URL '{raw_url}'"))?;
    let host = url
        .host_str()
        .context("request URL has no host")?
        .to_string();
    let resolve_host = host.clone();
    let ips = tokio::time::timeout(
        Duration::from_secs(5),
        tokio::task::spawn_blocking(move || resolve_and_check_host_ips(&resolve_host, policy)),
    )
    .await
    .context("DNS validation timed out")?
    .context("DNS validation task failed")??;
    let pinned_addrs = socket_addrs_for_url(&url, &ips, policy)?;

    let mut builder = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(20));
    if matches!(url.host(), Some(Host::Domain(_))) {
        builder = builder.resolve_to_addrs(&host, &pinned_addrs);
    }
    let client = builder.build().context("building guarded HTTP client")?;

    let method = request
        .get("method")
        .and_then(Value::as_str)
        .context("paused request is missing its method")?
        .parse::<reqwest::Method>()
        .context("invalid paused request method")?;
    let headers = guarded_request_headers(request.get("headers"))?;
    let post_data = request.get("postData").and_then(Value::as_str);
    if request
        .get("hasPostData")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        && post_data.is_none()
    {
        bail!("paused request body is not available for guarded replay");
    }
    let body = post_data.unwrap_or_default().as_bytes();
    if body.len() > MAX_INTERCEPTED_REQUEST_BYTES {
        bail!(
            "browser request body exceeds {} bytes",
            MAX_INTERCEPTED_REQUEST_BYTES
        );
    }

    let response = client
        .request(method, url)
        .headers(headers)
        .body(body.to_vec())
        .send()
        .await
        .context("guarded browser request failed")?;
    if response
        .headers()
        .get_all(reqwest::header::CONTENT_ENCODING)
        .iter()
        .any(|value| !value.as_bytes().eq_ignore_ascii_case(b"identity"))
    {
        bail!("guarded browser response used an unsupported content encoding");
    }
    if response
        .content_length()
        .is_some_and(|length| length > MAX_INTERCEPTED_RESPONSE_BYTES as u64)
    {
        bail!(
            "browser response exceeds {} bytes",
            MAX_INTERCEPTED_RESPONSE_BYTES
        );
    }
    let status = response.status().as_u16();
    let response_headers = guarded_response_headers(response.headers())?;
    let mut response_body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("reading guarded browser response")?;
        if response_body.len().saturating_add(chunk.len()) > MAX_INTERCEPTED_RESPONSE_BYTES {
            bail!(
                "browser response exceeds {} bytes",
                MAX_INTERCEPTED_RESPONSE_BYTES
            );
        }
        response_body.extend_from_slice(&chunk);
    }
    Ok(GuardedResponse {
        status,
        headers: response_headers,
        body_base64: BASE64_STANDARD.encode(response_body),
    })
}

fn socket_addrs_for_url(
    url: &Url,
    ips: &[IpAddr],
    policy: BrowserPolicy,
) -> Result<Vec<SocketAddr>> {
    let host = url.host_str().context("request URL has no host")?;
    check_resolved_ips(host, ips, policy)?;
    let port = url
        .port_or_known_default()
        .context("request URL has no usable port")?;
    let mut addrs: Vec<_> = ips.iter().map(|ip| SocketAddr::new(*ip, port)).collect();
    addrs.sort_unstable();
    addrs.dedup();
    Ok(addrs)
}

fn guarded_request_headers(value: Option<&Value>) -> Result<reqwest::header::HeaderMap> {
    let mut output = reqwest::header::HeaderMap::new();
    let mut total = 0usize;
    let Some(headers) = value.and_then(Value::as_object) else {
        return Ok(output);
    };
    for (name, value) in headers {
        let Some(value) = value.as_str() else {
            bail!("paused request contains a non-string header");
        };
        total = total.saturating_add(name.len()).saturating_add(value.len());
        if total > MAX_INTERCEPTED_HEADERS_BYTES {
            bail!("browser request headers are too large");
        }
        if is_hop_by_hop_header(name) || name.eq_ignore_ascii_case("host") {
            continue;
        }
        let name = reqwest::header::HeaderName::from_bytes(name.as_bytes())
            .context("invalid paused request header name")?;
        let value = reqwest::header::HeaderValue::from_bytes(value.as_bytes())
            .context("invalid paused request header value")?;
        output.append(name, value);
    }
    output.insert(
        reqwest::header::ACCEPT_ENCODING,
        reqwest::header::HeaderValue::from_static("identity"),
    );
    Ok(output)
}

fn guarded_response_headers(headers: &reqwest::header::HeaderMap) -> Result<Vec<Value>> {
    let mut output = Vec::new();
    let mut total = 0usize;
    for (name, value) in headers {
        if is_hop_by_hop_header(name.as_str())
            || name == reqwest::header::CONTENT_LENGTH
            || name == reqwest::header::CONTENT_ENCODING
        {
            continue;
        }
        let value = value
            .to_str()
            .context("guarded response contains a non-text header")?;
        total = total
            .saturating_add(name.as_str().len())
            .saturating_add(value.len());
        if total > MAX_INTERCEPTED_HEADERS_BYTES {
            bail!("browser response headers are too large");
        }
        output.push(json!({"name": name.as_str(), "value": value}));
    }
    Ok(output)
}

fn is_hop_by_hop_header(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "proxy-connection"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
    )
}

async fn drive(
    mut cdp: CdpConn,
    commands: &[BrowserCommand],
    policy: BrowserPolicy,
) -> Result<BrowserExecResult> {
    let session = cdp.prepare_primary_page().await?;
    let mut notes = Vec::new();
    let mut images = Vec::new();
    let mut ax_nodes: Vec<AxNode> = Vec::new();
    for command in commands {
        match command {
            BrowserCommand::Goto { url } => {
                check_url_with_dns(url, policy)?;
                cdp.call_session(&session, "Page.navigate", json!({"url": url}))
                    .await?;
                cdp.pump_for(Duration::from_millis(300)).await?;
                if let Some(committed) = committed_page_url(&mut cdp, &session).await {
                    check_url_with_dns(&committed, policy).with_context(|| {
                        format!("committed URL after navigate is not allowed: {committed}")
                    })?;
                    notes.push(format!("navigated to {committed}"));
                } else {
                    notes.push(format!("navigated to {url}"));
                }
            }
            BrowserCommand::Click { target } => {
                let (x, y) = resolve_point(&ax_nodes, target)?;
                dispatch_click(&mut cdp, &session, x, y).await?;
                cdp.pump_for(Duration::from_millis(100)).await?;
                notes.push(format!("clicked {x:.0},{y:.0}"));
            }
            BrowserCommand::Type { text, target } => {
                if let Some(target) = target {
                    let (x, y) = resolve_point(&ax_nodes, target)?;
                    dispatch_click(&mut cdp, &session, x, y).await?;
                }
                cdp.call_session(&session, "Input.insertText", json!({"text": text}))
                    .await?;
                cdp.pump_for(Duration::from_millis(100)).await?;
                notes.push(format!("typed {} chars", text.chars().count()));
            }
            BrowserCommand::Screenshot => {
                let result = cdp
                    .call_session(&session, "Page.captureScreenshot", json!({"format": "png"}))
                    .await?;
                let data = result
                    .get("data")
                    .and_then(|v| v.as_str())
                    .context("screenshot missing data")?
                    .to_string();
                images.push(BrowserImage {
                    data,
                    media_type: "image/png".into(),
                });
                notes.push("captured screenshot".into());
            }
            BrowserCommand::Ax => {
                let tree = cdp
                    .call_session(&session, "Accessibility.getFullAXTree", json!({}))
                    .await?;
                ax_nodes = flatten_ax(&tree);
                let mut body = String::from("AX tree:\n");
                for (i, node) in ax_nodes.iter().enumerate() {
                    body.push_str(&format!(
                        "  [{i}] {} {}\n",
                        node.role,
                        node.name.as_deref().unwrap_or("")
                    ));
                }
                notes.push(body);
            }
            BrowserCommand::Wait { millis } => {
                cdp.pump_for(Duration::from_millis(*millis)).await?;
                notes.push(format!("waited {millis}ms"));
            }
            BrowserCommand::Eval { expression } => {
                let result = cdp
                    .call_session(
                        &session,
                        "Runtime.evaluate",
                        json!({"expression": expression, "returnByValue": true}),
                    )
                    .await?;
                cdp.pump_for(Duration::from_millis(100)).await?;
                notes.push(format!("eval: {result}"));
            }
            BrowserCommand::Scroll { dx, dy } => {
                cdp.call_session(
                    &session,
                    "Input.dispatchMouseEvent",
                    json!({
                        "type": "mouseWheel",
                        "x": 0,
                        "y": 0,
                        "deltaX": dx,
                        "deltaY": dy
                    }),
                )
                .await?;
                cdp.pump_for(Duration::from_millis(100)).await?;
                notes.push(format!("scrolled {dx},{dy}"));
            }
        }
    }
    // Drain requests triggered by the final command. The owned Chrome process
    // is killed immediately after this returns, while Fetch remains enabled.
    cdp.pump_for(Duration::from_millis(100)).await?;
    Ok(BrowserExecResult {
        text: notes.join("\n"),
        images,
    })
}

async fn committed_page_url(cdp: &mut CdpConn, session: &str) -> Option<String> {
    if let Ok(tree) = cdp
        .call_session(session, "Page.getFrameTree", json!({}))
        .await
        && let Some(url) = tree
            .pointer("/frameTree/frame/url")
            .and_then(|v| v.as_str())
        && looks_like_http_url(url)
    {
        return Some(url.to_string());
    }
    if let Ok(history) = cdp
        .call_session(session, "Page.getNavigationHistory", json!({}))
        .await
    {
        let index = history
            .get("currentIndex")
            .and_then(|i| i.as_i64())
            .unwrap_or(0);
        if let Some(url) = history
            .get("entries")
            .and_then(|e| e.as_array())
            .and_then(|entries| entries.get(index as usize))
            .and_then(|entry| entry.get("url"))
            .and_then(|u| u.as_str())
            && looks_like_http_url(url)
        {
            return Some(url.to_string());
        }
    }
    None
}

fn looks_like_http_url(url: &str) -> bool {
    let lower = url.to_ascii_lowercase();
    (lower.starts_with("http://") || lower.starts_with("https://"))
        && !lower.contains("about:blank")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[test]
    fn cdp_endpoint_must_be_a_literal_loopback_browser_socket() {
        assert!(validate_loopback_cdp_url("http://127.0.0.1:9222", false).is_ok());
        assert!(validate_loopback_cdp_url("http://localhost:9222", false).is_err());
        assert!(validate_loopback_cdp_url("http://example.com:9222", false).is_err());
        assert!(validate_loopback_cdp_url("http://127.0.0.1:9222", true).is_err());
        assert!(validate_loopback_cdp_url("ws://127.0.0.1/devtools/page/one", true).is_err());
        assert!(validate_loopback_cdp_url("ws://[::1]:9222/devtools/browser/one", true).is_ok());
    }

    #[test]
    fn chrome_has_no_direct_origin_route() {
        let args = chrome_security_args(43123);
        assert!(
            args.iter()
                .any(|arg| arg == "--proxy-server=http://127.0.0.1:43123")
        );
        assert!(
            args.iter()
                .any(|arg| arg == "--proxy-bypass-list=<-loopback>")
        );
        assert!(
            args.iter()
                .any(|arg| arg.starts_with("--host-resolver-rules="))
        );
        assert!(
            args.iter()
                .any(|arg| arg == "--force-webrtc-ip-handling-policy=disable_non_proxied_udp")
        );
    }

    #[test]
    #[cfg(unix)]
    fn chrome_profile_is_private_and_race_free() {
        let profile = tempfile_profile().unwrap();
        let metadata = std::fs::symlink_metadata(profile.path()).unwrap();
        assert!(metadata.is_dir());
        assert!(!metadata.file_type().is_symlink());
        assert_eq!(metadata.permissions().mode() & 0o077, 0);
    }

    #[test]
    fn dns_rebinding_transport_uses_only_validated_addresses() {
        let url = Url::parse("https://rebind.example/path").unwrap();
        let checked_ip: IpAddr = "93.184.216.34".parse().unwrap();
        let addrs = socket_addrs_for_url(&url, &[checked_ip], BrowserPolicy::default()).unwrap();
        assert_eq!(addrs, vec![SocketAddr::new(checked_ip, 443)]);

        let rebound_private: IpAddr = "127.0.0.1".parse().unwrap();
        assert!(socket_addrs_for_url(&url, &[rebound_private], BrowserPolicy::default()).is_err());
        assert_eq!(
            socket_addrs_for_url(
                &url,
                &[rebound_private],
                BrowserPolicy {
                    allow_private: true
                }
            )
            .unwrap(),
            vec![SocketAddr::new(rebound_private, 443)]
        );

        let metadata: IpAddr = "169.254.169.254".parse().unwrap();
        assert!(
            socket_addrs_for_url(
                &url,
                &[metadata],
                BrowserPolicy {
                    allow_private: true
                }
            )
            .is_err()
        );
    }

    #[test]
    fn request_guard_rejects_secondary_and_streaming_requests() {
        let base = json!({
            "sessionId": "main",
            "method": "Fetch.requestPaused",
            "params": {
                "requestId": "request-1",
                "resourceType": "Document",
                "request": {"url": "https://example.com/", "method": "GET", "headers": {}}
            }
        });
        assert!(paused_request_fields(&base, "main").is_ok());

        let mut secondary = base.clone();
        secondary["sessionId"] = json!("popup");
        assert!(paused_request_fields(&secondary, "main").is_err());

        for resource_type in ["WebSocket", "EventSource", "WebTransport"] {
            let mut streaming = base.clone();
            streaming["params"]["resourceType"] = json!(resource_type);
            assert!(paused_request_fields(&streaming, "main").is_err());
        }
    }

    #[tokio::test]
    async fn private_subresource_is_rejected_before_transport() {
        let request = json!({
            "url": "http://127.0.0.1:9/private-subresource",
            "method": "GET",
            "headers": {}
        });
        let error = guarded_fetch(&request, BrowserPolicy::default())
            .await
            .expect_err("private request must fail");
        assert!(error.to_string().contains("private/loopback"), "{error}");
    }

    #[tokio::test]
    #[ignore = "requires loopback bind"]
    async fn private_override_works_but_redirects_are_rechecked() {
        let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
            .await
            .unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0u8; 2048];
            let _ = stream.read(&mut request).await.unwrap();
            stream
                .write_all(
                    b"HTTP/1.1 302 Found\r\nLocation: http://169.254.169.254/latest\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .await
                .unwrap();
        });
        let first_request = json!({
            "url": format!("http://127.0.0.1:{port}/redirect"),
            "method": "GET",
            "headers": {}
        });
        let open = BrowserPolicy {
            allow_private: true,
        };
        let response = guarded_fetch(&first_request, open).await.unwrap();
        assert_eq!(
            response.status, 302,
            "the guarded client must not follow redirects"
        );
        server.await.unwrap();

        let redirected_request = json!({
            "url": "http://169.254.169.254/latest",
            "method": "GET",
            "headers": {}
        });
        assert!(guarded_fetch(&redirected_request, open).await.is_err());
    }

    #[tokio::test]
    async fn unsafe_attach_modes_fail_without_connecting() {
        let commands = [BrowserCommand::Screenshot];
        let policy = BrowserPolicy::default();
        assert!(run("live", &commands, policy).await.is_err());
        assert!(run("extension", &commands, policy).await.is_err());
        assert!(run("surprise", &commands, policy).await.is_err());
    }
}

#[cfg(test)]
mod e2e {
    #[tokio::test]
    #[ignore = "requires Chrome/Chromium"]
    async fn headless_goto_example() {
        crate::configure(crate::BrowserConfig {
            enabled: true,
            allow_private: false,
        });
        let result = crate::run_exec(
            r#"{"script":"goto https://example.com\nscreenshot","mode":"headless"}"#,
        )
        .await
        .expect("headless browser_exec");
        assert!(
            result.text.contains("navigated") || !result.images.is_empty(),
            "{}",
            result.text
        );
    }

    #[tokio::test]
    #[ignore = "requires Chrome/Chromium and public network access"]
    async fn eval_navigation_to_metadata_is_intercepted_before_request() {
        crate::configure(crate::BrowserConfig {
            enabled: true,
            allow_private: false,
        });
        let error = crate::run_exec(
            r#"{"script":"goto https://example.com\neval location.href='http://169.254.169.254/latest'\nwait 100","mode":"headless"}"#,
        )
        .await
        .expect_err("script-triggered metadata navigation");
        let message = format!("{error:#}");
        assert!(
            message.contains("metadata") || message.contains("link-local"),
            "{message}"
        );
    }
}

#[derive(Clone, Debug)]
struct AxNode {
    role: String,
    name: Option<String>,
    x: f64,
    y: f64,
}

fn flatten_ax(tree: &Value) -> Vec<AxNode> {
    let mut out = Vec::new();
    let nodes = tree
        .get("nodes")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    for node in nodes {
        let role = node
            .get("role")
            .and_then(|v| v.get("value"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let name = node
            .get("name")
            .and_then(|v| v.get("value"))
            .and_then(|v| v.as_str())
            .map(str::to_string);
        if role == "none" || role == "InlineTextBox" {
            continue;
        }
        out.push(AxNode {
            role,
            name,
            x: 8.0,
            y: 8.0 + (out.len() as f64) * 12.0,
        });
        if out.len() >= 80 {
            break;
        }
    }
    out
}

fn resolve_point(ax: &[AxNode], target: &ClickTarget) -> Result<(f64, f64)> {
    match target {
        ClickTarget::Point { x, y } => Ok((*x, *y)),
        ClickTarget::Index(i) => {
            let node = ax
                .get(*i as usize)
                .with_context(|| format!("no AX node {i}; call `ax` first"))?;
            Ok((node.x, node.y))
        }
    }
}

async fn dispatch_click(cdp: &mut CdpConn, session: &str, x: f64, y: f64) -> Result<()> {
    cdp.call_session(
        session,
        "Input.dispatchMouseEvent",
        json!({"type": "mousePressed", "x": x, "y": y, "button": "left", "clickCount": 1}),
    )
    .await?;
    cdp.call_session(
        session,
        "Input.dispatchMouseEvent",
        json!({"type": "mouseReleased", "x": x, "y": y, "button": "left", "clickCount": 1}),
    )
    .await?;
    Ok(())
}
