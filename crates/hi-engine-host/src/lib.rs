//! Native substrate for loading and running the optional WASM decision engine.
//!
//! This crate intentionally contains no provider or tool implementation. It
//! only owns module lifecycle, ABI validation, resource limits, generation
//! pinning, and the narrow component call that turns host input into host-
//! validated actions.

use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    thread::JoinHandle,
    time::Duration,
};

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use hi_engine_api::{
    ENGINE_API_MAJOR, ENGINE_STATE_SCHEMA_VERSION, EngineAction, EngineInput, EngineManifest,
    ProtocolError, decode_actions, encode_input,
};
use sha2::{Digest, Sha256};
use wasmtime::{
    Config, Engine, Store, StoreLimits, StoreLimitsBuilder,
    component::{Component, Linker, TypedFunc},
};

mod native_director;

pub use native_director::*;

pub const DEFAULT_MAX_MODULE_BYTES: usize = 16 * 1024 * 1024;
pub const DEFAULT_GUEST_FUEL: u64 = 10_000_000;
pub const DEFAULT_GUEST_MEMORY_BYTES: usize = 64 * 1024 * 1024;
pub const DEFAULT_GUEST_STEP_TIMEOUT_MS: u64 = 2_000;

// Wasmtime's macOS trap handler owns a process-global Mach port. Constructing
// and dropping multiple engines concurrently can race that port's teardown;
// the handler thread then aborts the process when it receives a malformed
// message. Keep one configured engine alive for the process and clone its
// cheap handle for each runtime. This also avoids paying engine setup cost for
// every Agent instance when the WASM module is not enabled.
static SHARED_WASMTIME_ENGINE: std::sync::OnceLock<Result<Engine, String>> =
    std::sync::OnceLock::new();

fn shared_wasmtime_engine() -> Result<Engine> {
    SHARED_WASMTIME_ENGINE
        .get_or_init(|| {
            let mut config = Config::new();
            config.consume_fuel(true);
            config.epoch_interruption(true);
            config.wasm_component_model(true);
            // Wasmtime's default macOS Mach-port trap handler is process-global
            // and can abort during concurrent fork/test teardown. Use the
            // signal-based handler instead; the shared engine above keeps its
            // handler lifetime stable for the entire process.
            #[cfg(target_vendor = "apple")]
            config.macos_use_mach_ports(false);
            Engine::new(&config).map_err(|error| format!("creating Wasmtime engine: {error:#}"))
        })
        .clone()
        .map_err(|error| anyhow!(error.clone()))
}

pub fn parse_trusted_keys(values: &[String]) -> Result<Vec<VerifyingKey>> {
    values
        .iter()
        .map(|value| {
            let raw = hex::decode(value.trim())
                .with_context(|| "HI_ENGINE_TRUSTED_KEY must be 64 hexadecimal bytes")?;
            let bytes: [u8; 32] = raw
                .try_into()
                .map_err(|_| anyhow!("HI_ENGINE_TRUSTED_KEY must be 64 hexadecimal bytes"))?;
            VerifyingKey::from_bytes(&bytes).context("invalid trusted engine public key")
        })
        .collect()
}

/// The only native effect boundary exposed to an engine integrator. The WASM
/// component itself has no imports; the host translates validated actions into
/// this broker after applying the current policy, confirmation, and revision
/// checks.
#[async_trait]
pub trait EffectBroker: Send + Sync {
    async fn execute_tool(&self, request: hi_engine_api::ToolRequest) -> Result<EffectResult>;

    async fn execute_parallel(
        &self,
        requests: Vec<hi_engine_api::ToolRequest>,
    ) -> Result<Vec<EffectResult>> {
        let mut results = Vec::with_capacity(requests.len());
        for request in requests {
            results.push(self.execute_tool(request).await?);
        }
        Ok(results)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EffectResult {
    pub idempotency_key: String,
    pub status: String,
    pub output: String,
}

/// Decision engines are deliberately synchronous at the protocol boundary:
/// provider/tool I/O belongs to the substrate and arrives as later
/// [`EngineInput`] events. This keeps a guest step bounded and replayable.
pub trait DecisionEngine: Send {
    fn mode(&self) -> hi_engine_api::EngineMode;
    fn step(&mut self, input: &EngineInput) -> Result<Vec<EngineAction>>;
    fn serialize_state(&mut self) -> Result<Vec<u8>>;
}

/// Adapter used while the existing Rust turn loop is being extracted. It lets
/// replay tests and the eventual native orchestrator use the same protocol
/// without moving the current side-effecting implementation into this crate.
type NativeStepper = Box<dyn FnMut(&EngineInput) -> Result<Vec<EngineAction>> + Send>;

pub struct NativeDecisionEngine {
    stepper: NativeStepper,
    state: Vec<u8>,
}

impl NativeDecisionEngine {
    pub fn new(
        stepper: impl FnMut(&EngineInput) -> Result<Vec<EngineAction>> + Send + 'static,
    ) -> Self {
        Self {
            stepper: Box::new(stepper),
            state: Vec::new(),
        }
    }
}

impl DecisionEngine for NativeDecisionEngine {
    fn mode(&self) -> hi_engine_api::EngineMode {
        hi_engine_api::EngineMode::Native
    }

    fn step(&mut self, input: &EngineInput) -> Result<Vec<EngineAction>> {
        let actions = (self.stepper)(input)?;
        for action in &actions {
            action
                .validate()
                .map_err(|error| anyhow!("native engine returned invalid action: {error}"))?;
        }
        Ok(actions)
    }

    fn serialize_state(&mut self) -> Result<Vec<u8>> {
        Ok(self.state.clone())
    }
}

#[derive(Clone, Debug)]
pub struct ModuleValidationPolicy {
    pub allow_unsigned: bool,
    pub required_api_major: u16,
    pub required_state_schema_version: u32,
    pub max_module_bytes: usize,
    pub trusted_keys: Vec<VerifyingKey>,
    pub max_guest_fuel: u64,
    pub max_guest_memory_bytes: usize,
    pub max_guest_step_ms: u64,
}

impl Default for ModuleValidationPolicy {
    fn default() -> Self {
        Self {
            allow_unsigned: false,
            required_api_major: ENGINE_API_MAJOR,
            required_state_schema_version: ENGINE_STATE_SCHEMA_VERSION,
            max_module_bytes: DEFAULT_MAX_MODULE_BYTES,
            trusted_keys: Vec::new(),
            max_guest_fuel: DEFAULT_GUEST_FUEL,
            max_guest_memory_bytes: DEFAULT_GUEST_MEMORY_BYTES,
            max_guest_step_ms: DEFAULT_GUEST_STEP_TIMEOUT_MS,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ModuleArtifact {
    pub wasm_path: PathBuf,
    pub manifest_path: PathBuf,
    pub bytes: Arc<[u8]>,
    pub manifest: EngineManifest,
    pub sha256: String,
}

impl ModuleArtifact {
    pub fn load(path: impl AsRef<Path>, policy: &ModuleValidationPolicy) -> Result<Self> {
        let wasm_path = path.as_ref().to_path_buf();
        let bytes = std::fs::read(&wasm_path)
            .with_context(|| format!("reading engine module {}", wasm_path.display()))?;
        Self::from_bytes(wasm_path, bytes, policy)
    }

    pub fn from_bytes(
        wasm_path: impl Into<PathBuf>,
        bytes: Vec<u8>,
        policy: &ModuleValidationPolicy,
    ) -> Result<Self> {
        if bytes.is_empty() {
            bail!("engine module is empty")
        }
        if bytes.len() > policy.max_module_bytes {
            bail!(
                "engine module exceeds {}-byte limit",
                policy.max_module_bytes
            )
        }
        let wasm_path = wasm_path.into();
        let manifest_path = manifest_path_for(&wasm_path);
        let manifest_bytes = std::fs::read(&manifest_path).with_context(|| {
            format!(
                "reading engine manifest {}; build a signed .manifest.json beside the module",
                manifest_path.display()
            )
        })?;
        let manifest: EngineManifest = serde_json::from_slice(&manifest_bytes)
            .with_context(|| format!("parsing engine manifest {}", manifest_path.display()))?;
        validate_artifact(&bytes, &manifest, policy)?;
        let sha256 = hex::encode(Sha256::digest(&bytes));
        Ok(Self {
            wasm_path,
            manifest_path,
            bytes: Arc::from(bytes),
            manifest,
            sha256,
        })
    }
}

pub fn manifest_path_for(wasm_path: &Path) -> PathBuf {
    let mut path = wasm_path.to_path_buf();
    path.set_extension("manifest.json");
    path
}

/// Resolve an explicitly configured module first, then the release artifact
/// beside the running binary. PATH lookup is intentionally omitted: a logic
/// module is executable policy and must come from a known location.
pub fn discover_module_path(explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(path) = explicit.filter(|path| path.is_file()) {
        return Some(path.to_path_buf());
    }
    if let Some(path) = std::env::var_os("HI_ENGINE_MODULE").map(PathBuf::from)
        && path.is_file()
    {
        return Some(path);
    }
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|parent| parent.join("engine.wasm")))
        .filter(|path| path.is_file())
}

pub fn validate_artifact(
    bytes: &[u8],
    manifest: &EngineManifest,
    policy: &ModuleValidationPolicy,
) -> Result<()> {
    manifest
        .validate()
        .map_err(|error| anyhow!("invalid engine manifest: {error}"))?;
    if manifest.api_major != policy.required_api_major {
        bail!(
            "engine API major {} is unsupported; expected {}",
            manifest.api_major,
            policy.required_api_major
        )
    }
    if manifest.state_schema_version != policy.required_state_schema_version {
        bail!(
            "engine state schema {} is unsupported; expected {}",
            manifest.state_schema_version,
            policy.required_state_schema_version
        )
    }
    if !manifest.required_capabilities.is_empty() {
        bail!(
            "engine manifest requests unsupported host capabilities: {}",
            manifest.required_capabilities.join(", ")
        )
    }
    let digest = hex::encode(Sha256::digest(bytes));
    if !digest.eq_ignore_ascii_case(&manifest.module_sha256) {
        bail!("engine module SHA-256 does not match its manifest")
    }
    match &manifest.signature_hex {
        None if policy.allow_unsigned => Ok(()),
        None => bail!("engine module is unsigned; enable explicit local development loading"),
        Some(_) if policy.trusted_keys.is_empty() => {
            bail!("engine module is signed but no trusted engine key is configured")
        }
        Some(signature_hex) => {
            let raw = hex::decode(signature_hex).context("invalid engine signature encoding")?;
            let signature = Signature::from_slice(&raw).context("invalid engine signature")?;
            let payload = manifest
                .signing_bytes()
                .context("serializing engine manifest")?;
            if policy
                .trusted_keys
                .iter()
                .any(|key| key.verify(&payload, &signature).is_ok())
            {
                Ok(())
            } else {
                bail!("engine manifest signature did not match a trusted key")
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleInfo {
    pub generation: u64,
    pub guest_version: String,
    pub api_major: u16,
    pub api_minor: u16,
    pub module_sha256: String,
}

struct LoadedModule {
    info: ModuleInfo,
    component: Arc<Component>,
}

struct RuntimeState {
    active: Option<Arc<LoadedModule>>,
    pending: Option<Arc<LoadedModule>>,
    previous: Option<Arc<LoadedModule>>,
    active_turns: usize,
}

/// Module lifecycle manager. A lease pins one generation for the lifetime of
/// a turn. Reload never mutates an existing lease.
pub struct EngineRuntime {
    engine: Engine,
    policy: ModuleValidationPolicy,
    state: Mutex<RuntimeState>,
    next_generation: AtomicU64,
    watch: Mutex<Option<WatchState>>,
}

struct WatchState {
    stop: Arc<std::sync::atomic::AtomicBool>,
    thread: JoinHandle<()>,
}

impl EngineRuntime {
    pub fn new(policy: ModuleValidationPolicy) -> Result<Arc<Self>> {
        let engine = shared_wasmtime_engine()?;
        Ok(Arc::new(Self {
            engine,
            policy,
            state: Mutex::new(RuntimeState {
                active: None,
                pending: None,
                previous: None,
                active_turns: 0,
            }),
            next_generation: AtomicU64::new(1),
            watch: Mutex::new(None),
        }))
    }

    pub fn disabled() -> Result<Arc<Self>> {
        Self::new(ModuleValidationPolicy::default())
    }

    pub fn reload(&self, path: impl AsRef<Path>) -> Result<ModuleInfo> {
        let artifact = ModuleArtifact::load(path, &self.policy)?;
        let component = Component::from_binary(&self.engine, &artifact.bytes)
            .context("compiling engine component")?;
        validate_component_surface(&self.engine, &component)?;
        let generation = self.next_generation.fetch_add(1, Ordering::Relaxed);
        let loaded = Arc::new(LoadedModule {
            info: ModuleInfo {
                generation,
                guest_version: artifact.manifest.guest_version.clone(),
                api_major: artifact.manifest.api_major,
                api_minor: artifact.manifest.api_minor,
                module_sha256: artifact.sha256.clone(),
            },
            component: Arc::new(component),
        });
        let info = loaded.info.clone();
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("engine state poisoned"))?;
        // Never replace a running generation. The pending slot is intentionally
        // last-write-wins so a development watcher cannot queue an unbounded
        // sequence of stale builds.
        if state.active_turns == 0 {
            state.previous = state.active.take();
            state.active = Some(loaded);
            state.pending = None;
        } else {
            state.pending = Some(loaded);
        }
        Ok(info)
    }

    pub fn begin_turn(self: &Arc<Self>) -> Result<EngineLease> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("engine state poisoned"))?;
        state.active_turns = state.active_turns.saturating_add(1);
        Ok(EngineLease {
            runtime: Arc::clone(self),
            module: state.active.clone(),
        })
    }

    pub fn current(&self) -> Result<Option<ModuleInfo>> {
        let state = self
            .state
            .lock()
            .map_err(|_| anyhow!("engine state poisoned"))?;
        Ok(state.active.as_ref().map(|module| module.info.clone()))
    }

    pub fn pending(&self) -> Result<Option<ModuleInfo>> {
        let state = self
            .state
            .lock()
            .map_err(|_| anyhow!("engine state poisoned"))?;
        Ok(state.pending.as_ref().map(|module| module.info.clone()))
    }

    /// Watch a prebuilt module and its manifest for atomic replacement. The
    /// watcher is deliberately opt-in and debounced by requiring both files'
    /// metadata to remain unchanged for one polling interval.
    pub fn start_watch(self: &Arc<Self>, path: impl Into<PathBuf>) -> Result<()> {
        let path = path.into();
        let mut watch = self
            .watch
            .lock()
            .map_err(|_| anyhow!("engine watcher poisoned"))?;
        if watch.is_some() {
            return Ok(());
        }
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_for_thread = Arc::clone(&stop);
        let runtime = Arc::downgrade(self);
        let thread = std::thread::Builder::new()
            .name("hi-engine-watch".into())
            .spawn(move || {
                let mut last = file_signature(&path);
                while !stop_for_thread.load(Ordering::Acquire) {
                    std::thread::sleep(Duration::from_secs(1));
                    let current = file_signature(&path);
                    if current.is_some() && current != last {
                        // A build may replace the wasm and manifest in either
                        // order. Wait for the next stable signature before
                        // attempting a load; a failed candidate never replaces
                        // the active generation.
                        std::thread::sleep(Duration::from_secs(1));
                        if file_signature(&path) == current {
                            if let Some(runtime) = runtime.upgrade()
                                && let Err(error) = runtime.reload(&path)
                            {
                                tracing::warn!(path = %path.display(), %error, "engine watch candidate rejected");
                            }
                            last = current;
                        }
                    }
                }
            })
            .context("starting engine module watcher")?;
        *watch = Some(WatchState { stop, thread });
        Ok(())
    }

    pub fn stop_watch(&self) {
        if let Ok(mut watch) = self.watch.lock()
            && let Some(state) = watch.take()
        {
            state.stop.store(true, Ordering::Release);
            // The thread only polls once per second. It is intentionally not
            // joined here so stopping the watcher cannot stall a turn or exit.
            drop(state.thread);
        }
    }

    pub fn is_watching(&self) -> bool {
        self.watch.lock().ok().is_some_and(|watch| watch.is_some())
    }

    /// Remove a trapping or otherwise rejected active module from selection.
    /// A known-good previous generation becomes pending while a turn is still
    /// pinned, so no in-flight guest can be changed underneath its lease.
    pub fn rollback_active(&self) -> bool {
        let Ok(mut state) = self.state.lock() else {
            return false;
        };
        let replacement = state.previous.take();
        state.pending = replacement;
        if state.active_turns == 0 {
            state.active = state.pending.take();
        } else {
            state.active = None;
        }
        true
    }

    fn finish_turn(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.active_turns = state.active_turns.saturating_sub(1);
            if state.active_turns == 0
                && let Some(pending) = state.pending.take()
            {
                state.previous = state.active.take();
                state.active = Some(pending);
            }
        }
    }
}

impl Drop for EngineRuntime {
    fn drop(&mut self) {
        if let Ok(watch) = self.watch.get_mut()
            && let Some(state) = watch.take()
        {
            state.stop.store(true, Ordering::Release);
            drop(state.thread);
        }
    }
}

fn file_signature(path: &Path) -> Option<(u128, u64, u128, u64)> {
    let manifest = manifest_path_for(path);
    let wasm = std::fs::metadata(path).ok()?;
    let manifest = std::fs::metadata(manifest).ok()?;
    Some((
        wasm.modified()
            .ok()?
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?
            .as_nanos(),
        wasm.len(),
        manifest
            .modified()
            .ok()?
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?
            .as_nanos(),
        manifest.len(),
    ))
}

fn validate_component_surface(engine: &Engine, component: &Component) -> Result<()> {
    let component_type = component.component_type();
    let imports: Vec<_> = component_type
        .imports(engine)
        .map(|(name, _)| name.to_string())
        .collect();
    if !imports.is_empty() {
        bail!(
            "engine component requests unsupported host imports: {}",
            imports.join(", ")
        )
    }
    let exports: Vec<_> = component_type
        .exports(engine)
        .map(|(name, _)| name.to_string())
        .collect();
    for required in ["step", "serialize-state"] {
        if !exports.iter().any(|name| name == required) {
            bail!("engine component is missing required export `{required}`")
        }
    }
    Ok(())
}

pub struct EngineLease {
    runtime: Arc<EngineRuntime>,
    module: Option<Arc<LoadedModule>>,
}

impl EngineLease {
    pub fn generation(&self) -> Option<u64> {
        self.module.as_ref().map(|module| module.info.generation)
    }

    pub fn info(&self) -> Option<ModuleInfo> {
        self.module.as_ref().map(|module| module.info.clone())
    }

    pub fn wasm_engine(&self) -> Result<Option<WasmDecisionEngine>> {
        self.module
            .as_ref()
            .map(|module| WasmDecisionEngine::new(Arc::clone(&self.runtime), Arc::clone(module)))
            .transpose()
    }
}

impl Drop for EngineLease {
    fn drop(&mut self) {
        self.runtime.finish_turn();
    }
}

/// A single turn's WASM instance. It owns no persistent guest state and is
/// dropped with its lease, so a replacement cannot alter an active turn.
pub struct WasmDecisionEngine {
    store: Store<GuestStoreState>,
    step: TypedFunc<(String,), (String,)>,
    serialize_state: TypedFunc<(), (Vec<u8>,)>,
    _runtime: Arc<EngineRuntime>,
    info: ModuleInfo,
}

struct GuestStoreState {
    limits: StoreLimits,
}

impl WasmDecisionEngine {
    fn new(runtime: Arc<EngineRuntime>, module: Arc<LoadedModule>) -> Result<Self> {
        let linker = Linker::new(&runtime.engine);
        let mut store = Store::new(
            &runtime.engine,
            GuestStoreState {
                limits: StoreLimitsBuilder::new()
                    .memory_size(runtime.policy.max_guest_memory_bytes)
                    .instances(1)
                    .tables(1)
                    .memories(1)
                    .build(),
            },
        );
        store.limiter(|state| &mut state.limits);
        store
            .set_fuel(runtime.policy.max_guest_fuel)
            .context("setting engine fuel budget")?;
        // The WIT world intentionally has no imports. Instantiation fails
        // closed if a candidate component requests any host capability.
        let instance = linker
            .instantiate(&mut store, &module.component)
            .context("instantiating engine component")?;
        let step = instance
            .get_typed_func::<(String,), (String,)>(&mut store, "step")
            .context("engine component does not export step(string)->string")?;
        let serialize_state = instance
            .get_typed_func::<(), (Vec<u8>,)>(&mut store, "serialize-state")
            .context("engine component does not export serialize-state()->list<u8>")?;
        Ok(Self {
            store,
            step,
            serialize_state,
            _runtime: runtime,
            info: module.info.clone(),
        })
    }

    pub fn info(&self) -> &ModuleInfo {
        &self.info
    }

    pub fn step(&mut self, input: &EngineInput) -> Result<Vec<EngineAction>> {
        let encoded =
            encode_input(input).map_err(|error| anyhow!("invalid engine input: {error}"))?;
        let step = self.step;
        let output = self
            .call_with_deadline(|store| step.call(store, (encoded,)))
            .context("engine component step trapped")?;
        decode_actions(&output.0).map_err(|error| anyhow!("invalid engine actions: {error}"))
    }

    pub fn serialize_state(&mut self) -> Result<Vec<u8>> {
        let serialize_state = self.serialize_state;
        let state = self
            .call_with_deadline(|store| serialize_state.call(store, ()))
            .context("engine component state serialization trapped")?;
        if state.0.len() > hi_engine_api::MAX_ENGINE_PAYLOAD_BYTES {
            return Err(anyhow!(ProtocolError::PayloadTooLarge("guest state")));
        }
        Ok(state.0)
    }

    /// Interrupt a guest call even when it burns fuel slowly or enters a
    /// non-terminating loop. The component has no imports, so epoch
    /// interruption is the host's wall-clock deadline mechanism. A short-lived
    /// notifier thread is joined immediately after the call and therefore
    /// cannot outlive a turn or retain guest state.
    fn call_with_deadline<T>(
        &mut self,
        call: impl FnOnce(&mut Store<GuestStoreState>) -> wasmtime::Result<T>,
    ) -> Result<T> {
        let deadline = self._runtime.policy.max_guest_step_ms;
        if deadline == 0 {
            bail!("engine step deadline must be greater than zero")
        }
        self.store.set_epoch_deadline(1);
        let engine = self._runtime.engine.clone();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let interrupter = std::thread::spawn(move || {
            if done_rx
                .recv_timeout(Duration::from_millis(deadline))
                .is_err()
            {
                engine.increment_epoch();
            }
        });
        let result = call(&mut self.store);
        let _ = done_tx.send(());
        let _ = interrupter.join();
        result
    }
}

impl DecisionEngine for WasmDecisionEngine {
    fn mode(&self) -> hi_engine_api::EngineMode {
        hi_engine_api::EngineMode::Wasm
    }

    fn step(&mut self, input: &EngineInput) -> Result<Vec<EngineAction>> {
        WasmDecisionEngine::step(self, input)
    }

    fn serialize_state(&mut self) -> Result<Vec<u8>> {
        WasmDecisionEngine::serialize_state(self)
    }
}

/// Host-side idempotency guard for guest actions. It claims a complete action
/// batch atomically before any effect broker is called, so a duplicate guest
/// action can never partially execute a batch.
#[derive(Default)]
pub struct ActionLedger {
    claimed: Mutex<HashSet<String>>,
}

impl ActionLedger {
    pub fn claim(&self, actions: &[EngineAction]) -> Result<()> {
        let mut keys = Vec::new();
        for action in actions {
            action
                .validate()
                .map_err(|error| anyhow!("invalid engine action: {error}"))?;
            match action {
                EngineAction::RequestModel {
                    idempotency_key, ..
                }
                | EngineAction::Present {
                    idempotency_key, ..
                }
                | EngineAction::UpdateState {
                    idempotency_key, ..
                }
                | EngineAction::Wait { idempotency_key }
                | EngineAction::Complete {
                    idempotency_key, ..
                }
                | EngineAction::Fail {
                    idempotency_key, ..
                } => keys.push(idempotency_key),
                EngineAction::ExecuteTool { request } => keys.push(&request.idempotency_key),
                EngineAction::ExecuteParallel { requests } => {
                    keys.extend(requests.iter().map(|request| &request.idempotency_key));
                }
            }
        }
        let mut claimed = self
            .claimed
            .lock()
            .map_err(|_| anyhow!("engine action ledger poisoned"))?;
        if keys.iter().any(|key| claimed.contains(*key)) {
            bail!("engine action idempotency key was already claimed")
        }
        let mut batch = HashSet::with_capacity(keys.len());
        for key in keys {
            if !batch.insert(key.clone()) {
                bail!("engine action batch contains a duplicate idempotency key")
            }
        }
        claimed.extend(batch);
        Ok(())
    }

    pub fn clear(&self) {
        if let Ok(mut claimed) = self.claimed.lock() {
            claimed.clear();
        }
    }
}

/// Small native-side status view used by TUI/CLI without exposing Wasmtime
/// internals to frontends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineStatus {
    pub mode: &'static str,
    pub current: Option<ModuleInfo>,
    pub pending: Option<ModuleInfo>,
}

pub fn status(runtime: &EngineRuntime) -> EngineStatus {
    EngineStatus {
        mode: "native-host",
        current: runtime.current().ok().flatten(),
        pending: runtime.pending().ok().flatten(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_engine_api::{EngineAction, ToolRequest, encode_actions};

    fn module_bytes() -> Vec<u8> {
        // A valid core wasm module is sufficient to test envelope validation;
        // component execution is covered by the component fixture tests in
        // environments that build the guest artifact.
        vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]
    }

    #[test]
    fn unsigned_artifact_requires_explicit_development_policy() {
        let bytes = module_bytes();
        let digest = hex::encode(Sha256::digest(&bytes));
        let manifest = EngineManifest::unsigned("0.1.0", digest);
        let strict = ModuleValidationPolicy::default();
        assert!(validate_artifact(&bytes, &manifest, &strict).is_err());
        let local = ModuleValidationPolicy {
            allow_unsigned: true,
            ..strict
        };
        assert!(validate_artifact(&bytes, &manifest, &local).is_ok());
    }

    #[test]
    fn manifest_capabilities_are_rejected_until_host_imports_exist() {
        let bytes = module_bytes();
        let digest = hex::encode(Sha256::digest(&bytes));
        let mut manifest = EngineManifest::unsigned("0.1.0", digest);
        manifest.required_capabilities.push("filesystem".into());
        let policy = ModuleValidationPolicy {
            allow_unsigned: true,
            ..Default::default()
        };
        let error = validate_artifact(&bytes, &manifest, &policy).unwrap_err();
        assert!(error.to_string().contains("unsupported host capabilities"));
    }

    #[test]
    fn signed_artifact_never_bypasses_trust_policy() {
        let bytes = module_bytes();
        let digest = hex::encode(Sha256::digest(&bytes));
        let mut manifest = EngineManifest::unsigned("0.1.0", digest);
        manifest.signature_hex = Some("00".repeat(64));
        let local = ModuleValidationPolicy {
            allow_unsigned: true,
            ..Default::default()
        };
        let error = validate_artifact(&bytes, &manifest, &local).unwrap_err();
        assert!(error.to_string().contains("no trusted engine key"));
    }

    #[test]
    fn action_validation_is_shared_with_the_host() {
        let encoded = encode_actions(&[EngineAction::ExecuteTool {
            request: ToolRequest {
                idempotency_key: "request:tool".into(),
                request_id: "request".into(),
                occurrence_id: "occurrence".into(),
                name: "read".into(),
                arguments_json: "{}".into(),
            },
        }])
        .unwrap();
        assert!(hi_engine_api::decode_actions(&encoded).is_ok());
    }

    #[test]
    fn action_ledger_claims_batches_atomically() {
        let ledger = ActionLedger::default();
        let request = ToolRequest {
            idempotency_key: "request:tool".into(),
            request_id: "request".into(),
            occurrence_id: "occurrence".into(),
            name: "read".into(),
            arguments_json: "{}".into(),
        };
        ledger
            .claim(&[EngineAction::ExecuteTool {
                request: request.clone(),
            }])
            .unwrap();
        assert!(
            ledger
                .claim(&[EngineAction::ExecuteTool { request }])
                .is_err()
        );
    }

    #[test]
    fn manifest_path_replaces_only_the_extension() {
        assert_eq!(
            manifest_path_for(Path::new("/tmp/engine.wasm")),
            PathBuf::from("/tmp/engine.manifest.json")
        );
    }

    #[test]
    fn runtime_starts_without_an_active_module() {
        let runtime = EngineRuntime::disabled().unwrap();
        assert!(runtime.current().unwrap().is_none());
        let lease = runtime.begin_turn().unwrap();
        assert_eq!(lease.generation(), None);
    }

    #[test]
    fn rollback_without_a_previous_module_is_safe() {
        let runtime = EngineRuntime::disabled().unwrap();
        assert!(runtime.rollback_active());
        assert!(runtime.current().unwrap().is_none());
        assert!(runtime.pending().unwrap().is_none());
    }
}
