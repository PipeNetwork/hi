#![cfg(feature = "native-cuda")]

use std::fs::File;
use std::io::{ErrorKind, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use hi_gguf::GgufFile;
use hi_local_core::model::ModelFamily;
use serde_json::{Value, json};

const ONE_BY_ONE_BMP_DATA_URL: &str = "data:image/bmp;base64,Qk06AAAAAAAAADYAAAAoAAAAAQAAAAEAAAABABgAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAD/AA==";
const SMOKE_MODEL_ID: &str = "cuda-real-smoke";

struct FixtureSpec {
    label: &'static str,
    env_name: &'static str,
    expected_family: ModelFamily,
    relative_candidates: &'static [&'static str],
}

const FAMILY_FIXTURES: &[FixtureSpec] = &[
    FixtureSpec {
        label: "llama3-instruct",
        env_name: "HI_CUDA_SMOKE_LLAMA3_GGUF",
        expected_family: ModelFamily::Llama,
        relative_candidates: &[
            "llama3-instruct/model.gguf",
            "llama-3-instruct/model.gguf",
            "llama-3.1-instruct/model.gguf",
            "llama-3.2-instruct/model.gguf",
        ],
    },
    FixtureSpec {
        label: "mistral",
        env_name: "HI_CUDA_SMOKE_MISTRAL_GGUF",
        // llama.cpp converts Mistral-7B with architecture "llama" (it IS the
        // llama layout), so family detection correctly reports Llama; a gguf
        // declaring "mistral" would map to ModelFamily::Mistral instead.
        expected_family: ModelFamily::Llama,
        relative_candidates: &["mistral/model.gguf", "mistral-7b-instruct/model.gguf"],
    },
    FixtureSpec {
        label: "gemma",
        env_name: "HI_CUDA_SMOKE_GEMMA_GGUF",
        expected_family: ModelFamily::Gemma,
        relative_candidates: &["gemma/model.gguf", "gemma-instruct/model.gguf"],
    },
    FixtureSpec {
        label: "phi",
        env_name: "HI_CUDA_SMOKE_PHI_GGUF",
        expected_family: ModelFamily::Phi,
        relative_candidates: &["phi/model.gguf", "phi-3-mini-instruct/model.gguf"],
    },
    FixtureSpec {
        label: "mixtral",
        env_name: "HI_CUDA_SMOKE_MIXTRAL_GGUF",
        // llama.cpp's Mixtral conversions declare architecture "llama" (MoE-ness
        // lives in the ffn_gate_inp/expert tensors, which the loader detects
        // regardless); a gguf declaring "mixtral" would map to
        // ModelFamily::Mixtral instead.
        expected_family: ModelFamily::Llama,
        relative_candidates: &["mixtral/model.gguf", "mixtral-instruct/model.gguf"],
    },
    FixtureSpec {
        label: "deepseek-dense",
        env_name: "HI_CUDA_SMOKE_DEEPSEEK_DENSE_GGUF",
        expected_family: ModelFamily::DeepSeek,
        // NOTE: DeepSeek-V2-Lite does NOT satisfy this fixture — its MLA
        // variant (direct attn_q + kv_a_mqa/kv_b, no q_a/q_b low-rank split)
        // is currently unsupported by the loader; a full-MLA (V2/V3-class)
        // gguf is required here.
        relative_candidates: &["deepseek-dense/model.gguf", "deepseek/model.gguf"],
    },
    FixtureSpec {
        label: "r1-distill-qwen-1.5b",
        env_name: "HI_CUDA_SMOKE_R1_DISTILL_GGUF",
        // DeepSeek's R1 distills onto Qwen2.5 bases keep the qwen2 layout.
        expected_family: ModelFamily::Qwen2,
        relative_candidates: &["r1-distill-qwen-1.5b/model.gguf"],
    },
    FixtureSpec {
        label: "glm-dense",
        env_name: "HI_CUDA_SMOKE_GLM_DENSE_GGUF",
        expected_family: ModelFamily::GlmFlash,
        relative_candidates: &["glm-dense/model.gguf", "glm4/model.gguf"],
    },
    // The shipping fleet: these are the models the backend is benchmarked and
    // tuned on, so the matrix must exercise them, not just the third-party
    // family samples above.
    FixtureSpec {
        label: "qwen25-vl-3b",
        env_name: "HI_CUDA_SMOKE_QWEN25_VL_TEXT_GGUF",
        expected_family: ModelFamily::Qwen2,
        relative_candidates: &[
            "qwen25-vl/model.gguf",
            "qwen25-vl-3b-gguf/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
        ],
    },
    FixtureSpec {
        label: "qwen25-vl-3b-nvfp4",
        env_name: "HI_CUDA_SMOKE_QWEN25_NVFP4_GGUF",
        expected_family: ModelFamily::Qwen2,
        relative_candidates: &["qwen25-vl-3b-gguf/Qwen2.5-VL-3B-Instruct-NVFP4.gguf"],
    },
    FixtureSpec {
        label: "qwen3-32b",
        env_name: "HI_CUDA_SMOKE_QWEN3_DENSE_GGUF",
        expected_family: ModelFamily::Qwen3,
        relative_candidates: &[
            "qwen3-32b/model.gguf",
            "qwen3-32b-gguf/Qwen3-32B-Q4_K_M.gguf",
        ],
    },
    FixtureSpec {
        label: "qwen3-30b-a3b-moe",
        env_name: "HI_CUDA_SMOKE_QWEN3_MOE_GGUF",
        expected_family: ModelFamily::Qwen3,
        relative_candidates: &[
            "qwen3-30b-a3b/model.gguf",
            "qwen3-30b-a3b-gguf/Qwen3-30B-A3B-Q4_K_M.gguf",
        ],
    },
    // Qwen3.5: hybrid gated-delta linear attention + full attention every 4th
    // layer, split ssm_beta/ssm_alpha projections, partial rope (64 of 256).
    FixtureSpec {
        label: "qwen35-9b",
        env_name: "HI_CUDA_SMOKE_QWEN35_GGUF",
        expected_family: ModelFamily::Qwen3,
        relative_candidates: &[
            "qwen35-9b/model.gguf",
            "qwen35-9b-gguf/Qwen3.5-9B-Q4_K_M.gguf",
        ],
    },
];

#[test]
fn cuda_real_text_http_smoke() -> Result<()> {
    let _guard = smoke_gpu_lock();
    let Some(model) = optional_fixture_path(
        "HI_CUDA_SMOKE_TEXT_GGUF",
        &[
            "llama3-instruct/model.gguf",
            "tinyllama/model.gguf",
            "tinyllama-1.1b-chat-gguf/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        ],
    ) else {
        eprintln!(
            "skipping CUDA real text smoke; set HI_CUDA_SMOKE_TEXT_GGUF or HI_CUDA_FIXTURES_DIR"
        );
        return Ok(());
    };

    let mut server = SmokeServer::start(&model, None, &[])?;
    let health = server.wait_for_health(Duration::from_secs(180))?;
    assert_eq!(health["status"], "ok");
    assert_eq!(health["execution"]["status"], "gpu");
    assert_eq!(health["scheduler"]["mode"], "continuous-iteration");
    assert_eq!(health["kv_cache"]["status"], "paged");

    let greedy = server.chat(json!({
        "model": SMOKE_MODEL_ID,
        "messages": [{"role": "user", "content": "Say hi."}],
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": false
    }))?;
    assert_chat_completion(&greedy)?;

    Ok(())
}

#[test]
fn cuda_real_text_stress_http_smoke() -> Result<()> {
    if !env_flag("HI_CUDA_SMOKE_TEXT_STRESS") {
        eprintln!("skipping CUDA real text stress smoke; set HI_CUDA_SMOKE_TEXT_STRESS=1");
        return Ok(());
    }

    let _guard = smoke_gpu_lock();
    let Some(model) = optional_fixture_path(
        "HI_CUDA_SMOKE_TEXT_GGUF",
        &[
            "llama3-instruct/model.gguf",
            "tinyllama/model.gguf",
            "tinyllama-1.1b-chat-gguf/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        ],
    ) else {
        eprintln!(
            "skipping CUDA real text stress smoke; set HI_CUDA_SMOKE_TEXT_GGUF or HI_CUDA_FIXTURES_DIR"
        );
        return Ok(());
    };

    let mut server = SmokeServer::start(&model, None, &[])?;
    let health = server.wait_for_health(Duration::from_secs(180))?;
    assert_eq!(health["status"], "ok");
    assert_eq!(health["execution"]["status"], "gpu");
    assert_eq!(health["scheduler"]["mode"], "continuous-iteration");
    assert_eq!(health["kv_cache"]["status"], "paged");

    let concurrent = std::thread::scope(|scope| {
        let left = scope.spawn(|| {
            server.chat(json!({
                "model": SMOKE_MODEL_ID,
                "messages": [{"role": "user", "content": "Name one color."}],
                "max_tokens": 1,
                "temperature": 0.0,
                "stream": false
            }))
        });
        let right = scope.spawn(|| {
            server.chat(json!({
                "model": SMOKE_MODEL_ID,
                "messages": [{"role": "user", "content": "Name one color."}],
                "max_tokens": 1,
                "temperature": 0.0,
                "stream": false
            }))
        });
        let left = left
            .join()
            .map_err(|_| anyhow!("left concurrent greedy smoke panicked"))??;
        let right = right
            .join()
            .map_err(|_| anyhow!("right concurrent greedy smoke panicked"))??;
        Ok::<_, anyhow::Error>((left, right))
    })?;
    assert_chat_completion(&concurrent.0)?;
    assert_chat_completion(&concurrent.1)?;

    if env_flag("HI_CUDA_SMOKE_TEXT_SAMPLED_STRESS") {
        let sampled = server.chat(json!({
            "model": SMOKE_MODEL_ID,
            "messages": [{"role": "user", "content": "Name one color."}],
            "max_tokens": 1,
            "temperature": 0.8,
            "top_p": 0.9,
            "seed": 7,
            "stream": false
        }))?;
        assert_chat_completion(&sampled)?;
    }

    Ok(())
}

#[test]
fn cuda_real_text_page_exhaustion_http_smoke() -> Result<()> {
    let _guard = smoke_gpu_lock();
    let Some(model) = optional_fixture_path(
        "HI_CUDA_SMOKE_TEXT_GGUF",
        &[
            "llama3-instruct/model.gguf",
            "tinyllama/model.gguf",
            "tinyllama-1.1b-chat-gguf/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        ],
    ) else {
        eprintln!(
            "skipping CUDA real page-exhaustion smoke; set HI_CUDA_SMOKE_TEXT_GGUF or HI_CUDA_FIXTURES_DIR"
        );
        return Ok(());
    };

    let mut server = SmokeServer::start(
        &model,
        None,
        &["--max-batched-tokens", "16", "--kv-page-size", "16"],
    )?;
    let health = server.wait_for_health(Duration::from_secs(180))?;
    assert_eq!(health["status"], "ok");
    assert_eq!(health["kv_cache"]["status"], "paged");
    assert_eq!(health["kv_cache"]["page_size"], 16);
    assert_eq!(health["kv_cache"]["pages_total"], 1);

    let (status, rejected) = server.chat_with_status(json!({
        "model": SMOKE_MODEL_ID,
        "messages": [{"role": "user", "content": "Use enough cache."}],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": false
    }))?;
    assert_eq!(status, 503, "{rejected}");
    assert_eq!(rejected["error"]["code"], "insufficient_gpu_memory");
    assert_eq!(rejected["error"]["details"]["page_size"], 16);
    assert_eq!(rejected["error"]["details"]["pages_total"], 1);
    assert!(
        rejected["error"]["details"]["required_pages"]
            .as_u64()
            .unwrap_or(0)
            >= 2,
        "{rejected}"
    );

    let health = server.wait_for_health_counter(
        "scheduler.admission.by_reason.insufficient_gpu_memory",
        1,
        Duration::from_secs(30),
    )?;
    assert_eq!(health["kv_cache"]["pages_used"], 0);

    Ok(())
}

#[test]
fn cuda_real_qwen25_vl_image_http_smoke() -> Result<()> {
    let _guard = smoke_gpu_lock();
    let Some(model) = optional_fixture_path(
        "HI_CUDA_SMOKE_QWEN25_VL_GGUF",
        &[
            "qwen25-vl/model.gguf",
            "qwen2.5-vl/model.gguf",
            "qwen25-vl-3b-gguf/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
        ],
    ) else {
        eprintln!(
            "skipping CUDA real Qwen2.5-VL smoke; set HI_CUDA_SMOKE_QWEN25_VL_GGUF or HI_CUDA_FIXTURES_DIR"
        );
        return Ok(());
    };
    let Some(mmproj) = optional_fixture_path(
        "HI_CUDA_SMOKE_QWEN25_VL_MMPROJ",
        &[
            "qwen25-vl/mmproj.gguf",
            "qwen2.5-vl/mmproj.gguf",
            "qwen25-vl-3b-gguf/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf",
        ],
    ) else {
        eprintln!(
            "skipping CUDA real Qwen2.5-VL smoke; set HI_CUDA_SMOKE_QWEN25_VL_MMPROJ or HI_CUDA_FIXTURES_DIR"
        );
        return Ok(());
    };

    let mut server = SmokeServer::start(&model, Some(&mmproj), &["--max-batched-tokens", "4096"])?;
    let health = server.wait_for_health(Duration::from_secs(240))?;
    assert_eq!(health["status"], "ok");
    assert_eq!(health["execution"]["status"], "gpu");
    assert_eq!(health["scheduler"]["mode"], "continuous-iteration");
    assert_eq!(health["multimodal_projector"]["status"], "mmproj-loaded");

    let image = server.chat(json!({
        "model": SMOKE_MODEL_ID,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in two words."},
                {"type": "image_url", "image_url": {"url": ONE_BY_ONE_BMP_DATA_URL, "detail": "low"}}
            ]
        }],
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": false
    }))?;
    assert_chat_completion(&image)?;

    let video = server.chat(json!({
        "model": SMOKE_MODEL_ID,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this short video in two words."},
                {"type": "video", "video": {"frames": [
                    {"url": ONE_BY_ONE_BMP_DATA_URL},
                    {"url": ONE_BY_ONE_BMP_DATA_URL}
                ], "fps": 1.0, "max_frames": 2}}
            ]
        }],
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": false
    }))?;
    assert_chat_completion(&video)?;

    Ok(())
}

#[test]
fn cuda_real_family_fixture_metadata_smoke() -> Result<()> {
    let mut checked = Vec::new();
    let mut missing = Vec::new();
    let mut failures: Vec<String> = Vec::new();
    for spec in FAMILY_FIXTURES {
        let Some(path) = optional_fixture_path(spec.env_name, spec.relative_candidates) else {
            missing.push(spec.label);
            continue;
        };
        let outcome = (|| -> Result<()> {
            let gguf = GgufFile::open(&path)
                .with_context(|| format!("opening {} fixture {}", spec.label, path.display()))?;
            let config = gguf
                .qwen_config()
                .with_context(|| format!("parsing {} fixture {}", spec.label, path.display()))?;
            if config.family != spec.expected_family {
                bail!(
                    "{} fixture {} reported family {:?}, expected {:?}",
                    spec.label,
                    path.display(),
                    config.family,
                    spec.expected_family
                );
            }
            gguf.validate_qwen_tensors()
                .with_context(|| format!("validating {} fixture {}", spec.label, path.display()))?;
            Ok(())
        })();
        match outcome {
            Ok(()) => checked.push(spec.label),
            Err(err) => failures.push(format!("{}: {err:#}", spec.label)),
        }
    }
    if !failures.is_empty() {
        bail!(
            "CUDA family fixture metadata failure(s) (checked ok: {}):\n{}",
            checked.join(", "),
            failures.join("\n")
        );
    }

    if env_flag("HI_CUDA_REQUIRE_REAL_FIXTURE_MATRIX") && !missing.is_empty() {
        bail!(
            "missing required CUDA real fixture(s): {}; set per-family HI_CUDA_SMOKE_*_GGUF paths or populate HI_CUDA_FIXTURES_DIR",
            missing.join(", ")
        );
    }
    if checked.is_empty() {
        eprintln!(
            "skipping CUDA real family fixture metadata smoke; set HI_CUDA_FIXTURES_DIR or per-family HI_CUDA_SMOKE_*_GGUF paths"
        );
    } else {
        eprintln!(
            "checked CUDA real family fixture(s): {}",
            checked.join(", ")
        );
    }
    Ok(())
}

#[test]
fn cuda_real_family_text_http_smoke() -> Result<()> {
    if !env_flag("HI_CUDA_SMOKE_FAMILY_MATRIX") {
        eprintln!("skipping CUDA real family text HTTP smoke; set HI_CUDA_SMOKE_FAMILY_MATRIX=1");
        return Ok(());
    }

    let _guard = smoke_gpu_lock();
    let mut checked = Vec::new();
    let mut missing = Vec::new();
    let mut failures: Vec<String> = Vec::new();
    for spec in FAMILY_FIXTURES {
        let Some(path) = optional_fixture_path(spec.env_name, spec.relative_candidates) else {
            missing.push(spec.label);
            continue;
        };
        // One broken fixture must not hide the rest of the matrix: run every
        // family, collect failures, and report them all at the end.
        let outcome = (|| -> Result<()> {
            let mut server = SmokeServer::start(&path, None, &[])?;
            let health = server.wait_for_health(Duration::from_secs(300))?;
            if health["status"] != "ok" {
                bail!("health not ok: {health}");
            }
            if health["execution"]["status"] != "gpu" {
                bail!("execution not gpu: {health}");
            }
            if health["scheduler"]["mode"] != "continuous-iteration" {
                bail!("scheduler not continuous: {health}");
            }
            if health["kv_cache"]["status"] != "paged" {
                bail!("kv cache not paged: {health}");
            }
            let attention_status = health["attention"]["status"].as_str().ok_or_else(|| {
                anyhow!("family fixture health missing attention status: {health}")
            })?;
            if !matches!(attention_status, "tiled-paged" | "paged-generic") {
                bail!("unsupported CUDA attention status {attention_status}: {health}");
            }
            let response = server.chat(json!({
                "model": SMOKE_MODEL_ID,
                "messages": [{"role": "user", "content": "Reply with one short word."}],
                "max_tokens": 1,
                "temperature": 0.0,
                "stream": false
            }))?;
            assert_chat_completion(&response)?;
            Ok(())
        })();
        match outcome {
            Ok(()) => checked.push(spec.label),
            Err(err) => failures.push(format!("{}: {err:#}", spec.label)),
        }
    }
    if !failures.is_empty() {
        bail!(
            "CUDA family matrix failure(s) (checked ok: {}):\n{}",
            checked.join(", "),
            failures.join("\n")
        );
    }

    if env_flag("HI_CUDA_REQUIRE_REAL_FIXTURE_MATRIX") && !missing.is_empty() {
        bail!(
            "missing required CUDA real HTTP fixture(s): {}; set per-family HI_CUDA_SMOKE_*_GGUF paths or populate HI_CUDA_FIXTURES_DIR",
            missing.join(", ")
        );
    }
    if checked.is_empty() {
        bail!(
            "HI_CUDA_SMOKE_FAMILY_MATRIX=1 but no family fixtures were found; set HI_CUDA_FIXTURES_DIR or per-family HI_CUDA_SMOKE_*_GGUF paths"
        );
    }
    eprintln!(
        "checked CUDA real family HTTP fixture(s): {}",
        checked.join(", ")
    );
    Ok(())
}

fn smoke_gpu_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .expect("CUDA smoke test mutex poisoned")
}

fn optional_fixture_path(env_name: &str, relative_candidates: &[&str]) -> Option<PathBuf> {
    if let Ok(path) = std::env::var(env_name) {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
        eprintln!(
            "ignoring {env_name}={}; path does not exist",
            path.display()
        );
    }

    let root = std::env::var_os("HI_CUDA_FIXTURES_DIR").map(PathBuf::from)?;
    relative_candidates
        .iter()
        .map(|relative| root.join(relative))
        .find(|path| path.exists())
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn assert_chat_completion(body: &Value) -> Result<()> {
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| {
            anyhow!("chat completion response did not include message content: {body}")
        })?;
    let finish_reason = body["choices"][0]["finish_reason"]
        .as_str()
        .ok_or_else(|| anyhow!("chat completion response did not include finish_reason: {body}"))?;
    if finish_reason != "stop" && finish_reason != "length" {
        bail!("unexpected finish_reason {finish_reason}: {body}");
    }
    if body["choices"][0]["message"]["role"] != "assistant" {
        bail!("chat completion response did not include assistant role: {body}");
    }
    if content.len() > 4096 {
        bail!("chat completion content is unexpectedly large");
    }
    Ok(())
}

struct SmokeServer {
    child: Child,
    addr: SocketAddr,
    stderr_path: PathBuf,
}

impl SmokeServer {
    fn start(model: &Path, mmproj: Option<&Path>, extra_args: &[&str]) -> Result<Self> {
        let addr = free_loopback_addr()?;
        let temp_dir = smoke_temp_dir()?;
        let stderr_path = temp_dir.join("hi-local.stderr.log");
        let stderr = File::create(&stderr_path)
            .with_context(|| format!("creating smoke stderr log {}", stderr_path.display()))?;
        let stdout = File::create(temp_dir.join("hi-local.stdout.log"))?;
        let mut command = Command::new(env!("CARGO_BIN_EXE_hi-local"));
        command
            .arg("serve")
            .arg("--backend")
            .arg("cuda")
            .arg("--execution")
            .arg("gpu")
            .arg("--host")
            .arg(addr.ip().to_string())
            .arg("--port")
            .arg(addr.port().to_string())
            .arg("--model-id")
            .arg(SMOKE_MODEL_ID)
            .arg("--max-batch-size")
            .arg("2")
            .arg("--max-active-requests")
            .arg("2")
            .arg("--max-wait-us")
            .arg("100000")
            .arg("--kv-cache-mode")
            .arg("paged")
            .args(extra_args);
        if let Some(mmproj) = mmproj {
            command.arg("--mmproj-path").arg(mmproj);
        }
        command
            .arg(model)
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr));

        let child = command.spawn().with_context(|| {
            format!(
                "starting hi-local CUDA smoke server for {}",
                model.display()
            )
        })?;
        Ok(Self {
            child,
            addr,
            stderr_path,
        })
    }

    fn wait_for_health(&mut self, timeout: Duration) -> Result<Value> {
        let started = Instant::now();
        let mut last_err = None;
        while started.elapsed() < timeout {
            // A dead child never becomes healthy: surface its exit and stderr
            // immediately instead of burning the whole timeout (a broken
            // fixture used to cost 300s of wall clock before reporting).
            if let Ok(Some(status)) = self.child.try_wait() {
                return Err(anyhow!(
                    "smoke server exited ({status}) before becoming healthy; stderr:\n{}",
                    self.stderr_tail()
                ));
            }
            match self.get_json("/health") {
                Ok(health) if health["status"] == "ok" => return Ok(health),
                Ok(health) => last_err = Some(anyhow!("health was not ready: {health}")),
                Err(err) => last_err = Some(err),
            }
            std::thread::sleep(Duration::from_millis(250));
        }
        Err(anyhow!(
            "timed out waiting for smoke server health; last error: {}; stderr:\n{}",
            last_err
                .map(|err| err.to_string())
                .unwrap_or_else(|| "none".to_string()),
            self.stderr_tail()
        ))
    }

    fn wait_for_health_counter(
        &self,
        dotted_path: &str,
        min: u64,
        timeout: Duration,
    ) -> Result<Value> {
        let started = Instant::now();
        let mut last = None;
        while started.elapsed() < timeout {
            let health = self.get_json("/health")?;
            let value = json_path_u64(&health, dotted_path).unwrap_or(0);
            if value >= min {
                return Ok(health);
            }
            last = Some(value);
            std::thread::sleep(Duration::from_millis(100));
        }
        bail!(
            "timed out waiting for health counter {dotted_path}>={min}; last={}",
            last.unwrap_or(0)
        )
    }

    fn chat(&self, body: Value) -> Result<Value> {
        self.post_json("/v1/chat/completions", body)
    }

    fn chat_with_status(&self, body: Value) -> Result<(u16, Value)> {
        self.post_json_with_status("/v1/chat/completions", body)
    }

    fn get_json(&self, path: &str) -> Result<Value> {
        let request = format!(
            "GET {path} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
            self.addr
        );
        let response = self.round_trip(request)?;
        parse_json_response(&response)
    }

    fn post_json(&self, path: &str, body: Value) -> Result<Value> {
        let response = self.round_trip(post_request(self.addr, path, body)?)?;
        parse_json_response(&response)
    }

    fn post_json_with_status(&self, path: &str, body: Value) -> Result<(u16, Value)> {
        let response = self.round_trip(post_request(self.addr, path, body)?)?;
        parse_json_response_with_status(&response)
    }

    fn round_trip(&self, request: String) -> Result<String> {
        let mut stream = TcpStream::connect(self.addr)
            .with_context(|| format!("connecting to smoke server {}", self.addr))?;
        stream
            .set_read_timeout(Some(Duration::from_secs(300)))
            .context("setting smoke read timeout")?;
        stream
            .set_write_timeout(Some(Duration::from_secs(10)))
            .context("setting smoke write timeout")?;
        stream
            .write_all(request.as_bytes())
            .context("writing smoke HTTP request")?;
        let response = read_http_response(&mut stream).context("reading smoke HTTP response")?;
        Ok(response)
    }

    fn stderr_tail(&self) -> String {
        let Ok(mut file) = File::open(&self.stderr_path) else {
            return String::new();
        };
        let mut text = String::new();
        let _ = file.read_to_string(&mut text);
        let start = text.len().saturating_sub(4096);
        text[start..].to_string()
    }
}

impl Drop for SmokeServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn post_request(addr: SocketAddr, path: &str, body: Value) -> Result<String> {
    let body = body.to_string();
    Ok(format!(
        "POST {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    ))
}

fn parse_json_response(response: &str) -> Result<Value> {
    let (status_code, body) = parse_json_response_with_status(response)?;
    if status_code != 200 {
        bail!("smoke HTTP request failed with status {status_code}: {body}");
    }
    Ok(body)
}

fn parse_json_response_with_status(response: &str) -> Result<(u16, Value)> {
    let (head, body) = response
        .split_once("\r\n\r\n")
        .ok_or_else(|| anyhow!("invalid HTTP response: {response}"))?;
    let status_line = head
        .lines()
        .next()
        .ok_or_else(|| anyhow!("HTTP response was empty"))?;
    let status_code = http_status_code(status_line)?;
    let body = serde_json::from_str(body)
        .with_context(|| format!("parsing JSON response body: {body}"))?;
    Ok((status_code, body))
}

fn http_status_code(status_line: &str) -> Result<u16> {
    status_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| anyhow!("HTTP response status line was malformed: {status_line}"))?
        .parse()
        .with_context(|| format!("parsing HTTP response status line: {status_line}"))
}

fn read_http_response(stream: &mut TcpStream) -> Result<String> {
    let mut bytes = Vec::new();
    let mut buf = [0u8; 4096];
    let deadline = Instant::now() + Duration::from_secs(300);
    loop {
        let read = match stream.read(&mut buf) {
            Ok(read) => read,
            Err(err) if matches!(err.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                if http_response_complete(&bytes)? {
                    break;
                }
                if Instant::now() >= deadline {
                    return Err(err).context("timed out reading smoke HTTP response");
                }
                std::thread::sleep(Duration::from_millis(25));
                continue;
            }
            Err(err) => return Err(err.into()),
        };
        if read == 0 {
            break;
        }
        bytes.extend_from_slice(&buf[..read]);
        if http_response_complete(&bytes)? {
            break;
        }
    }
    String::from_utf8(bytes).context("smoke HTTP response was not UTF-8")
}

fn http_response_complete(bytes: &[u8]) -> Result<bool> {
    let Some(header_end) = find_header_end(bytes) else {
        return Ok(false);
    };
    let header_text = std::str::from_utf8(&bytes[..header_end])
        .context("smoke HTTP response headers were not UTF-8")?;
    if let Some(content_length) = header_value(header_text, "content-length") {
        let content_length = content_length
            .parse::<usize>()
            .with_context(|| format!("invalid content-length header {content_length}"))?;
        return Ok(bytes.len() >= header_end + 4 + content_length);
    }
    if header_value(header_text, "transfer-encoding")
        .map(|value| value.to_ascii_lowercase().contains("chunked"))
        .unwrap_or(false)
    {
        return Ok(bytes[header_end + 4..]
            .windows(5)
            .any(|chunk| chunk == b"0\r\n\r\n"));
    }
    Ok(false)
}

fn find_header_end(bytes: &[u8]) -> Option<usize> {
    bytes.windows(4).position(|window| window == b"\r\n\r\n")
}

fn header_value<'a>(headers: &'a str, name: &str) -> Option<&'a str> {
    headers.lines().skip(1).find_map(|line| {
        let (key, value) = line.split_once(':')?;
        key.eq_ignore_ascii_case(name).then_some(value.trim())
    })
}

fn json_path_u64(value: &Value, dotted_path: &str) -> Option<u64> {
    let mut current = value;
    for key in dotted_path.split('.') {
        current = current.get(key)?;
    }
    current.as_u64()
}

fn free_loopback_addr() -> Result<SocketAddr> {
    let listener = TcpListener::bind("127.0.0.1:0").context("binding smoke test port")?;
    let addr = listener.local_addr().context("reading smoke test port")?;
    drop(listener);
    Ok(addr)
}

fn smoke_temp_dir() -> Result<PathBuf> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "hi-local-cuda-smoke-{}-{nanos}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("creating smoke temp dir {}", dir.display()))?;
    Ok(dir)
}
