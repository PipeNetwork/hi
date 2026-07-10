use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=HI_CUDA_ARCH");
    println!("cargo:rerun-if-changed=src/kernels.cu");

    if env::var_os("CARGO_FEATURE_NATIVE_CUDA").is_none() {
        return;
    }

    let cuda_home = env::var_os("CUDA_HOME")
        .or_else(|| env::var_os("CUDA_PATH"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
    println!(
        "cargo:rustc-link-search=native={}",
        cuda_home.join("lib64").display()
    );
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");

    let mut build = cc::Build::new();
    build.cuda(true).file("src/kernels.cu").flag("-std=c++17");
    for flag in gencode_flags() {
        build.flag(&flag);
    }
    build.compile("hi_cuda_kernels");
}

// Build the `-gencode` flags. We want native SASS for the build host's GPU (so the
// tensor-core / dp4a kernels are scheduled for the real arch instead of JIT'd forward
// from an old virtual arch) *plus* a PTX fallback so the artifact still loads on other
// GPUs. `compute_NN` PTX alone is what the crate historically shipped and it JITs at
// load — measurably worse for the WMMA kernels on Blackwell.
//
// Override with HI_CUDA_ARCH:
//   - "sm_121" / "121" / "12.1"  -> native SASS for that arch + its PTX fallback
//   - "compute_75"               -> PTX only (legacy portable behavior, no native SASS)
//   - "compute_90,sm_90"         -> exactly those two, comma-separated, verbatim gencode targets
fn gencode_flags() -> Vec<String> {
    if let Ok(spec) = env::var("HI_CUDA_ARCH") {
        return parse_arch_spec(&spec);
    }
    // No override: emit native SASS for the detected GPU + a portable PTX fallback.
    // A JIT-portable compute_75 PTX floor keeps the artifact loadable on pre-Blackwell
    // cards (dp4a needs sm_61+; the WMMA attention needs sm_75+).
    match detect_compute_cap() {
        Some(cap) => vec![
            format!("-gencode=arch=compute_{cap},code=sm_{cap}"),
            format!("-gencode=arch=compute_{cap},code=compute_{cap}"),
        ],
        None => {
            println!(
                "cargo:warning=hi-cuda: could not detect GPU compute capability; \
                 falling back to compute_75 PTX (set HI_CUDA_ARCH to target a specific arch)"
            );
            vec!["-gencode=arch=compute_75,code=compute_75".to_string()]
        }
    }
}

fn parse_arch_spec(spec: &str) -> Vec<String> {
    let spec = spec.trim();
    // Verbatim comma-separated "arch=...,code=..." pairs pass straight through as targets.
    if spec.contains('=') {
        return vec![format!("-gencode={spec}")];
    }
    // Comma-separated shorthands like "sm_90,compute_90" -> one -gencode each.
    if spec.contains(',') {
        return spec
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .flat_map(|s| parse_arch_spec(s.trim()))
            .collect();
    }
    if let Some(cap) = spec.strip_prefix("compute_") {
        // Explicit virtual arch -> PTX only (legacy behavior: JIT at load, max portability).
        return vec![format!("-gencode=arch=compute_{cap},code=compute_{cap}")];
    }
    if let Some(cap) = spec.strip_prefix("sm_") {
        // Explicit real arch -> native SASS + its PTX fallback.
        return vec![
            format!("-gencode=arch=compute_{cap},code=sm_{cap}"),
            format!("-gencode=arch=compute_{cap},code=compute_{cap}"),
        ];
    }
    // Bare capability like "121" or "12.1".
    let cap = spec.replace('.', "");
    vec![
        format!("-gencode=arch=compute_{cap},code=sm_{cap}"),
        format!("-gencode=arch=compute_{cap},code=compute_{cap}"),
    ]
}

// Query the build host's GPU compute capability via nvidia-smi (e.g. "12.1" -> "121").
fn detect_compute_cap() -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let cap = text.lines().next()?.trim().replace('.', "");
    if cap.is_empty() || !cap.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(cap)
}
