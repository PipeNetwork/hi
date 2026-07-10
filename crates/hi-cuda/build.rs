use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
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

    // compute_75 PTX: dp4a-capable and JIT-portable to any sm_75+ GPU. Needed for
    // the __dp4a int8 dot-product in the Q4_0 GEMV. Override via HI_CUDA_ARCH.
    let cuda_arch = env::var("HI_CUDA_ARCH").unwrap_or_else(|_| "compute_75".to_string());
    cc::Build::new()
        .cuda(true)
        .file("src/kernels.cu")
        .flag("-std=c++17")
        .flag(format!("-gencode=arch={cuda_arch},code={cuda_arch}"))
        .compile("hi_cuda_kernels");
}
