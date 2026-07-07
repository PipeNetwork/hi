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

    cc::Build::new()
        .cuda(true)
        .file("src/kernels.cu")
        .flag("-std=c++17")
        .compile("hi_cuda_kernels");
}
