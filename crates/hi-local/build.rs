use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=HI_MLX_SYSTEM_MLX_PREFIX");
    println!("cargo:rerun-if-env-changed=HI_MLX_BUNDLE_RPATH");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos")
        || env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("aarch64")
    {
        return;
    }

    if env::var_os("HI_MLX_BUNDLE_RPATH").is_some_and(|value| value == "1") {
        // hi-local is the final executable. Link arguments emitted by the
        // transitive hi-mlx build script do not reliably reach this binary,
        // so the sidecar must add its own relocatable search path.
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/../lib/mlx");
    } else if let Ok(prefix) = env::var("HI_MLX_SYSTEM_MLX_PREFIX")
        && !prefix.is_empty()
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{prefix}/lib");
    }
}
