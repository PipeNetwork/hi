use std::env;

fn main() {
    // When hi-mlx is built against a prebuilt MLX (HI_MLX_SYSTEM_MLX_PREFIX, see
    // pmetal-mlx-sys), embed an rpath so the hi-mlx binary resolves libmlx.dylib at runtime.
    // This must live in the binary crate: a -sys crate's build script cannot add an rpath to
    // the final linked artifact.
    println!("cargo:rerun-if-env-changed=HI_MLX_SYSTEM_MLX_PREFIX");
    if let Ok(prefix) = env::var("HI_MLX_SYSTEM_MLX_PREFIX")
        && !prefix.is_empty()
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{prefix}/lib");
    }
}
