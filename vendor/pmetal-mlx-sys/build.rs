extern crate cmake;

use cmake::Config;
use std::{env, path::PathBuf, process::Command};

/// Find the clang runtime library path dynamically using xcrun
fn find_clang_rt_path() -> Option<String> {
    // Use xcrun to find the active toolchain path
    let output = Command::new("xcrun")
        .args(["--show-sdk-platform-path"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    // Get the developer directory which contains the toolchain
    let output = Command::new("xcode-select")
        .args(["--print-path"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let developer_dir = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let toolchain_base = format!(
        "{}/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang",
        developer_dir
    );

    // Find the clang version directory (it varies by Xcode version)
    let clang_dir = std::fs::read_dir(&toolchain_base).ok()?;
    for entry in clang_dir.flatten() {
        let darwin_path = entry.path().join("lib/darwin");
        let clang_rt_lib = darwin_path.join("libclang_rt.osx.a");
        if clang_rt_lib.exists() {
            return Some(darwin_path.to_string_lossy().to_string());
        }
    }

    None
}

/// Resolve the macOS deployment target.
///
/// Enforces a minimum of 14.0 (MLX's requirement for Metal support).
/// If `MACOSX_DEPLOYMENT_TARGET` is set to a higher value, that is used instead.
/// Cargo/Tauri often default to 10.13, which MLX's CMakeLists.txt rejects.
#[cfg(target_os = "macos")]
fn resolve_deployment_target() -> String {
    const MLX_MIN_MACOS: (u32, u32) = (14, 0);

    if let Ok(val) = env::var("MACOSX_DEPLOYMENT_TARGET") {
        let parts: Vec<u32> = val.split('.').filter_map(|s| s.parse().ok()).collect();
        let major = parts.first().copied().unwrap_or(0);
        let minor = parts.get(1).copied().unwrap_or(0);
        if (major, minor) >= MLX_MIN_MACOS {
            return val;
        }
    }
    format!("{}.{}", MLX_MIN_MACOS.0, MLX_MIN_MACOS.1)
}

/// Copy src/mlx-c to a staging directory and inject the metallib search-path
/// patch into the CMakeLists.txt. This avoids modifying the mlx-c git submodule
/// while ensuring the patch is applied when MLX is fetched via FetchContent.
fn prepare_mlx_c_source() -> PathBuf {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let staged = out_dir.join("mlx-c-staged");
    let src = PathBuf::from("src/mlx-c");

    // Copy the entire mlx-c source tree to the staging area
    if staged.exists() {
        std::fs::remove_dir_all(&staged).expect("Failed to clean staged mlx-c");
    }
    copy_dir_recursive(&src, &staged).expect("Failed to copy mlx-c to staging");

    // Copy our patch file into the staged source
    let patches_dir = staged.join("patches");
    std::fs::create_dir_all(&patches_dir).expect("Failed to create patches dir");
    std::fs::copy(
        "patches/metallib-search-path.patch",
        patches_dir.join("metallib-search-path.patch"),
    )
    .expect("Failed to copy metallib patch");

    // Inject PATCH_COMMAND into the FetchContent_Declare for MLX
    let cmake_path = staged.join("CMakeLists.txt");
    let cmake_content =
        std::fs::read_to_string(&cmake_path).expect("Failed to read CMakeLists.txt");
    let patched = cmake_content.replace(
        "GIT_TAG v0.31.1)",
        "GIT_TAG v0.31.1\n    PATCH_COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/patches/metallib-search-path.patch || true)",
    );
    std::fs::write(&cmake_path, patched).expect("Failed to write patched CMakeLists.txt");

    // Tell cargo to rerun if the patch changes
    println!("cargo:rerun-if-changed=patches/metallib-search-path.patch");

    staged
}

fn copy_dir_recursive(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let dest_path = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_recursive(&entry.path(), &dest_path)?;
        } else if ty.is_symlink() {
            // Resolve symlinks (common in git submodules)
            let target = std::fs::read_link(entry.path())?;
            let resolved = if target.is_absolute() {
                target
            } else {
                entry.path().parent().unwrap().join(&target)
            };
            if resolved.is_dir() {
                copy_dir_recursive(&resolved, &dest_path)?;
            } else {
                std::fs::copy(&resolved, &dest_path)?;
            }
        } else {
            std::fs::copy(entry.path(), &dest_path)?;
        }
    }
    Ok(())
}

fn build_and_link_mlx_c() {
    // MLX requires macOS >= 14.0 for Metal support. Override the deployment
    // target early so the cmake crate (and cc crate) don't inject a lower
    // -mmacosx-version-min flag into CFLAGS/CXXFLAGS. Without this, Cargo's
    // default target (10.13) causes MLX's CMakeLists.txt to reject the build.
    #[cfg(target_os = "macos")]
    {
        let target = resolve_deployment_target();
        env::set_var("MACOSX_DEPLOYMENT_TARGET", &target);
    }

    // If HI_MLX_SYSTEM_MLX_PREFIX points at a prebuilt MLX install (e.g. the pip or Homebrew
    // MLX, which ships a complete metallib), link against it instead of compiling MLX from
    // source. This is required on macOS with a very new Metal Toolchain (Metal 4), where MLX's
    // from-source build produces runtime-JIT'd Metal kernels that fail to compile bf16 kernels;
    // the prebuilt MLX avoids runtime JIT via its precompiled metallib.
    let system_mlx = env::var("HI_MLX_SYSTEM_MLX_PREFIX").ok().filter(|p| !p.is_empty());
    println!("cargo:rerun-if-env-changed=HI_MLX_SYSTEM_MLX_PREFIX");

    let mlx_c_src = prepare_mlx_c_source();
    let mut config = Config::new(&mlx_c_src);
    config.very_verbose(true);
    config.define("CMAKE_INSTALL_PREFIX", ".");
    if let Some(prefix) = &system_mlx {
        config.define("MLX_C_USE_SYSTEM_MLX", "ON");
        config.define("CMAKE_PREFIX_PATH", prefix);
    }

    #[cfg(target_os = "macos")]
    {
        let target = resolve_deployment_target();
        config.define("CMAKE_OSX_DEPLOYMENT_TARGET", &target);
    }

    // Use Xcode's clang to ensure compatibility with the macOS SDK
    config.define("CMAKE_C_COMPILER", "/usr/bin/cc");
    config.define("CMAKE_CXX_COMPILER", "/usr/bin/c++");

    #[cfg(debug_assertions)]
    {
        config.define("CMAKE_BUILD_TYPE", "Debug");
    }

    #[cfg(not(debug_assertions))]
    {
        config.define("CMAKE_BUILD_TYPE", "Release");
    }

    config.define("MLX_BUILD_METAL", "OFF");
    config.define("MLX_BUILD_ACCELERATE", "OFF");

    #[cfg(feature = "metal")]
    {
        config.define("MLX_BUILD_METAL", "ON");
    }

    #[cfg(feature = "accelerate")]
    {
        config.define("MLX_BUILD_ACCELERATE", "ON");
    }

    // build the mlx-c project
    let dst = config.build();

    println!("cargo:rustc-link-search=native={}/build/lib", dst.display());
    if let Some(prefix) = &system_mlx {
        // Link the prebuilt MLX dylib (mlxc is still built statically from our source), and add
        // an rpath so the final binary resolves libmlx.dylib at runtime.
        println!("cargo:rustc-link-search=native={prefix}/lib");
        println!("cargo:rustc-link-lib=static=mlxc");
        println!("cargo:rustc-link-lib=dylib=mlx");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{prefix}/lib");
    } else {
        println!("cargo:rustc-link-lib=static=mlx");
        println!("cargo:rustc-link-lib=static=mlxc");
    }

    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=dylib=objc");
    println!("cargo:rustc-link-lib=framework=Foundation");

    #[cfg(feature = "metal")]
    {
        println!("cargo:rustc-link-lib=framework=Metal");
    }

    #[cfg(feature = "accelerate")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    // Link against Xcode's clang runtime for ___isPlatformVersionAtLeast symbol
    // This is needed on macOS 26+ where the bundled LLVM runtime may be outdated
    // See: https://github.com/conda-forge/llvmdev-feedstock/issues/244
    if let Some(clang_rt_path) = find_clang_rt_path() {
        println!("cargo:rustc-link-search={}", clang_rt_path);
        println!("cargo:rustc-link-lib=static=clang_rt.osx");
    }

    // Cache mlx.metallib to ~/.cache/pmetal/lib/ so the binary works regardless
    // of where it's installed. This is critical for `cargo install` where the
    // build directory is cleaned up after the binary is placed.
    #[cfg(feature = "metal")]
    {
        let metallib = dst.join("build/lib/mlx.metallib");
        if system_mlx.is_none() && metallib.exists() {
            if let Ok(home) = env::var("HOME") {
                let cache_dir = PathBuf::from(home).join(".cache/pmetal/lib");
                let dest = cache_dir.join("mlx.metallib");
                let should_copy = if dest.exists() {
                    // Replace if the build artifact is newer
                    dest.metadata()
                        .and_then(|d| {
                            metallib.metadata().map(|s| {
                                s.modified().ok().zip(d.modified().ok()).is_some_and(
                                    |(src_t, dst_t)| src_t > dst_t,
                                )
                            })
                        })
                        .unwrap_or(false)
                } else {
                    true
                };
                if should_copy {
                    let _ = std::fs::create_dir_all(&cache_dir);
                    match std::fs::copy(&metallib, &dest) {
                        Ok(_) => println!(
                            "cargo:warning=Cached mlx.metallib to {}",
                            dest.display()
                        ),
                        Err(e) => println!(
                            "cargo:warning=Failed to cache mlx.metallib: {}",
                            e
                        ),
                    }
                }
            }
        }
    }
}

fn main() {
    build_and_link_mlx_c();

    // generate bindings
    let bindings = bindgen::Builder::default()
        .rust_target("1.73.0".parse().expect("rust-version"))
        .header("src/mlx-c/mlx/c/mlx.h")
        .header("src/mlx-c/mlx/c/linalg.h")
        .header("src/mlx-c/mlx/c/error.h")
        .header("src/mlx-c/mlx/c/transforms_impl.h")
        .clang_arg("-Isrc/mlx-c")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
