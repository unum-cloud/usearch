fn main() {
    let mut build = cxx_build::bridge("rust/lib.rs");

    build
        .file("rust/lib.cpp")
        .flag_if_supported("-Wno-unknown-pragmas")
        .warnings(false)
        .include("include")
        .include("rust")
        .include("fp16/include")
        .include("simsimd/include");

    // Check for optional features
    if cfg!(feature = "openmp") {
        build.define("USEARCH_USE_OPENMP", "1");
    } else {
        build.define("USEARCH_USE_OPENMP", "0");
    }

    if cfg!(feature = "fp16lib") {
        build.define("USEARCH_USE_FP16LIB", "1");
    } else {
        build.define("USEARCH_USE_FP16LIB", "0");
    }

    if cfg!(feature = "simsimd") {
        build
            .define("USEARCH_USE_SIMSIMD", "1")
            .define("SIMSIMD_DYNAMIC_DISPATCH", "1")
            .define("SIMSIMD_NATIVE_F16", "0");
    } else {
        build.define("USEARCH_USE_SIMSIMD", "0");
    }

    // Conditional compilation depending on the target operating system.
    if cfg!(target_os = "linux") {
        build
            .flag_if_supported("-std=c++17")
            .flag_if_supported("-O3")
            .flag_if_supported("-ffast-math")
            .flag_if_supported("-fdiagnostics-color=always")
            .flag_if_supported("-g1"); // Simplify debugging
    } else if cfg!(target_os = "macos") {
        build
            .flag_if_supported("-mmacosx-version-min=10.15")
            .flag_if_supported("-std=c++17")
            .flag_if_supported("-O3")
            .flag_if_supported("-ffast-math")
            .flag_if_supported("-fcolor-diagnostics")
            .flag_if_supported("-g1"); // Simplify debugging
    } else if cfg!(target_os = "windows") {
        build
            .flag_if_supported("/std:c++17")
            .flag_if_supported("/O2")
            .flag_if_supported("/fp:fast")
            .flag_if_supported("/W1"); // Reduce warnings verbosity
    }

    let mut result = build.try_compile("usearch");
    if result.is_err() {
        print!("cargo:warning=Failed to compile with all SIMD backends...");

        let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
        let flags_to_try = match target_arch.as_str() {
            "arm" | "aarch64" => vec!["SIMSIMD_TARGET_NEON", "SIMSIMD_TARGET_SVE"],
            _ => vec![
                "SIMSIMD_TARGET_SAPPHIRE",
                "SIMSIMD_TARGET_ICE",
                "SIMSIMD_TARGET_SKYLAKE",
                "SIMSIMD_TARGET_HASWELL",
            ],
        };

        for flag in flags_to_try {
            build.define(flag, "0");
            result = build.try_compile("usearch");
            if result.is_err() {
                println!(
                    "cargo:warning=Failed to compile after disabling {}, trying next configuration...",
                    flag
                );
            } else {
                break;
            }
        }
    }

    // Ensure one build has been successful
    result.unwrap();

    println!("cargo:rerun-if-changed=rust/lib.rs");
    println!("cargo:rerun-if-changed=rust/lib.cpp");
    println!("cargo:rerun-if-changed=rust/lib.hpp");
    println!("cargo:rerun-if-changed=include/index_plugins.hpp");
    println!("cargo:rerun-if-changed=include/index_dense.hpp");
    println!("cargo:rerun-if-changed=include/usearch/index.hpp");
}
