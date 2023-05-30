fn main() {
    cxx_build::bridge("rust/lib.rs")
        .file("rust/lib.cpp")
        .flag_if_supported("-std=c++11")
        .flag_if_supported("-Wno-unknown-pragmas")
        .include("include")
        .include("rust")
        .include("src")
        .include("fp16/include")
        .include("robin-map/include")
        .include("simsimd/include")
        .compile("usearch");

    println!("cargo:rerun-if-changed=rust/lib.rs");
    println!("cargo:rerun-if-changed=rust/lib.cpp");
    println!("cargo:rerun-if-changed=rust/lib.hpp");
    println!("cargo:rerun-if-changed=src/punned.hpp");
    println!("cargo:rerun-if-changed=include/usearch/usearch.hpp");
}
