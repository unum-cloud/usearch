fn main() {
    cxx_build::bridge("src/main.rs")
        .file("src/rust.cpp")
        .flag_if_supported("-std=c++11")
        .include("include")
        .include("src")
        .include("fp16/include")
        .include("simsimd/include")
        .compile("usearch");

    println!("cargo:rerun-if-changed=src/main.rs");
    println!("cargo:rerun-if-changed=src/rust.cpp");
    println!("cargo:rerun-if-changed=src/rust.hpp");
    println!("cargo:rerun-if-changed=src/advanced.hpp");
    println!("cargo:rerun-if-changed=include/usearch/usearch.hpp");
}
