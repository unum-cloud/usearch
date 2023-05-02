fn main() {
    cxx_build::bridge("rust/lib.rs")
        .file("rust/lib.cpp")
        .flag_if_supported("-std=c++11")
        .include("include")
        .include("rust")
        .include("src")
        .include("fp16/include")
        .include("simsimd/include")
        .compile("usearch");

    println!("cargo:rerun-if-changed=rust/lib.rs");
    println!("cargo:rerun-if-changed=rust/lib.cpp");
    println!("cargo:rerun-if-changed=rust/lib.hpp");
    println!("cargo:rerun-if-changed=src/advanced.hpp");
    println!("cargo:rerun-if-changed=include/usearch/usearch.hpp");
}
