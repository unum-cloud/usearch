fn main() {
    cxx_build::bridge("rust/lib.rs")
        .file("rust/lib.cpp")
        .flag_if_supported("-std=c++11")
        .flag_if_supported("-Wno-unknown-pragmas")
        .include("include")
        .include("rust")
        .include("fp16/include")
        .include("simsimd/include")
        .compile("usearch");

    println!("cargo:rerun-if-changed=rust/lib.rs");
    println!("cargo:rerun-if-changed=rust/lib.cpp");
    println!("cargo:rerun-if-changed=rust/lib.hpp");
    println!("cargo:rerun-if-changed=include/index_plugins.hpp");
    println!("cargo:rerun-if-changed=include/index_dense.hpp");
    println!("cargo:rerun-if-changed=include/usearch/index.hpp");
}
