{
    "targets": [
        {
            "target_name": "usearch",
            "sources": ["javascript/lib.cpp"],
            "include_dirs": [
                "<!@(node -p \"require('node-addon-api').include\")",
                "include",
                "fp16/include",
                "simsimd/include",
            ],
            "dependencies": ["<!(node -p \"require('node-addon-api').gyp\")"],
            "defines": ["USEARCH_USE_SIMSIMD=1", "USEARCH_USE_NATIVE_F16=0"],
            "cflags": [
                "-fexceptions",
                "-Wno-unknown-pragmas",
                "-Wno-maybe-uninitialized",
            ],
            "cflags_cc": [
                "-fexceptions",
                "-Wno-unknown-pragmas",
                "-Wno-maybe-uninitialized",
                "-std=c++17",
            ],
            "xcode_settings": {
                "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
                "CLANG_CXX_LIBRARY": "libc++",
                "MACOSX_DEPLOYMENT_TARGET": "10.15",
            },
            "msvs_settings": {
                "VCCLCompilerTool": {
                    "ExceptionHandling": 1,
                    "AdditionalOptions": ["-std:c++17"],
                }
            },
        }
    ]
}
