{
    "targets": [
        {
            "target_name": "usearch",
            "sources": [
                "javascript/lib.cpp"
            ],
            "include_dirs": [
                "<!@(node -p \"require('node-addon-api').include\")",
                "include",
                "fp16/include",
                "simsimd/include"
            ],
            "dependencies": [
                "<!(node -p \"require('node-addon-api').gyp\")"
            ],
            "cflags!": ["-fno-exceptions", "-Wunknown-pragmas"],
            "cflags_cc!": ["-fno-exceptions", "-Wunknown-pragmas"],
            "conditions": [
                ["OS==\"mac\"", {
                    "xcode_settings": {
                        "GCC_ENABLE_CPP_EXCEPTIONS": "YES"
                    }
                }]
            ]
        }
    ]
}
