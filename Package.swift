// swift-tools-version:5.9

import PackageDescription

let cxxSettings: [PackageDescription.CXXSetting] = [
    .headerSearchPath("../include/"),
    .headerSearchPath("../fp16/include/"),
    .headerSearchPath("../simsimd/include/"),
    .define("USEARCH_USE_FP16LIB", to: "1"),
    .define("USEARCH_USE_SIMSIMD", to: "1"),
]

let package = Package(
    name: "USearch",
    products: [
        .library(
            name: "USearchObjective",
            targets: ["USearchObjective"]
        ),
        .library(
            name: "USearch",
            targets: ["USearch"]
        ),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "USearchObjective",
            path: "objc",
            sources: ["USearchObjective.mm", "../simsimd/c/lib.c"],
            cxxSettings: cxxSettings
        ),
        .target(
            name: "usearch_c",
            path: "c",
            sources: ["usearch.h", "lib.cpp"],
            publicHeadersPath: ".",
            cxxSettings: cxxSettings,
            swiftSettings: [
                .interoperabilityMode(.Cxx)
            ]
        ),
        .target(
            name: "USearch",
            dependencies: ["usearch_c"],
            path: "swift",
            exclude: ["README.md", "Test.swift"],
            sources: ["USearchIndex.swift", "USearchIndex+Sugar.swift", "Util.swift"],
            cxxSettings: cxxSettings,
            swiftSettings: [
                .interoperabilityMode(.Cxx)
            ]
        ),
        .testTarget(
            name: "USearchTestsSwift",
            dependencies: ["USearch"],
            path: "swift",
            sources: ["Test.swift"],
            swiftSettings: [
                .interoperabilityMode(.Cxx)
            ]
        ),
    ],
    cxxLanguageStandard: CXXLanguageStandard.cxx11
)
