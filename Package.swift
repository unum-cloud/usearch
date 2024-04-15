// swift-tools-version:5.3

import PackageDescription

let package = Package(
    name: "USearch",
    products: [
        .library(
            name: "USearch",
            targets: ["USearchObjective", "USearch"]
        )
    ],
    dependencies: [],
    targets: [
        .target(
            name: "USearchObjective",
            path: "objc",
            sources: ["USearchObjective.mm"],
            cxxSettings: [
                .headerSearchPath("../include/"),
                .headerSearchPath("../fp16/include/"),
                .headerSearchPath("../simismd/include/"),
            ]
        ),
        .target(
            name: "USearch",
            dependencies: ["USearchObjective"],
            path: "swift",
            exclude: ["README.md", "Test.swift"],
            sources: ["USearch.swift", "Index+Sugar.swift"]
        ),
        .testTarget(
            name: "USearchTests",
            dependencies: ["USearch"],
            path: "swift",
            exclude: ["USearch.swift", "Index+Sugar.swift", "README.md"],
            sources: ["Test.swift"]
        ),
    ],
    cxxLanguageStandard: CXXLanguageStandard.cxx11
)
