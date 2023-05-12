// swift-tools-version:5.3

import PackageDescription

let package = Package(
    name: "USearch",
    products: [
        .library(
            name: "USearch",
            targets: ["ObjectiveUSearch", "SwiftyUSearch"]
        )
    ],
    dependencies: [],
    targets: [
        .target(
            name: "ObjectiveUSearch",
            path: "objc",
            sources: ["ObjectiveUSearch.mm"],
            cxxSettings: [
                .headerSearchPath("../include/"),
                .headerSearchPath("../src/"),
                .headerSearchPath("../fp16/include/"),
                .headerSearchPath("../simismd/include/")
            ]
        ),
        .target(
            name: "SwiftyUSearch",
            dependencies: ["ObjectiveUSearch"],
            path: "swift",
            sources: ["SwiftyUSearch.swift", "Index+Sugar.swift"]
        ),
        .testTarget(
            name: "SwiftyUSearchTests",
            dependencies: ["SwiftyUSearch"],
            path: "swift",
            sources: ["Test.swift"]
        )
    ],
    cxxLanguageStandard: CXXLanguageStandard.cxx11
)
