// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import CompilerPluginSupport
import PackageDescription

let package = Package(
    name: "mlx-swift-lm",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .tvOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        .library(
            name: "MLXLLM",
            targets: ["MLXLLM"]),
        .library(
            name: "MLXVLM",
            targets: ["MLXVLM"]),
        .library(
            name: "MLXLMCommon",
            targets: ["MLXLMCommon"]),
        // DFlash speculative decoding (z-lab drafter family). Vendored from
        // Layr-Labs/mlx-swift-lm `dflash-framework-updates` @ d41c3003 so the
        // engine's DFlash arm binds the REAL `DFlashDraftModel` instead of
        // aliasing the Gemma-4 MTP assistant head.
        .library(
            name: "MLXSpeculative",
            targets: ["MLXSpeculative"]),
        .library(
            name: "MLXEmbedders",
            targets: ["MLXEmbedders"]),
        .library(
            name: "MLXHuggingFace",
            targets: ["MLXHuggingFace"]),
        .library(
            name: "MLXLMServer",
            targets: ["MLXLMServer"]),
        .executable(
            name: "mlx-server",
            targets: ["mlx-server"]),
        .library(
            name: "BenchmarkHelpers",
            targets: ["BenchmarkHelpers"]),
        .library(
            name: "IntegrationTestHelpers",
            targets: ["IntegrationTestHelpers"]),
    ],
    dependencies: [
        // ENGINE-LOCAL: the engine consumes both vendored packages as PATH
        // dependencies, so neither appears in Package.resolved and both are
        // hard-frozen at the revisions recorded in the engine's own
        // Package.swift. Upstream main declares a floating `branch: "main"`
        // URL here; taking that verbatim would let the MLX kernel layer
        // re-resolve on every checkout, which is exactly what the engine's
        // --force-resolved-versions discipline exists to prevent.
        .package(path: "../mlx-swift"),
        .package(url: "https://github.com/swiftlang/swift-syntax.git", "600.0.0" ..< "604.0.0"),
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.23.0"),
        .package(url: "https://github.com/huggingface/swift-huggingface.git", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.3.2"),
    ],
    targets: [
        .target(
            name: "MLXLLM",
            dependencies: [
                "MLXLMCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ],
            path: "Libraries/MLXLLM",
            exclude: [
                "README.md"
            ]
        ),
        // DFlash drafter + greedy round. Depends on MLXLLM for the
        // `DFlashTargetModel` seam (the protocol lives there so model
        // implementations can conform without reversing the package
        // dependency direction — see DFlashTarget.swift).
        .target(
            name: "MLXSpeculative",
            dependencies: [
                "MLXLLM",
                "MLXLMCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Libraries/MLXSpeculative"
        ),
        .target(
            name: "MLXVLM",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ],
            path: "Libraries/MLXVLM",
            exclude: [
                "README.md"
            ]
        ),
        .target(
            name: "MLXLMCommon",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                // Cmlx: low-level mlx_slice_dynamic / mlx_slice_update_dynamic
                // entry points used by DynamicSlice.swift (compiled-decode infra).
                .product(name: "Cmlx", package: "mlx-swift"),
            ],
            path: "Libraries/MLXLMCommon",
            exclude: [
                "README.md"
            ],
            resources: [
                // CBv2 paged-attention MSL source, JIT-compiled at runtime
                // via MLXFast.metalKernel (NOT compiled by SwiftPM).
                .copy("ContinuousBatchingV2/Paged/pagedattention.metal")
            ]
        ),
        .target(
            name: "MLXEmbedders",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .target(name: "MLXLMCommon"),
            ],
            path: "Libraries/MLXEmbedders",
            exclude: [
                "README.md"
            ]
        ),
        .target(
            name: "MLXLMServer",
            dependencies: [
                "MLXLLM",
                "MLXVLM",
                "MLXLMCommon",
                "MLXEmbedders",
                "MLXHuggingFace",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Libraries/MLXLMServer"
        ),
        .executableTarget(
            name: "mlx-server",
            dependencies: ["MLXLMServer"],
            path: "Executables/mlx-server"
        ),
        .target(
            name: "BenchmarkHelpers",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "MLXEmbedders",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Libraries/BenchmarkHelpers"
        ),
        .target(
            name: "IntegrationTestHelpers",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "MLXEmbedders",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Libraries/IntegrationTestHelpers",
            exclude: ["README.md"]
        ),
        .testTarget(
            name: "MLXLMTests",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "MLXEmbedders",
            ],
            path: "Tests/MLXLMTests",
            exclude: [
                "README.md",
                // Stale VLM MTP spike: references Gemma4 (MLXVLM) MTP API
                // removed by the vMLX decode port (#55). Broken at the
                // engine-v2 branch base and blocks the whole test target;
                // excluded until the VLM MTP spike is updated or deleted.
                "Gemma4VLMMTPSpikeTests.swift",
            ],
            resources: [
                .process("Resources/1080p_30.mov"),
                .process("Resources/audio_only.mov"),
                .process("Resources/Gemma4MTPPrompts.json"),
                .process("Resources/gemma4-26B-A4B-assistant-config.json"),
                .process("Resources/gemma4-E4B-assistant-config.json"),
                .process("Resources/mtp-oracle/gemma4-e2b-block3-max64.json"),
                .process("Resources/block_hash_vectors.json"),
            ]
        ),
        .testTarget(
            name: "MLXLMServerTests",
            dependencies: [
                "MLXLMServer",
                "MLXLMCommon",
                .product(name: "HummingbirdTesting", package: "hummingbird"),
            ],
            path: "Tests/MLXLMServerTests"
        ),
        .macro(
            name: "MLXHuggingFaceMacros",
            dependencies: [
                .product(name: "SwiftSyntaxMacros", package: "swift-syntax"),
                .product(name: "SwiftCompilerPlugin", package: "swift-syntax"),
            ],
            path: "Libraries/MLXHuggingFaceMacros"
        ),
        .target(
            name: "MLXHuggingFace",
            dependencies: [
                "MLXHuggingFaceMacros",
                "MLXLMCommon",
            ],
            path: "Libraries/MLXHuggingFace"
        ),
        .executableTarget(
            name: "BenchLoad",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "BenchmarkHelpers",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/BenchLoad"
        ),
        // WS-G (engine-v2) benchmark harness: tiny-model v2-style step loop
        // (CI-runnable) + `--model <path>` legacy-engine baseline for real
        // weights. Reports land as markdown next to the other benchmarks/
        // reports. `sources:` keeps the target scoped to the one Swift file.
        .executableTarget(
            name: "CBv2Benchmark",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "benchmarks",
            exclude: [
                "mtp-2026-05-09-130614.md",
                "mtp-2026-05-09-142113.md",
                "mtp-2026-05-09-183133.md",
                "mtp-2026-05-09-190948.md",
                "mtp-2026-05-09-203007.md",
            ],
            sources: ["CBv2Benchmark.swift"]
        ),
        // Build-time git-revision stamp for BenchCBv2. The bench report must
        // name the revision the binary was compiled from, not whatever HEAD
        // is when the report is written. Attached to BenchCBv2Core, which is
        // where `buildRevision()` reads the generated constant.
        .plugin(
            name: "BenchRevisionStamp",
            capability: .buildTool(),
            path: "Plugins/BenchRevisionStamp"
        ),
        // Real-weights validation driver for the CBv2 engine: correctness
        // smoke (batch-composition + chunked-prefill invariance) and the
        // v2-vs-v2-paged perf matrix on local model directories.
        //
        // A LIBRARY, with `BenchCBv2` below reduced to a `@main` shim over
        // `BenchCBv2Driver.run()`. The split is load-bearing: when a test
        // target depends on an executable target, SwiftPM runs that binary as
        // the swift-testing host and hands it `--test-bundle-path`, which this
        // driver's strict option parser rejects — aborting the swift-testing
        // pass for the whole package. Every `@Test` in mlx-swift-lm, including
        // the paged-KV CI gates, executed nothing while BenchCBv2Tests
        // depended on the executable. Nothing may depend on `BenchCBv2`.
        .target(
            name: "BenchCBv2Core",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                "MLXHuggingFace",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/BenchCBv2Core",
            plugins: ["BenchRevisionStamp"]
        ),
        .executableTarget(
            name: "BenchCBv2",
            dependencies: ["BenchCBv2Core"],
            path: "Sources/BenchCBv2"
        ),
        // Harness-integrity tests: option parsing, engine resolution, and
        // report/optimization provenance. Model-free, so they run in CI.
        .testTarget(
            name: "BenchCBv2Tests",
            dependencies: ["BenchCBv2Core"],
            path: "Tests/BenchCBv2Tests"
        ),
    ]
)

if Context.environment["MLX_SWIFT_BUILD_DOC"] == "1"
    || Context.environment["SPI_GENERATE_DOCS"] == "1"
{
    package.dependencies.append(
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.3.0")
    )
}
