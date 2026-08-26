// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "mlxfast-challenge-dev",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "mlxfast-swift", targets: ["MLXFastCLI"]),
        .executable(
            name: "mlxfast-runtime-worker",
            targets: ["MLXFastRuntimeWorkerCLI"]
        ),
        .library(name: "MLXFastCore", targets: ["MLXFastCore"]),
        .library(name: "MLXFastTransform", targets: ["MLXFastTransform"]),
        .library(name: "MLXFastModel", targets: ["MLXFastModel"]),
        .library(name: "MLXFastHarness", targets: ["MLXFastHarness"]),
    ],
    dependencies: [
        // Exact vendored revisions:
        // mlx-swift    df1fdc5f7821a1fabe921fdefbc42ac74dcfb6bc
        // mlx-swift-lm ed55bee  (Layr-Labs/mlx-swift-lm main, "Gemma 4 v0.8.2")
        //
        // mlx-swift-lm ADVANCED 2026-08-22 from bc1c0ee (2026-07-05, 65 commits
        // behind) to main. That adoption brought the production Gemma 4 MTP
        // layer and the CBv2 engine, and removed the v1 batching/compiled-decode
        // stack upstream deleted at ffede00 -- see docs/gemma4-port-notes.md.
        //
        // mlx-swift is UNCHANGED at df1fdc5f. Upstream main declares its
        // mlx-swift dependency as a floating `branch: "main"` URL; the vendored
        // copy is patched back to `.package(path: "../mlx-swift")` so the
        // kernel layer stays hard-frozen and out of Package.resolved. Both
        // vendored packages are path dependencies for that reason.
        .package(path: "Vendor/mlx-swift"),
        .package(path: "Vendor/mlx-swift-lm"),
        // The resolved dependency graph is frozen. In the engine repository
        // the enforcement that survives is the one that lives here: setup.sh
        // refuses to build over a Package.swift/Package.resolved that differs
        // from the committed state, and every build and resolve passes
        // --force-resolved-versions so SwiftPM fails closed instead of
        // silently re-resolving. The contest-side byte-verification of this
        // manifest against a trusted ref ran from .github/scripts/, which was
        // stripped with the rest of the contest layer; that check now belongs
        // to whatever pipeline grades a submission, not to this tree.
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.3.3"),
    ],
    targets: [
        .target(name: "MLXFastCore"),
        .target(
            name: "MLXFastTransform",
            dependencies: ["MLXFastCore"]
        ),
        .target(
            name: "MLXFastModel",
            dependencies: [
                "MLXFastCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                // DFlash block-decode track (laguna-xs-2.1-dflash-v1):
                // LagunaDFlashBlockSession wraps the vendored speculative round.
                // The serial track does not reach this code.
            ]
        ),
        .target(
            name: "MLXFastRuntimeWorkerSupport",
            dependencies: [
                "MLXFastCore",
                "MLXFastTransform",
                "MLXFastModel",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                // DFlash block-decode worker: loads the organizer-pinned target
                // through the vendored factory (LagunaModel is the type that
                // conforms to DFlashTargetModel) and the DFlash drafter.
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                // The REAL DFlash drafter (`DFlashDraftModel`) the gemma4
                // `dflash` arm binds from `dflash-head/`, plus the greedy
                // draft/verify/rollback round it runs. Vendored 2026-08-25;
                // the arm previously aliased the Gemma-4 MTP assistant-head
                // loader, which cannot decode a DFlash `config.json`.
                .product(name: "MLXSpeculative", package: "mlx-swift-lm"),
            ],
            path: "Sources/MLXFastHarness"
        ),
        // The trusted-harness source scope is this manifest,
        // Package.resolved, Sources/MLXFastCLI, Sources/MLXFastTrustedHarness
        // and Sources/MLXFastCore. It is declared here so a submission cannot
        // expand or repoint the targets feeding the trusted binary without
        // that showing up as a manifest diff. The byte-verification of this
        // scope against trusted git content ran from .github/scripts/, which
        // this engine repository no longer carries; enforcing it is the
        // grading pipeline's job.
        .target(
            name: "MLXFastHarness",
            dependencies: [
                "MLXFastCore",
                "MLXFastTransform",
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/MLXFastTrustedHarness",
            swiftSettings: [
                .define("MLXFAST_TRUSTED_HARNESS")
            ]
        ),
        .executableTarget(
            name: "MLXFastCLI",
            dependencies: [
                "MLXFastCore",
                "MLXFastTransform",
                "MLXFastHarness",
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "MLXFastRuntimeWorkerCLI",
            dependencies: [
                "MLXFastCore",
                "MLXFastModel",
                "MLXFastRuntimeWorkerSupport",
            ]
        ),
        .testTarget(
            name: "MLXFastTests",
            dependencies: [
                "MLXFastCore",
                "MLXFastTransform",
                "MLXFastModel",
                "MLXFastHarness",
                "MLXFastRuntimeWorkerSupport",
            ]
        ),
    ]
)
