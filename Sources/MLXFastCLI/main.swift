import Darwin
import Foundation
import MLXFastCore
import MLXFastHarness
import MLXFastTransform
import Tokenizers

let exitCode = MLXFastCLI.run(arguments: Array(CommandLine.arguments.dropFirst()))
exit(Int32(exitCode))

private enum MLXFastCLI {
    static func run(arguments: [String]) -> Int {
        guard let command = arguments.first, command != "help", command != "--help", command != "-h" else {
            printUsage()
            return 0
        }

        let options = ParsedOptions(Array(arguments.dropFirst()))

        do {
            switch command {
            case "transform":
                try runTransform(options)
                return 0
            case "verify-transform":
                try runVerifyTransform(options)
                return 0
            case "correctness":
                return try runCorrectness(options)
            case "correctness-trace":
                try runCorrectnessTrace(options)
                return 0
            case "preflight":
                try runPreflight(options)
                return 0
            case "benchmark":
                try runBenchmark(options)
                return 0
            case "attach-gpqa-gates":
                try runAttachGPQAGates(options)
                return 0
            case "attach-free-run-gate":
                try runAttachFreeRunGate(options)
                return 0
            case "attach-benchmark-oracle":
                try runAttachBenchmarkOracle(options)
                return 0
            case "generate-golden":
                try runGenerateGolden(options)
                return 0
            case "record-reference-tape":
                try runRecordReferenceTape(options)
                return 0
            case "analyze-ngram-similarity":
                try runAnalyzeNGramSimilarity(options)
                return 0
            case "generate-gpqa-answers":
                try runGenerateGPQAAnswers(options)
                return 0
            case "checkpoint-shards":
                try runCheckpointShards(options)
                return 0
            case "dflash-benchmark":
                try runDFlashBenchmark(options)
                return 0
            case "dflash-probe":
                // Serial K=1 control: the denominator of the paired score.
                try runDFlashBenchmark(options, serialControl: true)
                return 0
            case "mtp-verify":
                // MTP ARM DEFERRED (2026-08-22, Gemma 4 26B A4B harness port).
                // The verb is RETAINED and REFUSES rather than being deleted,
                // for the same reason `dflash-runtime-worker` is: the arm is
                // deferred, not abandoned, so keeping the name reserved means
                // the follow-up increment restores behaviour behind an entry
                // point that already exists, and any caller still invoking it
                // is told what happened instead of "unknown command".
                //
                // What went: the whole MTP surface was written against the
                // Qwen tower -- `Qwen36MTPTarget` conforms `Qwen35TextModel`
                // and `MLXLLM.Qwen35Model` and nothing else, and the head
                // attachment merges a Qwen head into a Qwen backbone. Renaming
                // the model types would have left code that compiles and
                // cannot run, which is worse than a refusal.
                throw MLXFastError.invalidInput(
                    "mtp-verify is not runnable on this engine: the MTP arm "
                        + "lands with the Gemma harness port's follow-up "
                        + "increment. See docs/gemma4-port-notes.md section 8."
                )
            case "dflash-reference":
                try runDFlashReference(options)
                return 0
            default:
                fputs("mlxfast-swift: unknown command '\(command)'\n\n", stderr)
                printUsage()
                return 2
            }
        } catch {
            fputs("mlxfast-swift: \(error)\n", stderr)
            return 1
        }
    }

    private static func runTransform(_ options: ParsedOptions) throws {
        try reexecUnderParentToolSandboxIfRequested(subcommand: "transform")
        try options.validate(valueOptions: ["--reference", "--output"])
        let referencePath = options.value(
            for: "--reference",
            default: environmentValue(
                "MLXFAST_REFERENCE_DIR",
                fallback: MLXFastConstants.defaultReferencePath
            )
        )
        let outputPath = options.value(
            for: "--output",
            default: environmentValue(
                "MLXFAST_WEIGHTS_PATH",
                fallback: MLXFastConstants.defaultWeightsPath
            )
        )
        let report = try SwiftTransform.run(
            TransformOptions(referencePath: referencePath, outputPath: outputPath)
        )
        print("reference: \(report.referencePath)")
        print("output: \(report.outputPath)")
        print("dense tensors: \(report.denseTensorCount) across \(report.denseShardCount) shard(s)")
        print("config: \(report.configPath)")
        print("index: \(report.indexPath)")
    }

    private static func runVerifyTransform(_ options: ParsedOptions) throws {
        try options.validate(valueOptions: ["--reference", "--weights", "--tmp-parent", "--max-bytes"])
        let referencePath = options.value(
            for: "--reference",
            default: environmentValue(
                "MLXFAST_REFERENCE_DIR",
                fallback: MLXFastConstants.defaultReferencePath
            )
        )
        let weightsPath = options.value(
            for: "--weights",
            default: environmentValue(
                "MLXFAST_WEIGHTS_PATH",
                fallback: MLXFastConstants.defaultWeightsPath
            )
        )
        let temporaryParentPath = options.value(for: "--tmp-parent", default: "")
        let maxBytesRaw = options.value(
            for: "--max-bytes",
            default: environmentValue(
                "MLXFAST_MAX_WEIGHTS_BYTES",
                fallback: "\(MLXFastConstants.defaultMaxTransformedWeightsBytes)"
            )
        )
        let maxByteCount = try parseTransformedWeightsByteLimit(
            raw: maxBytesRaw,
            defaultByteCount: MLXFastConstants.defaultMaxTransformedWeightsBytes,
            optionLabel: "--max-bytes"
        )
        let report = try TransformVerifier.verify(
            TransformVerificationOptions(
                referencePath: referencePath,
                weightsPath: weightsPath,
                temporaryParentPath: temporaryParentPath.isEmpty ? nil : temporaryParentPath,
                maxByteCount: maxByteCount
            )
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(report)
        FileHandle.standardOutput.write(data)
        print("")
    }

    private static func runCorrectness(_ options: ParsedOptions) throws -> Int {
        try options.validate(valueOptions: ["--weights", "--golden"])
        let weightsPath = options.value(
            for: "--weights",
            default: environmentValue(
                "MLXFAST_WEIGHTS_PATH",
                fallback: MLXFastConstants.defaultWeightsPath
            )
        )
        let goldenPath = options.value(
            for: "--golden",
            default: environmentValue(
                "MLXFAST_CORRECTNESS_GOLDEN_PATH",
                fallback: defaultCorrectnessGoldenPath()
            )
        )
        let report = try Gemma4Runtime.runCorrectness(
            CorrectnessOptions(
                weightsPath: weightsPath,
                goldenPath: goldenPath
            ),
            worker: try runtimeWorkerOptions(blockedGoldenPath: goldenPath)
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(report)
        FileHandle.standardOutput.write(data)
        print("")
        if !report.passed, report.error.contains("token mismatch") {
            fputs("mlxfast-swift: \(Gemma4Runtime.nonM5GoldenMismatchCaveat)\n", stderr)
        }
        return report.passed ? 0 : 1
    }

    private static func runCorrectnessTrace(_ options: ParsedOptions) throws {
        try options.validate(valueOptions: ["--weights", "--golden", "--case", "--step", "--top-k"])
        let weightsPath = options.value(
            for: "--weights",
            default: environmentValue(
                "MLXFAST_WEIGHTS_PATH",
                fallback: MLXFastConstants.defaultWeightsPath
            )
        )
        let goldenPath = options.value(
            for: "--golden",
            default: environmentValue(
                "MLXFAST_CORRECTNESS_GOLDEN_PATH",
                fallback: defaultCorrectnessGoldenPath()
            )
        )
        let stepRaw = options.value(for: "--step", default: "")
        guard let step = Int(stepRaw), step >= 0 else {
            throw MLXFastError.invalidInput("correctness-trace requires --step N with N >= 0")
        }
        let topKRaw = options.value(for: "--top-k", default: "8")
        guard let topK = Int(topKRaw), topK > 0 else {
            throw MLXFastError.invalidInput("--top-k must be a positive integer")
        }
        let caseName = options.value(for: "--case", default: "")
        let report = try Gemma4Runtime.traceCorrectness(
            CorrectnessTraceOptions(
                weightsPath: weightsPath,
                goldenPath: goldenPath,
                caseName: caseName.isEmpty ? nil : caseName,
                step: step,
                topK: topK
            ),
            worker: try runtimeWorkerOptions(
                blockedGoldenPath: goldenPath
            )
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(report)
        FileHandle.standardOutput.write(data)
        print("")
    }

    private static func runPreflight(_ options: ParsedOptions) throws {
        try options.validate(valueOptions: ["--weights", "--golden"])
        let weightsPath = options.value(
            for: "--weights",
            default: environmentValue(
                "MLXFAST_WEIGHTS_PATH",
                fallback: MLXFastConstants.defaultWeightsPath
            )
        )
        let goldenPath = options.value(
            for: "--golden",
            default: environmentValue(
                "MLXFAST_CORRECTNESS_GOLDEN_PATH",
                fallback: MLXFastConstants.defaultGoldenPath
            )
        )
        let report = try BenchmarkPreflight.check(
            weightsPath: weightsPath,
            goldenPath: goldenPath
        )
        guard let worker = try runtimeWorkerOptions(
            blockedGoldenPath: goldenPath
        ) else {
            throw MLXFastError.invalidInput(
                "preflight requires the participant runtime worker"
            )
        }
        try Gemma4Runtime.runPreflightWithWorker(
            weightsPath: weightsPath,
            worker: worker
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(report)
        FileHandle.standardOutput.write(data)
        print("")
    }

    private static func runBenchmark(_ options: ParsedOptions) throws {
        try options.validate(
            valueOptions: ["--weights", "--golden", "--score-path"],
            flagOptions: ["--local-submit", "--local-iterate"]
        )
        let localSubmit = options.hasFlag("--local-submit")
        let localIterate = options.hasFlag("--local-iterate")
        guard !(localSubmit && localIterate) else {
            throw MLXFastError.invalidInput("--local-submit and --local-iterate cannot be used together")
        }
        let weightsPath = options.value(
            for: "--weights",
            default: environmentValue(
                "MLXFAST_WEIGHTS_PATH",
                fallback: MLXFastConstants.defaultWeightsPath
            )
        )
        let goldenPath = options.value(
            for: "--golden",
            default: environmentValue(
                "MLXFAST_CORRECTNESS_GOLDEN_PATH",
                fallback: localSubmit
                    ? MLXFastConstants.defaultPublicLocalSubmitGoldenPath
                    : localIterate
                        ? MLXFastConstants.defaultPublicCorrectnessGoldenPath
                        : MLXFastConstants.defaultGoldenPath
            )
        )
        let scorePath = options.value(
            for: "--score-path",
            default: environmentValue(
                "MLXFAST_SCORE_PATH",
                fallback: localIterate
                    ? MLXFastConstants.defaultLocalIterateScorePath
                    : MLXFastConstants.defaultScorePath
            )
        )
        if localSubmit || localIterate {
            let decodeSteps = localSubmit
                ? MLXFastConstants.localSubmitBenchmarkDecodeSteps
                : MLXFastConstants.localIterateBenchmarkDecodeSteps
            let timingRepeats = localSubmit ? MLXFastConstants.localSubmitBenchmarkRepeats : 1
            let modeName = localSubmit ? "local-submit" : "local-iterate"
            let runtime = localSubmit ? "swift-local-submit" : "swift-local-iterate"
            let payload = Gemma4Runtime.localIterate(
                LocalIterateOptions(
                    weightsPath: weightsPath,
                    goldenPath: goldenPath,
                    benchmarkDecodeSteps: decodeSteps,
                    timingRepeats: timingRepeats,
                    modeName: modeName,
                    runtime: runtime
                ),
                // Local edit loop: stream the worker's stderr live so debug
                // prints in submitted model code are visible while iterating.
                // runtimeWorkerOptions forces this off on official runs.
                worker: try runtimeWorkerOptions(
                    blockedGoldenPath: goldenPath,
                    forwardsWorkerStderr: true
                )
            )
            try writeScorePayload(payload, to: scorePath)
            try emitScorePayloadToStdout(payload)
            return
        }
        let semanticOutputPath = environmentValue("MLXFAST_SEMANTIC_GPQA_OUTPUT_PATH", fallback: "")
        let semanticCaseCount = try parsePositiveInt(
            environmentValue(
                "MLXFAST_SEMANTIC_GPQA_CASE_COUNT",
                fallback: "\(MLXFastConstants.semanticGPQACaseCount)"
            ),
            optionName: "MLXFAST_SEMANTIC_GPQA_CASE_COUNT"
        )
        let semanticMaxNewTokens = try parsePositiveInt(
            environmentValue(
                "MLXFAST_SEMANTIC_GPQA_MAX_NEW_TOKENS",
                fallback: "\(MLXFastConstants.semanticGPQAMaxNewTokens)"
            ),
            optionName: "MLXFAST_SEMANTIC_GPQA_MAX_NEW_TOKENS"
        )
        if !semanticOutputPath.isEmpty {
            try requirePrivateOutputPath(semanticOutputPath, description: "semantic GPQA answer output")
        }
        // Lets a run skip the base teacher-forced case (still runs behavior/GPQA/TTFT/
        // timing) when that case is verified in a separate phase of the ranked
        // pipeline. Defaults to the full official window. See the comment on
        // BenchmarkPreflight/validateBenchmarkOptions for why 0 is accepted: the
        // harness never treats a steps=0 run as self-certifying correctness; only the
        // trusted pipeline that assembles the final score may do that.
        let correctnessSteps = try parseNonNegativeInt(
            environmentValue(
                "MLXFAST_BENCHMARK_CORRECTNESS_STEPS",
                fallback: "\(MLXFastConstants.correctnessSteps)"
            ),
            optionName: "MLXFAST_BENCHMARK_CORRECTNESS_STEPS"
        )
        // Phase controls. The SERIAL timed benchmark is retired: DFlash is the
        // default track (benchmark.json), its timed score comes from
        // measure-dflash-job.sh, and this `mlxfast-swift benchmark` command exists
        // now only to drive the SHARED correctness/GPQA gates (the teacher-forced
        // base case that proves the model is still sound, track-independent).
        // So SKIP_TIMED defaults to "1": a bare invocation runs gates only and
        // never the serial timed phase. checkGates defaults on, so the pair
        // (checkGates=true, skipTimed=true) satisfies the "check or time
        // something" guard. To deliberately run the retired serial timing, set
        // MLXFAST_BENCHMARK_SKIP_TIMED=0 explicitly. The DFlash gates step already
        // passes CHECK_GATES=1 SKIP_TIMED=1 and is unaffected.
        let checkGates = environmentValue("MLXFAST_BENCHMARK_CHECK_GATES", fallback: "1") != "0"
        let skipTimedBenchmark = environmentValue("MLXFAST_BENCHMARK_SKIP_TIMED", fallback: "1") == "1"
        let payload = Gemma4Runtime.benchmark(
            BenchmarkOptions(
                weightsPath: weightsPath,
                goldenPath: goldenPath,
                correctnessSteps: correctnessSteps,
                semanticGPQAOutputPath: semanticOutputPath.isEmpty ? nil : semanticOutputPath,
                semanticGPQATokenizerPath: weightsPath,
                semanticGPQACaseCount: semanticCaseCount,
                semanticGPQAMaxNewTokens: semanticMaxNewTokens,
                checkGates: checkGates,
                skipTimedBenchmark: skipTimedBenchmark
            ),
            worker: try runtimeWorkerOptions(blockedGoldenPath: goldenPath)
        )
        try writeScorePayload(payload, to: scorePath)
        try emitScorePayloadToStdout(payload)
        fputs("wrote \(scorePath)\n", stderr)
    }

    // Emits the in-memory payload, not a re-read of scorePath: the benchmark
    // process links the editable submission modules and runs unsandboxed, so a
    // file it wrote to scorePath could be tampered with (e.g. via an atexit
    // handler) between the write above and this call. Serializing the value
    // already held in memory means stdout reflects exactly what this trusted
    // process computed, independent of anything written to disk afterward.
    // benchmark.sh captures this stdout, after the process has fully exited, as
    // the sole source of truth for score.json.
    private static func emitScorePayloadToStdout(_ payload: ScorePayload) throws {
        // benchmark.sh seals score.json from THIS stdout, so it -- not the
        // writeScorePayload file it discards -- is the published per-machine
        // artifact. Coarsen the diagnostic analog fields here too, or the
        // timing/memory covert-channel coarsening applied in writeScorePayload
        // (and the combined score) is bypassed on the sealed path.
        let publishedPayload = ScorePayload(
            score: payload.score,
            passed: payload.passed,
            metrics: payload.metrics.withCoarsenedPublicDiagnostics()
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(publishedPayload)
        FileHandle.standardOutput.write(data)
        if data.last != 0x0a { print("") }
    }

    private static func runAttachGPQAGates(_ options: ParsedOptions) throws {
        try reexecUnderParentToolSandboxIfRequested(subcommand: "attach-gpqa-gates")
        try options.validate(
            valueOptions: ["--golden", "--gpqa", "--tokenizer", "--output", "--case-count", "--max-new-tokens"]
        )
        let goldenPath = options.value(
            for: "--golden",
            default: environmentValue(
                "MLXFAST_CORRECTNESS_GOLDEN_PATH",
                fallback: MLXFastConstants.defaultGoldenPath
            )
        )
        let gpqaPath = options.value(
            for: "--gpqa",
            default: environmentValue("MLXFAST_GPQA_REFERENCE_PATH", fallback: "")
        )
        guard !gpqaPath.isEmpty else {
            throw MLXFastError.invalidInput("attach-gpqa-gates requires --gpqa or MLXFAST_GPQA_REFERENCE_PATH")
        }
        let tokenizerPath = options.value(
            for: "--tokenizer",
            default: environmentValue("MLXFAST_TOKENIZER_PATH", fallback: MLXFastConstants.defaultWeightsPath)
        )
        let outputPath = options.value(for: "--output", default: goldenPath)
        let caseCount = try parsePositiveInt(
            options.value(for: "--case-count", default: "\(MLXFastConstants.correctnessGPQACaseCount)"),
            optionName: "--case-count"
        )
        let maxNewTokens = try parsePositiveInt(
            options.value(for: "--max-new-tokens", default: "\(MLXFastConstants.correctnessGPQAMaxNewTokens)"),
            optionName: "--max-new-tokens"
        )
        guard maxNewTokens <= MLXFastConstants.correctnessMaxBehaviorSteps else {
            throw MLXFastError.invalidInput(
                "--max-new-tokens must be <= \(MLXFastConstants.correctnessMaxBehaviorSteps)"
            )
        }

        try requireFile(goldenPath, description: "correctness golden file")
        try requireFile(gpqaPath, description: "GPQA reference cases file")
        try requireFile(
            URL(fileURLWithPath: tokenizerPath).appendingPathComponent("tokenizer.json").path,
            description: "tokenizer.json"
        )
        try requireFile(
            URL(fileURLWithPath: tokenizerPath).appendingPathComponent("tokenizer_config.json").path,
            description: "tokenizer_config.json"
        )

        let tokenizer = try loadLocalTokenizer(at: tokenizerPath)
        let goldenData = try Data(contentsOf: URL(fileURLWithPath: goldenPath))
        let golden = try JSONDecoder().decode(GoldenDocument.self, from: goldenData)
        let gpqaData = try Data(contentsOf: URL(fileURLWithPath: gpqaPath))
        let gpqa = try JSONDecoder().decode(GPQAReferenceDocument.self, from: gpqaData)
        var behaviorCases: [GoldenBehaviorCase] = []
        var skippedOverBudgetGPQACases = 0
        for testCase in gpqa.cases {
            guard behaviorCases.count < caseCount else {
                break
            }
            if let behaviorCase = try buildGPQABehaviorCaseIfWithinPromptBudget(
                testCase,
                tokenizer: tokenizer,
                maxNewTokens: maxNewTokens
            ) {
                behaviorCases.append(behaviorCase)
            } else {
                skippedOverBudgetGPQACases += 1
            }
        }
        guard behaviorCases.count == caseCount else {
            throw MLXFastError.invalidInput(
                "GPQA reference produced \(behaviorCases.count) token-budget-valid cases; "
                    + "need \(caseCount); skipped_over_budget=\(skippedOverBudgetGPQACases); "
                    + "max_prompt_tokens=\(MLXFastConstants.correctnessMaxBehaviorPromptTokens)"
            )
        }

        let existingGates = golden.correctnessGates
        let existingBehavior = existingGates?.behaviorCases ?? []
        let mergedGates = GoldenCorrectnessGates(
            anchors: existingGates?.anchors,
            freeRun: existingGates?.freeRun,
            behavior: existingBehavior + behaviorCases
        )
        let merged = GoldenDocument(
            version: golden.version ?? 1,
            modelType: golden.modelType,
            modelProvenance: golden.modelProvenance,
            cases: golden.cases,
            correctnessGates: mergedGates,
            benchmark: golden.benchmark
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try writeValidatedGoldenDocument(encoder.encode(merged), to: outputPath)
        print(
            "attached GPQA behavior gates cases=\(behaviorCases.count) "
                + "max_new_tokens=\(maxNewTokens) "
                + "skipped_over_budget=\(skippedOverBudgetGPQACases) "
                + "output=\(outputPath)"
        )
    }

    // Operator tool: attach a free-run gate whose greedy continuation covers the
    // timed decode offset range. The 64-step teacher-forced base case only
    // exercises single-token forwards at offsets 512..575, while the timed
    // decode reaches 512..639 -- a submission could special-case a cheaper model
    // path for offsets only the (identifiable) timing worker ever visits and no
    // structural gate would notice. A 512-token free-run case with >= 128
    // generated-and-checked steps makes the unscored correctness gate exercise
    // every timed decode offset with different prompt content, so an
    // offset-gated fast path has to survive correctness too. Run this offline
    // against the baseline reference weights, then upload the regenerated
    // golden through the organizer process (docs/private-benchmark-security.md).
    private static func runAttachFreeRunGate(_ options: ParsedOptions) throws {
        try options.validate(
            valueOptions: [
                "--golden", "--weights", "--output", "--name", "--steps",
                "--case", "--prompt-file", "--tokenizer", "--exact-prefix",
            ],
            flagOptions: ["--allow-partial"]
        )
        let goldenPath = options.value(
            for: "--golden",
            default: environmentValue(
                "MLXFAST_CORRECTNESS_GOLDEN_PATH",
                fallback: MLXFastConstants.defaultGoldenPath
            )
        )
        let weightsPath = options.value(
            for: "--weights",
            default: environmentValue("MLXFAST_WEIGHTS_PATH", fallback: MLXFastConstants.defaultWeightsPath)
        )
        let outputPath = options.value(for: "--output", default: goldenPath)
        let caseName = options.value(for: "--name", default: "free-run-decode-offset-coverage")
        let steps = try parsePositiveInt(
            options.value(for: "--steps", default: "\(MLXFastConstants.benchmarkDecodeSteps)"),
            optionName: "--steps"
        )
        guard steps <= MLXFastConstants.correctnessMaxFreeRunSteps else {
            throw MLXFastError.invalidInput(
                "--steps must be <= \(MLXFastConstants.correctnessMaxFreeRunSteps)"
            )
        }
        if steps < MLXFastConstants.benchmarkDecodeSteps {
            // The command exists to cover the timed decode offsets; a partial
            // gate silently leaves the specialization gap open, so fail closed
            // unless the operator explicitly opts in (debugging, staged runs).
            guard options.hasFlag("--allow-partial") else {
                throw MLXFastError.invalidInput(
                    "--steps \(steps) is below benchmarkDecodeSteps "
                        + "\(MLXFastConstants.benchmarkDecodeSteps), so the gate would not cover "
                        + "the full timed decode offset range; pass --allow-partial to write it anyway"
                )
            }
            fputs(
                "attach-free-run-gate: warning: --steps \(steps) is below "
                    + "benchmarkDecodeSteps \(MLXFastConstants.benchmarkDecodeSteps); "
                    + "the gate will NOT cover the full timed decode offset range (--allow-partial)\n",
                stderr
            )
        }
        let exactPrefixRaw = options.value(for: "--exact-prefix", default: "")
        var exactPrefixTokens: Int?
        if !exactPrefixRaw.isEmpty {
            let parsed = try parsePositiveInt(exactPrefixRaw, optionName: "--exact-prefix")
            guard parsed <= steps else {
                throw MLXFastError.invalidInput("--exact-prefix must be <= --steps (\(steps))")
            }
            exactPrefixTokens = parsed
        }

        try requireFile(goldenPath, description: "correctness golden file")
        try requireFile(
            URL(fileURLWithPath: weightsPath).appendingPathComponent("config.json").path,
            description: "weights config.json"
        )
        // Strict-validate the INPUT before any generation or write. --output
        // defaults to the input path, so a malformed input must fail here --
        // never after the original has been replaced on disk. Through the Qwen
        // loader: this verb generates continuation tokens ON QWEN WEIGHTS, so a
        // golden that names another model must not reach the generator.
        _ = try loadQwenGoldenFixture(from: goldenPath)
        let goldenData = try Data(contentsOf: URL(fileURLWithPath: goldenPath))
        let golden = try JSONDecoder().decode(GoldenDocument.self, from: goldenData)

        let requiredPromptTokens = MLXFastConstants.correctnessPromptTokens
        let promptTokens: [Int]
        let promptFile = options.value(for: "--prompt-file", default: "")
        let sourceCaseName = options.value(for: "--case", default: "")
        if !promptFile.isEmpty {
            let tokenizerPath = options.value(for: "--tokenizer", default: weightsPath)
            try requireFile(
                URL(fileURLWithPath: tokenizerPath).appendingPathComponent("tokenizer.json").path,
                description: "tokenizer.json"
            )
            let tokenizer = try loadLocalTokenizer(at: tokenizerPath)
            let promptText = try String(contentsOfFile: promptFile, encoding: .utf8)
            let encoded = tokenizer.encode(text: promptText, addSpecialTokens: false)
            guard encoded.count >= requiredPromptTokens else {
                throw MLXFastError.invalidInput(
                    "--prompt-file tokenized to \(encoded.count) tokens; free-run gates need at least \(requiredPromptTokens)"
                )
            }
            promptTokens = Array(encoded.prefix(requiredPromptTokens))
        } else if !sourceCaseName.isEmpty {
            guard let sourceCase = golden.cases.first(where: { $0.name == sourceCaseName }) else {
                throw MLXFastError.invalidInput("golden does not contain base case \(sourceCaseName)")
            }
            promptTokens = sourceCase.promptTokens
        } else {
            guard let firstCase = golden.cases.first else {
                throw MLXFastError.invalidInput("golden contains no base cases to source a prompt from")
            }
            promptTokens = firstCase.promptTokens
        }
        guard promptTokens.count == requiredPromptTokens else {
            throw MLXFastError.invalidInput(
                "free-run prompt has \(promptTokens.count) tokens; need exactly \(requiredPromptTokens)"
            )
        }

        fputs(
            "attach-free-run-gate: generating \(steps) reference continuation tokens "
                + "(covers decode offsets \(promptTokens.count)..<\(promptTokens.count + steps))\n",
            stderr
        )
        let expectedTokens = try Gemma4Runtime.generateGreedyTokens(
            GreedyGenerationOptions(
                weightsPath: weightsPath,
                promptTokens: promptTokens,
                steps: steps
            ),
            worker: try runtimeWorkerOptions(blockedGoldenPath: goldenPath)
        )

        let freeRunCase = GoldenFreeRunCase(
            name: caseName,
            promptTokens: promptTokens,
            expectedTokens: expectedTokens,
            exactPrefixTokens: exactPrefixTokens
        )
        let existingGates = golden.correctnessGates
        let mergedGates = GoldenCorrectnessGates(
            anchors: existingGates?.anchors,
            freeRun: (existingGates?.freeRunCases ?? []) + [freeRunCase],
            behavior: existingGates?.behavior
        )
        let merged = GoldenDocument(
            version: golden.version ?? 1,
            modelType: golden.modelType,
            modelProvenance: golden.modelProvenance,
            cases: golden.cases,
            correctnessGates: mergedGates,
            benchmark: golden.benchmark
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try writeValidatedGoldenDocument(encoder.encode(merged), to: outputPath)
        print(
            "attached free-run gate name=\(caseName) steps=\(steps) "
                + "decode_offsets=\(promptTokens.count)..<\(promptTokens.count + steps) "
                + "exact_prefix=\(exactPrefixTokens.map(String.init) ?? "full") "
                + "output=\(outputPath)"
        )
    }

    // Operator tool: attach the timed-benchmark oracle a ranked golden must
    // carry, derived from the golden's own hidden base case.
    //
    // Gemma4Runtime.benchmark refuses a golden with no `.benchmark` section even
    // on the gates-only phase (MLXFAST_BENCHMARK_CHECK_GATES=1 +
    // MLXFAST_BENCHMARK_SKIP_TIMED=1), so a hidden golden authored by
    // generate-golden + attach-free-run-gate alone -- neither of which can
    // emit an oracle -- fails the ranked "Correctness and gates" step. This
    // closes that provisioning gap. The derivation and the deliberate absence
    // of per-prompt baselines are specified in
    // goldenDocumentAttachingDerivedBenchmarkOracle; it needs no weights and
    // no model, so unlike the other attach verbs this one is pure file I/O.
    // Run it offline against the raw golden, then upload the result through
    // the organizer process (docs/private-benchmark-security.md).
    private static func runAttachBenchmarkOracle(_ options: ParsedOptions) throws {
        try options.validate(valueOptions: ["--golden", "--output"])
        let goldenPath = options.value(
            for: "--golden",
            default: environmentValue(
                "MLXFAST_CORRECTNESS_GOLDEN_PATH",
                fallback: MLXFastConstants.defaultGoldenPath
            )
        )
        let outputPath = options.value(for: "--output", default: goldenPath)

        try requireFile(goldenPath, description: "correctness golden file")
        // Strict-validate the INPUT before any write. --output defaults to the
        // input path, so a malformed input must fail here -- never after the
        // original has been replaced on disk. Through the Qwen loader: the
        // oracle it derives is what a ranked Qwen run is scored against.
        _ = try loadQwenGoldenFixture(from: goldenPath)
        let goldenData = try Data(contentsOf: URL(fileURLWithPath: goldenPath))
        let golden = try JSONDecoder().decode(GoldenDocument.self, from: goldenData)

        let merged = try goldenDocumentAttachingDerivedBenchmarkOracle(golden)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try writeValidatedGoldenDocument(encoder.encode(merged), to: outputPath)
        guard let oracle = merged.benchmark else {
            throw MLXFastError.invalidInput("attach-benchmark-oracle produced no benchmark oracle")
        }
        print(
            "attached benchmark oracle prefill_tokens=\(oracle.prefillPromptTokens.count) "
                + "decode_seed_tokens=\(oracle.decodeSeedTokens.count) "
                + "expected_decode_tokens=\(oracle.expectedDecodeTokens.count) "
                + "baselines=none "
                + "output=\(outputPath)"
        )
    }

    // Operator tool: generate a BASE golden case (the version-1 cases[] shape
    // consumed by `correctness` and the local benchmark modes) from a public
    // prompt text file against the reference weights. This is how the
    // checked-in public fixtures under correctness_prompts/ are produced:
    // tokenize the prompt with the weights-dir tokenizer using the same
    // addSpecialTokens convention as attach-free-run-gate's prompt-file path,
    // keep exactly the required 512 prompt tokens, greedy-generate the
    // requested continuation with the reference model, and write a fixture
    // that passes the strict loader at that step count. Greedy decoding is
    // deterministic, so fixtures generated from the same prompt at different
    // step counts are prefix-identical by construction.
    private static func runGenerateGolden(_ options: ParsedOptions) throws {
        try options.validate(
            valueOptions: ["--prompt-file", "--weights", "--tokenizer", "--output", "--name", "--steps"]
        )
        let promptFile = options.value(for: "--prompt-file", default: "")
        guard !promptFile.isEmpty else {
            throw MLXFastError.invalidInput("generate-golden requires --prompt-file PATH")
        }
        let weightsPath = options.value(
            for: "--weights",
            default: environmentValue("MLXFAST_WEIGHTS_PATH", fallback: MLXFastConstants.defaultWeightsPath)
        )
        let tokenizerPath = options.value(for: "--tokenizer", default: weightsPath)
        let outputPath = options.value(for: "--output", default: "")
        guard !outputPath.isEmpty else {
            throw MLXFastError.invalidInput("generate-golden requires --output PATH")
        }
        let caseName = options.value(for: "--name", default: "")
        guard !caseName.isEmpty else {
            throw MLXFastError.invalidInput("generate-golden requires --name NAME")
        }
        let steps = try parsePositiveInt(
            options.value(for: "--steps", default: ""),
            optionName: "--steps"
        )
        guard steps >= MLXFastConstants.correctnessSteps else {
            // The strict fixture loader rejects base cases shorter than the
            // correctness window, so fail before spending any generation time.
            throw MLXFastError.invalidInput(
                "--steps must be >= correctnessSteps \(MLXFastConstants.correctnessSteps)"
            )
        }

        try requireFile(promptFile, description: "golden prompt text file")
        try requireFile(
            URL(fileURLWithPath: tokenizerPath).appendingPathComponent("tokenizer.json").path,
            description: "tokenizer.json"
        )
        try requireFile(
            URL(fileURLWithPath: weightsPath).appendingPathComponent("config.json").path,
            description: "weights config.json"
        )

        let requiredPromptTokens = MLXFastConstants.correctnessPromptTokens
        let tokenizer = try loadLocalTokenizer(at: tokenizerPath)
        let promptText = try String(contentsOfFile: promptFile, encoding: .utf8)
        let encoded = tokenizer.encode(text: promptText, addSpecialTokens: false)
        guard encoded.count >= requiredPromptTokens else {
            throw MLXFastError.invalidInput(
                "--prompt-file tokenized to \(encoded.count) tokens; base golden cases need at least \(requiredPromptTokens)"
            )
        }
        let promptTokens = Array(encoded.prefix(requiredPromptTokens))

        fputs(
            "generate-golden: generating \(steps) reference continuation tokens "
                + "for case \(caseName) (prompt_tokens=\(promptTokens.count))\n",
            stderr
        )
        let expectedTokens = try Gemma4Runtime.generateGreedyTokens(
            GreedyGenerationOptions(
                weightsPath: weightsPath,
                promptTokens: promptTokens,
                steps: steps
            ),
            // Block the output path like the attach tools block their input
            // golden: when regenerating an existing fixture in place, the
            // worker running submitted-surface model code must not be able to
            // read the fixture it is being asked to reproduce.
            worker: try runtimeWorkerOptions(blockedGoldenPath: outputPath)
        )

        let document = GoldenDocument(
            version: 1,
            // Author the identity key the consumer will be held to. Without
            // this, generate-golden emits a golden that the very next
            // `preflight` rejects on model_type -- the reference emits it here
            // for exactly that reason. model_provenance stays alongside it,
            // additive as ever.
            modelType: MLXFastConstants.requiredGoldenModelType,
            modelProvenance: GoldenModelProvenance(
                repository: MLXFastConstants.referenceModelRepository,
                revision: MLXFastConstants.referenceModelRevision
            ),
            cases: [
                GoldenCase(
                    name: caseName,
                    promptTokens: promptTokens,
                    expectedTokens: expectedTokens
                )
            ],
            benchmark: nil
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try writeValidatedGoldenDocument(encoder.encode(document), to: outputPath)
        // The staging write above validates at the default correctness window;
        // re-validate at the full generated step count so the written fixture
        // provably satisfies the consumer that needs every step (local-submit
        // requires benchmarkDecodeSteps + 1 expected tokens, etc.) -- and,
        // through the Qwen loader, that it names the model it was generated on.
        _ = try loadQwenGoldenFixture(
            from: outputPath,
            requiredSteps: steps,
            requiredPromptTokens: requiredPromptTokens
        )
        print(
            "generated golden case=\(caseName) prompt_tokens=\(promptTokens.count) "
                + "expected_tokens=\(expectedTokens.count) output=\(outputPath)"
        )
    }

    // Operator tool: record the serial-reference "timed-prompt tape" the
    // ranked track's benchmark pool pins (benchd bench-core tape.rs,
    // TimedPromptTapeDocument). This is the ARM-NEUTRAL producer replacing
    // the Qwen-era `mtp-verify --generate` / DFlash `dflash-reference`
    // producers whose worker verbs left with the OQ-3 adoption
    // (docs/gemma4-port-notes.md OQ-3): it drives only the existing serial
    // correctness worker verbs — no drafter, no head, no speculative code —
    // and emits a reference artifact; it measures and scores nothing.
    // Run offline against the reference weights, then upload through the
    // organizer process like the other hidden artifacts.
    private static func runRecordReferenceTape(_ options: ParsedOptions) throws {
        try options.validate(
            valueOptions: [
                "--prompt-file", "--weights", "--tokenizer", "--output",
                "--steps", "--recording-backend",
            ],
            flagOptions: ["--allow-short"]
        )
        let promptFile = options.value(for: "--prompt-file", default: "")
        guard !promptFile.isEmpty else {
            throw MLXFastError.invalidInput(
                "record-reference-tape requires --prompt-file PATH"
            )
        }
        let weightsPath = options.value(
            for: "--weights",
            default: environmentValue(
                "MLXFAST_WEIGHTS_PATH",
                fallback: MLXFastConstants.defaultWeightsPath
            )
        )
        let tokenizerPath = options.value(for: "--tokenizer", default: weightsPath)
        let outputPath = options.value(for: "--output", default: "")
        guard !outputPath.isEmpty else {
            throw MLXFastError.invalidInput(
                "record-reference-tape requires --output PATH"
            )
        }
        // Default row count = the timed decode window plus the one-row margin
        // the pool tapes carry (seed argmax is separate; rows[i] is the token
        // at emitted index i+1). A shorter tape cannot oracle the full timed
        // window and benchd refuses it at measure time, so fail closed here
        // unless the operator explicitly opts in (debugging, staged runs) —
        // same posture as attach-free-run-gate's --allow-partial.
        let fullWindowRows = MLXFastConstants.benchmarkDecodeSteps + 1
        let steps = try parsePositiveInt(
            options.value(for: "--steps", default: "\(fullWindowRows)"),
            optionName: "--steps"
        )
        // Which serial implementation PRODUCES the rows. Default cbv2: the
        // width-1 CBv2 engine session the post-#29 free-run legs run — the
        // backend this track's oracle gates (port-notes 5.1,
        // within-backend). `legacy` (the teacher-forced correctness verbs)
        // stays selectable for provenance/reproduction of the
        // legacy-recorded pool tapes.
        let recordingBackend = try ReferenceTapeRecordingBackend(
            cliValue: options.value(
                for: "--recording-backend",
                default: ReferenceTapeRecordingBackend.standard.rawValue
            )
        )
        if steps < fullWindowRows {
            guard options.hasFlag("--allow-short") else {
                throw MLXFastError.invalidInput(
                    "--steps \(steps) is below the \(fullWindowRows) rows the "
                        + "timed decode window needs "
                        + "(benchmarkDecodeSteps + 1), so the tape could not "
                        + "oracle a ranked run; pass --allow-short to record "
                        + "it anyway"
                )
            }
            fputs(
                "record-reference-tape: warning: --steps \(steps) is below "
                    + "the full timed window's \(fullWindowRows) rows "
                    + "(--allow-short)\n",
                stderr
            )
        }

        try requireFile(promptFile, description: "reference tape prompt text file")
        try requireFile(
            URL(fileURLWithPath: tokenizerPath).appendingPathComponent("tokenizer.json").path,
            description: "tokenizer.json"
        )
        try requireFile(
            URL(fileURLWithPath: weightsPath).appendingPathComponent("config.json").path,
            description: "weights config.json"
        )

        // Same tokenization convention as generate-golden / the free-run
        // prompt-file path: encode without special tokens, then the seed is
        // exactly the first correctnessPromptTokens (1024) tokens.
        let tokenizer = try loadLocalTokenizer(at: tokenizerPath)
        let promptText = try String(contentsOfFile: promptFile, encoding: .utf8)
        let encoded = tokenizer.encode(text: promptText, addSpecialTokens: false)
        let seedTokens = try Gemma4Runtime.referenceTapeSeed(fromEncodedPrompt: encoded)

        // SESSION EVIDENCE: backend provenance lives here (and in the
        // progress lines below), never in the document — the tape's byte
        // format is backend-neutral by contract.
        fputs(
            "record-reference-tape: recording \(steps) reference rows "
                + "(seed_tokens=\(seedTokens.count)) on recording backend "
                + "'\(recordingBackend.rawValue)' plus a full "
                + "self-consistency replay through the same backend\n",
            stderr
        )
        let document = try Gemma4Runtime.recordReferenceTape(
            ReferenceTapeRecordingOptions(
                weightsPath: weightsPath,
                seedTokens: seedTokens,
                steps: steps
            ),
            // Block the output path like generate-golden blocks its output:
            // when regenerating an existing pool tape in place, the worker
            // running submitted-surface model code must not be able to read
            // the artifact it is being asked to reproduce.
            worker: try runtimeWorkerOptions(blockedGoldenPath: outputPath),
            backend: recordingBackend,
            progress: { message in
                fputs("record-reference-tape: \(message)\n", stderr)
            }
        )
        try writeValidatedReferenceTape(document, to: outputPath, requiredRows: steps)
        print(
            "recorded reference tape seed_tokens=\(document.seedTokens.count) "
                + "reference_seed_token=\(document.referenceSeedToken) "
                + "rows=\(document.rows.count) "
                + "reference_self_consistent=\(document.referenceSelfConsistent) "
                + "recording_backend=\(recordingBackend.rawValue) "
                + "output=\(outputPath)"
        )
    }

    // Same staging discipline as writeValidatedGoldenDocument: the emitted
    // bytes must survive the strict tape validation BEFORE they can touch the
    // destination, so an in-place regeneration can never destroy an existing
    // pool tape and leave a malformed one behind.
    private static func writeValidatedReferenceTape(
        _ document: ReferenceTapeDocument,
        to outputPath: String,
        requiredRows: Int
    ) throws {
        let outputData = try Gemma4Runtime.encodeReferenceTapeDocument(document)
        let outputURL = URL(fileURLWithPath: outputPath)
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let temporaryURL = outputURL.deletingLastPathComponent()
            .appendingPathComponent(".\(outputURL.lastPathComponent).tape-\(UUID().uuidString).tmp")
        defer {
            try? FileManager.default.removeItem(at: temporaryURL)
        }
        try outputData.write(to: temporaryURL, options: [.atomic])
        try Gemma4Runtime.validateReferenceTapeDocumentData(
            try Data(contentsOf: temporaryURL),
            requiredRows: requiredRows
        )
        if FileManager.default.fileExists(atPath: outputURL.path) {
            _ = try FileManager.default.replaceItemAt(outputURL, withItemAt: temporaryURL)
        } else {
            try FileManager.default.moveItem(at: temporaryURL, to: outputURL)
        }
    }

    private static func runAnalyzeNGramSimilarity(_ options: ParsedOptions) throws {
        try options.validate(
            valueOptions: ["--golden", "--case", "--orders", "--max-hit-rate"]
        )
        let goldenPath = options.value(for: "--golden", default: "")
        guard !goldenPath.isEmpty else {
            throw MLXFastError.invalidInput("analyze-ngram-similarity requires --golden PATH")
        }
        let orderText = options.value(
            for: "--orders",
            default: MLXFastConstants.benchmarkNGramSelfSimilarityOrders
                .map(String.init)
                .joined(separator: ",")
        )
        let orders = try orderText.split(separator: ",").map { component in
            guard let order = Int(component), order > 0 else {
                throw MLXFastError.invalidInput(
                    "--orders must be a comma-separated list of positive integers"
                )
            }
            return order
        }
        let maximumHitRateText = options.value(
            for: "--max-hit-rate",
            default: "\(MLXFastConstants.benchmarkMaxPromptLookupHitRate)"
        )
        guard let maximumHitRate = Double(maximumHitRateText),
              maximumHitRate.isFinite,
              (0...1).contains(maximumHitRate)
        else {
            throw MLXFastError.invalidInput("--max-hit-rate must be a finite value in 0...1")
        }

        // Qwen loader: the hit rate this reports is the anti-lottery property of
        // a QWEN golden, and the output is stamped with benchmarkEvaluationTargetID.
        let fixture = try loadQwenGoldenFixture(from: goldenPath)
        let requestedCase = options.value(for: "--case", default: "")
        let contextTokens: [Int]
        let continuationTokens: [Int]
        let source: String
        if !requestedCase.isEmpty {
            guard let goldenCase = fixture.cases.first(where: { $0.name == requestedCase }) else {
                throw MLXFastError.invalidInput("golden does not contain base case \(requestedCase)")
            }
            contextTokens = goldenCase.promptTokens
            continuationTokens = try benchmarkAnalysisContinuation(from: goldenCase)
            source = "case:\(goldenCase.name)"
        } else if let benchmark = fixture.benchmark {
            contextTokens = benchmark.decodeSeedTokens
            continuationTokens = [benchmark.expectedDecodeSeedToken]
                + Array(benchmark.expectedDecodeTokens.prefix(MLXFastConstants.benchmarkDecodeSteps))
            source = "benchmark"
        } else {
            guard let goldenCase = fixture.cases.first else {
                throw MLXFastError.invalidInput("golden contains no base case to analyze")
            }
            contextTokens = goldenCase.promptTokens
            continuationTokens = try benchmarkAnalysisContinuation(from: goldenCase)
            source = "case:\(goldenCase.name)"
        }

        let report = try NGramSelfSimilarity.analyze(
            contextTokens: contextTokens,
            continuationTokens: continuationTokens,
            orders: orders
        )
        let passed = report.passes(maximumHitRate: maximumHitRate)
        let output = NGramSimilarityAnalysisOutput(
            targetID: MLXFastConstants.benchmarkEvaluationTargetID,
            source: source,
            maximumHitRate: maximumHitRate,
            passed: passed,
            report: report
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        var encoded = try encoder.encode(output)
        encoded.append(0x0A)
        FileHandle.standardOutput.write(encoded)

        guard passed else {
            throw MLXFastError.invalidInput(
                "prompt-lookup hit rate \(report.longestMatchMostRecentHitRate) "
                    + "exceeds maximum \(maximumHitRate)"
            )
        }
    }

    private static func benchmarkAnalysisContinuation(from goldenCase: GoldenCase) throws -> [Int] {
        let requiredTokens = MLXFastConstants.benchmarkDecodeSteps + 1
        guard goldenCase.expectedTokens.count >= requiredTokens else {
            throw MLXFastError.invalidInput(
                "base case \(goldenCase.name) has \(goldenCase.expectedTokens.count) continuation tokens; "
                    + "need at least \(requiredTokens) to score the decode seed token plus "
                    + "\(MLXFastConstants.benchmarkDecodeSteps) timed tokens"
            )
        }
        return Array(goldenCase.expectedTokens.prefix(requiredTokens))
    }

    // Writes a merged golden by staging to a temp sibling and proving the
    // result loads through the strict fixture loader BEFORE it can touch the
    // destination. The attach commands default --output to the input golden,
    // so an in-place write followed by a failed validation would destroy the
    // original (typically the private golden) with nothing to roll back to.
    private static func writeValidatedGoldenDocument(_ outputData: Data, to outputPath: String) throws {
        let outputURL = URL(fileURLWithPath: outputPath)
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let temporaryURL = outputURL.deletingLastPathComponent()
            .appendingPathComponent(".\(outputURL.lastPathComponent).attach-\(UUID().uuidString).tmp")
        defer {
            try? FileManager.default.removeItem(at: temporaryURL)
        }
        try outputData.write(to: temporaryURL, options: [.atomic])
        // Qwen loader on the staged bytes: every caller of this helper is an
        // attach verb writing a Qwen golden, so an attach can never land a
        // document that has lost (or never carried) the model identity.
        _ = try loadQwenGoldenFixture(from: temporaryURL.path)
        if FileManager.default.fileExists(atPath: outputURL.path) {
            _ = try FileManager.default.replaceItemAt(outputURL, withItemAt: temporaryURL)
        } else {
            try FileManager.default.moveItem(at: temporaryURL, to: outputURL)
        }
    }

    private static func runGenerateGPQAAnswers(_ options: ParsedOptions) throws {
        try options.validate(
            valueOptions: ["--gpqa", "--weights", "--tokenizer", "--output", "--case-count", "--max-new-tokens"]
        )
        let gpqaPath = options.value(
            for: "--gpqa",
            default: environmentValue("MLXFAST_GPQA_REFERENCE_PATH", fallback: "")
        )
        guard !gpqaPath.isEmpty else {
            throw MLXFastError.invalidInput("generate-gpqa-answers requires --gpqa or MLXFAST_GPQA_REFERENCE_PATH")
        }
        let weightsPath = options.value(
            for: "--weights",
            default: environmentValue("MLXFAST_WEIGHTS_PATH", fallback: MLXFastConstants.defaultWeightsPath)
        )
        let tokenizerPath = options.value(
            for: "--tokenizer",
            default: environmentValue("MLXFAST_TOKENIZER_PATH", fallback: weightsPath)
        )
        let outputPath = options.value(
            for: "--output",
            default: environmentValue("MLXFAST_SEMANTIC_GPQA_OUTPUT_PATH", fallback: "")
        )
        guard !outputPath.isEmpty else {
            throw MLXFastError.invalidInput(
                "generate-gpqa-answers requires --output or MLXFAST_SEMANTIC_GPQA_OUTPUT_PATH"
            )
        }
        try requirePrivateOutputPath(outputPath, description: "semantic GPQA answer output")
        let caseCount = try parsePositiveInt(
            options.value(for: "--case-count", default: "\(MLXFastConstants.semanticGPQACaseCount)"),
            optionName: "--case-count"
        )
        let maxNewTokens = try parsePositiveInt(
            options.value(for: "--max-new-tokens", default: "\(MLXFastConstants.semanticGPQAMaxNewTokens)"),
            optionName: "--max-new-tokens"
        )
        guard maxNewTokens <= MLXFastConstants.correctnessMaxBehaviorSteps else {
            throw MLXFastError.invalidInput(
                "--max-new-tokens must be <= \(MLXFastConstants.correctnessMaxBehaviorSteps)"
            )
        }

        try requireFile(gpqaPath, description: "GPQA reference cases file")
        try requireFile(
            URL(fileURLWithPath: tokenizerPath).appendingPathComponent("tokenizer.json").path,
            description: "tokenizer.json"
        )
        try requireFile(
            URL(fileURLWithPath: tokenizerPath).appendingPathComponent("tokenizer_config.json").path,
            description: "tokenizer_config.json"
        )
        try requireFile(
            URL(fileURLWithPath: weightsPath).appendingPathComponent("config.json").path,
            description: "weights config.json"
        )

        let tokenizer = try loadLocalTokenizer(at: tokenizerPath)
        let data = try Data(contentsOf: URL(fileURLWithPath: gpqaPath))
        let gpqa = try JSONDecoder().decode(GPQAReferenceDocument.self, from: data)
        let worker = try runtimeWorkerOptions(blockedGoldenPath: gpqaPath)

        var answers: [SemanticGPQAAnswerCase] = []
        var skippedOverBudget = 0
        for testCase in gpqa.cases {
            guard answers.count < caseCount else {
                break
            }
            // Must match buildGPQABehaviorCaseIfWithinPromptBudget exactly:
            // this verb exists to reproduce the in-run capture offline, so any
            // divergence in framing makes the two incomparable.
            let promptTokens = tokenizer.encode(
                text: Gemma4ChatTemplate.userTurnDisablingThinking(testCase.prompt),
                addSpecialTokens: true
            )
            guard !promptTokens.isEmpty else {
                throw MLXFastError.invalidInput("\(testCase.identifier).prompt tokenized to zero tokens")
            }
            guard promptTokens.count <= MLXFastConstants.correctnessMaxBehaviorPromptTokens else {
                skippedOverBudget += 1
                continue
            }

            let generated = try Gemma4Runtime.generateGreedyTokens(
                GreedyGenerationOptions(
                    weightsPath: weightsPath,
                    promptTokens: promptTokens,
                    steps: maxNewTokens
                ),
                worker: worker
            )
            let answerTokens = Gemma4ChatTemplate.truncatedAtFirstEndOfTurn(
                generated,
                eosTokenId: tokenizer.eosTokenId
            )
            let decoded = tokenizer.decode(tokens: answerTokens, skipSpecialTokens: true)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            answers.append(
                SemanticGPQAAnswerCase(
                    id: testCase.identifier,
                    domain: testCase.domain,
                    subdomain: testCase.subdomain,
                    prompt: testCase.prompt,
                    answerKey: testCase.answerKey,
                    referenceAnswer: referenceAnswer(for: testCase),
                    candidateAnswer: decoded,
                    candidateTokens: answerTokens,
                    maxNewTokens: maxNewTokens
                )
            )
            fputs(
                "generate-gpqa-answers: generated \(answers.count)/\(caseCount) "
                    + "tokens=\(generated.count)\n",
                stderr
            )
        }
        guard answers.count == caseCount else {
            throw MLXFastError.invalidInput(
                "GPQA reference produced \(answers.count) token-budget-valid semantic cases; "
                    + "need \(caseCount); skipped_over_budget=\(skippedOverBudget)"
            )
        }

        let document = SemanticGPQAAnswerDocument(
            version: 1,
            cases: answers
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let outputURL = URL(fileURLWithPath: outputPath)
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try encoder.encode(document).write(to: outputURL, options: [.atomic])
        print("generated semantic GPQA answer cases=\(answers.count) output=\(outputPath)")
    }

    private static func loadLocalTokenizer(at path: String) throws -> any Tokenizer {
        let modelFolder = URL(fileURLWithPath: path).standardizedFileURL
        return try runBlockingAsync {
            try await AutoTokenizer.from(modelFolder: modelFolder, strict: false)
        }
    }

    private static func requirePrivateOutputPath(_ path: String, description: String) throws {
        let privateDir = environmentValue("MLXFAST_PRIVATE_DIR", fallback: "")
        guard !privateDir.isEmpty else {
            return
        }
        let outputPath = absolutePath(path)
        let privatePath = absolutePath(privateDir)
        guard outputPath.hasPrefix(privatePath + "/") else {
            throw MLXFastError.invalidInput("\(description) must be under MLXFAST_PRIVATE_DIR")
        }
    }

    private static func referenceAnswer(for testCase: GPQAReferenceCase) -> String {
        if let expected = trimmedNonEmpty(testCase.expectedResponse) {
            return expected
        }
        if let accepted = testCase.acceptedResponses?.compactMap({ trimmedNonEmpty($0) }), !accepted.isEmpty {
            return accepted.joined(separator: "\n")
        }
        if let answerKey = trimmedNonEmpty(testCase.answerKey) {
            if let answerText = multipleChoiceAnswerText(in: testCase.prompt, answerKey: answerKey) {
                return "\(answerKey). \(answerText)"
            }
            return "Correct option: \(answerKey)"
        }
        return ""
    }

    private static func multipleChoiceAnswerText(in prompt: String, answerKey: String) -> String? {
        let normalizedKey = answerKey.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        guard normalizedKey.count == 1 else {
            return nil
        }
        for rawLine in prompt.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            for marker in ["\(normalizedKey).", "\(normalizedKey):", "\(normalizedKey))"]
                where line.hasPrefix(marker)
            {
                let start = line.index(line.startIndex, offsetBy: marker.count)
                let value = line[start...].trimmingCharacters(in: .whitespacesAndNewlines)
                if !value.isEmpty {
                    return value
                }
            }
        }
        return nil
    }

    private static func buildGPQABehaviorCaseIfWithinPromptBudget(
        _ testCase: GPQAReferenceCase,
        tokenizer: any Tokenizer,
        maxNewTokens: Int
    ) throws -> GoldenBehaviorCase? {
        // ChatML-framed, thinking pre-closed: see Gemma4ChatTemplate. The raw
        // question produced un-framed continuations that never answered.
        let promptTokens = tokenizer.encode(
            text: Gemma4ChatTemplate.userTurnDisablingThinking(testCase.prompt),
            addSpecialTokens: true
        )
        guard !promptTokens.isEmpty else {
            throw MLXFastError.invalidInput("\(testCase.identifier).prompt tokenized to zero tokens")
        }
        guard promptTokens.count <= MLXFastConstants.correctnessMaxBehaviorPromptTokens else {
            return nil
        }
        let acceptedSequences = try acceptedReferenceTokenSequences(
            testCase: testCase,
            tokenizer: tokenizer,
            maxNewTokens: maxNewTokens,
            caseName: testCase.identifier
        )
        return GoldenBehaviorCase(
            name: testCase.identifier,
            promptTokens: promptTokens,
            acceptedTokenSequences: acceptedSequences,
            maxNewTokens: maxNewTokens,
            semanticPrompt: testCase.prompt,
            semanticAnswerKey: trimmedNonEmpty(testCase.answerKey),
            semanticReferenceAnswer: referenceAnswer(for: testCase),
            semanticDomain: trimmedNonEmpty(testCase.domain),
            semanticSubdomain: trimmedNonEmpty(testCase.subdomain)
        )
    }

    private static func acceptedReferenceTokenSequences(
        testCase: GPQAReferenceCase,
        tokenizer: any Tokenizer,
        maxNewTokens: Int,
        caseName: String
    ) throws -> [[Int]] {
        if let tokenSequences = testCase.acceptedTokenSequences {
            guard !tokenSequences.isEmpty else {
                throw MLXFastError.invalidInput("\(caseName).accepted_token_sequences must not be empty")
            }
            var acceptedPrefixes: [[Int]] = []
            for (index, sequence) in tokenSequences.enumerated() {
                guard !sequence.isEmpty else {
                    throw MLXFastError.invalidInput(
                        "\(caseName).accepted_token_sequences[\(index)] must not be empty"
                    )
                }
                acceptedPrefixes.append(Array(sequence.prefix(maxNewTokens)))
            }
            return uniqueSortedTokenSequences(acceptedPrefixes)
        }

        guard let acceptedResponses = testCase.acceptedResponses,
              !acceptedResponses.isEmpty
        else {
            throw MLXFastError.invalidInput(
                "\(caseName) requires accepted_token_sequences or accepted_responses generated from the reference model"
            )
        }

        let prefixes = ["", " ", "\n"]
        let suffixes = ["", ".", "\n"]
        var seen = Set<[Int]>()
        var sequences: [[Int]] = []
        for response in acceptedResponses {
            let trimmed = response.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else {
                continue
            }
            for prefix in prefixes {
                for suffix in suffixes {
                    let tokens = tokenizer.encode(text: prefix + trimmed + suffix, addSpecialTokens: false)
                    guard !tokens.isEmpty, tokens.count <= maxNewTokens else {
                        continue
                    }
                    if seen.insert(tokens).inserted {
                        sequences.append(tokens)
                    }
                }
            }
        }
        guard !sequences.isEmpty else {
            throw MLXFastError.invalidInput(
                "\(caseName) accepted_responses have no tokenization within \(maxNewTokens) token(s)"
            )
        }
        return sequences.sorted { lhs, rhs in
            if lhs.count != rhs.count {
                return lhs.count < rhs.count
            }
            return lhs.lexicographicallyPrecedes(rhs)
        }
    }

    private static func uniqueSortedTokenSequences(_ tokenSequences: [[Int]]) -> [[Int]] {
        var seen = Set<[Int]>()
        var sequences: [[Int]] = []
        for sequence in tokenSequences where seen.insert(sequence).inserted {
            sequences.append(sequence)
        }
        return sequences.sorted { lhs, rhs in
            if lhs.count != rhs.count {
                return lhs.count < rhs.count
            }
            return lhs.lexicographicallyPrecedes(rhs)
        }
    }

    private final class AsyncResultBox<T>: @unchecked Sendable {
        var result: Result<T, Error>?
    }

    private static func runBlockingAsync<T>(
        _ body: @escaping @Sendable () async throws -> T
    ) throws -> T {
        let semaphore = DispatchSemaphore(value: 0)
        let box = AsyncResultBox<T>()
        Task {
            do {
                box.result = .success(try await body())
            } catch {
                box.result = .failure(error)
            }
            semaphore.signal()
        }
        semaphore.wait()
        return try box.result!.get()
    }

    private static func parsePositiveInt(_ rawValue: String, optionName: String) throws -> Int {
        guard let value = Int(rawValue), value > 0 else {
            throw MLXFastError.invalidInput("\(optionName) must be a positive integer")
        }
        return value
    }

    private static func parseNonNegativeInt(_ rawValue: String, optionName: String) throws -> Int {
        guard let value = Int(rawValue), value >= 0 else {
            throw MLXFastError.invalidInput("\(optionName) must be a non-negative integer")
        }
        return value
    }

    /// DFlash block-decode measurement (track laguna-xs-2.1-dflash-v1).
    ///
    /// Unranked and fail-closed while the track contract's
    /// `official_scoring_enabled` stays false: this emits diagnostics only and
    /// never writes a ranked score. The reference verdicts come from a
    /// pinned-baseline-generated golden (contract layer L1), never from the
    /// candidate binary.
    /// `dflash-benchmark` (block decode) and `dflash-probe` (the serial K=1
    /// control) share this driver: same worker, same protocol, same forward,
    /// only the block width differs. That is what makes the paired ratio a
    /// like-for-like comparison instead of two implementations racing.
    ///
    /// The report JSON goes to STDOUT as a single object, which is what the box
    /// measurement wrapper reads; the human summary goes to stderr.
    private static func runDFlashBenchmark(
        _ options: ParsedOptions,
        serialControl: Bool = false
    ) throws {
        let subcommand = serialControl ? "dflash-probe" : "dflash-benchmark"
        try options.validate(
            valueOptions: [
                "--weights", "--drafter", "--golden", "--block-size",
                "--tokens", "--schedule-seed", "--output",
                // Calibration only -- see the guard below.
                "--work-binding-tolerance-absolute",
                "--work-binding-tolerance-relative",
            ]
        )
        let weightsPath = options.value(
            for: "--weights",
            default: environmentValue(
                "MLXFAST_WEIGHTS_PATH",
                fallback: MLXFastConstants.defaultWeightsPath
            )
        )
        // The control loads the drafter too and simply never drafts: keeping ONE
        // worker binary and ONE load path means the two sides of the ratio
        // cannot diverge in anything except the block width.
        let drafterPath = options.value(
            for: "--drafter",
            default: environmentValue("MLXFAST_DFLASH_DRAFTER_DIR", fallback: "")
        )
        guard !drafterPath.isEmpty else {
            throw MLXFastError.invalidInput(
                "\(subcommand) requires --drafter PATH (or "
                    + "MLXFAST_DFLASH_DRAFTER_DIR)"
            )
        }
        let goldenPath = options.value(for: "--golden", default: "")
        guard !goldenPath.isEmpty else {
            throw MLXFastError.invalidInput(
                "\(subcommand) requires --golden PATH (the pinned-baseline "
                    + "reference golden)"
            )
        }
        let requestedBlockSize = try positiveInteger(
            options.value(
                for: "--block-size",
                default: String(MLXFastConstants.experimentalDFlashMaxBlockSize)
            ),
            name: "--block-size"
        )
        let blockSize = serialControl ? 1 : requestedBlockSize
        let tokens = try positiveInteger(
            options.value(
                for: "--tokens",
                default: String(
                    MLXFastConstants.experimentalDFlashMaxTotalTokens
                )
            ),
            name: "--tokens"
        )
        let scheduleSeed = UInt64(
            options.value(for: "--schedule-seed", default: "0")
        ) ?? 0
        guard let workerOptions = try runtimeWorkerOptions(
            blockedGoldenPath: goldenPath,
            // Operator seam diagnostic: the worker's ring-cache trace is only
            // useful if it reaches the parent's stderr. `runtimeWorkerOptions`
            // ANDs this with `!officialRun`, so a ranked run stays silent.
            forwardsWorkerStderr: environmentValue(
                "MLX_DFLASH_TRACE_CACHE_SEAM",
                fallback: "0"
            ) == "1"
        ) else {
            throw MLXFastError.invalidInput(
                "\(subcommand) requires the participant runtime worker"
            )
        }

        // The work-binding tolerance is OPERATOR CALIBRATION MACHINERY, not a run
        // parameter: the honest gap distribution has to be measured before the
        // constant can be defended, and measuring it means running with the check
        // widened. A flag that widens a gate is a bypass surface, so it is
        // refused outright on the official path -- a ranked run always uses the
        // compiled-in constant.
        let toleranceAbsolute = options.value(
            for: "--work-binding-tolerance-absolute",
            default: ""
        )
        let toleranceRelative = options.value(
            for: "--work-binding-tolerance-relative",
            default: ""
        )
        let toleranceDefaults = DFlashWorkBindingTolerance()
        var tolerance = toleranceDefaults
        if !toleranceAbsolute.isEmpty || !toleranceRelative.isEmpty {
            guard environmentValue(
                "MLXFAST_OFFICIAL_BENCHMARK_RUN",
                fallback: "0"
            ) != "1" else {
                throw MLXFastError.invalidInput(
                    "official benchmark runs must use the compiled-in DFlash "
                        + "work-binding tolerance; drop "
                        + "--work-binding-tolerance-*"
                )
            }
            guard let absolute = Double(
                toleranceAbsolute.isEmpty
                    ? String(toleranceDefaults.absolute)
                    : toleranceAbsolute
            ), absolute >= 0,
                let relative = Double(
                    toleranceRelative.isEmpty
                        ? String(toleranceDefaults.relative)
                        : toleranceRelative
                ), relative >= 0
            else {
                throw MLXFastError.invalidInput(
                    "--work-binding-tolerance-* require non-negative numbers"
                )
            }
            tolerance = DFlashWorkBindingTolerance(
                absolute: absolute,
                relative: relative
            )
            fputs(
                "\(subcommand): WARNING calibration tolerance in use "
                    + "(absolute=\(absolute) relative=\(relative)); this run is "
                    + "NOT a contract-enforcing run\n",
                stderr
            )
        }

        let report = try Gemma4Runtime.experimentalDFlashBenchmark(
            options: ExperimentalDFlashOptions(
                targetWeightsPath: weightsPath,
                drafterPath: drafterPath,
                goldenPath: goldenPath,
                maxBlockSize: blockSize,
                totalTokenCount: tokens
            ),
            workerOptions: workerOptions,
            scheduleSeed: scheduleSeed,
            tolerance: tolerance
        )

        fputs(
            "\(subcommand): tokens=\(report.totalTokenCount) "
                + "rounds=\(report.roundCount) "
                + "block_size=\(report.blockSize) "
                + "decode_seconds=\(String(format: "%.4f", report.decodeSeconds)) "
                + "seconds_per_token="
                + "\(String(format: "%.6f", report.decodeSecondsPerToken)) "
                + "accepted_draft_rate="
                + "\(String(format: "%.4f", report.acceptedDraftRate)) "
                + "residual_divergences=\(report.residualDivergenceCount) "
                + "rejected_rows_reference_checked="
                + "\(report.rejectedRowsReferenceChecked) "
                + "max_rejected_tail_logit_delta="
                + "\(String(format: "%.4f", report.maxRejectedTailLogitDelta))\n",
            stderr
        )

        // Field names are the box measurement wrapper contract; see the
        // runner's measure-dflash-job.sh wrapper.
        var payload: [String: Any] = [
            "track_id": "laguna-xs-2.1-dflash-v1",
            "official_score_produced": false,
            "all_tokens_matched": report.allTokensAdmissible,
            "parent_measured_seconds_per_token": report.decodeSecondsPerToken,
            "decode_token_count": report.totalTokenCount,
            "block_size": report.blockSize,
            "uses_trained_drafter": report.usesTrainedDrafter,
            "max_block_request_seconds": report.maxBlockRequestSeconds,
            "p50_block_request_seconds": report.p50BlockRequestSeconds,
            "decode_seconds": report.decodeSeconds,
            "accepted_draft_rate": report.acceptedDraftRate,
            "residual_divergence_count": report.residualDivergenceCount,
            "admissible_exact_count": report.admissibleExactCount,
            "admissible_declared_frame_count":
                report.admissibleDeclaredFrameCount,
            "admissible_near_tie_count": report.admissibleNearTieCount,
            // L3 ledger.
            "emitted_token_total": report.totalTokenCount,
            "declared_rows_total": report.declaredRowTotal,
            "reference_checked_row_total": report.referenceCheckedRowTotal,
            // Amendment 21: the rejected tail is priced now, so an audit can see
            // whether it actually was. Zero here on a block-decode run means the
            // tail went unpriced -- the legacy fallback, not a pass.
            "rejected_rows_reference_checked":
                report.rejectedRowsReferenceChecked,
            "verify_block_replayed_round_count":
                report.verifyBlockReplayedRoundCount,
            "rejected_tail_comparison_count":
                report.rejectedTailComparisonCount,
            "max_rejected_tail_logit_delta": report.maxRejectedTailLogitDelta,
            "accepted_draft_total": report.acceptedDraftTotal,
            "rejected_draft_total": report.rejectedDraftTotal,
            "target_tail_total": report.targetTailTotal,
            "round_count": report.roundCount,
            "seed_token_count": report.seedTokenCount,
            "target_cache_offset_final": report.targetCacheOffsetFinal,
            // L2 work-binding gap distribution. Reported so the tolerance stays
            // traceable to measurement (contract Amendment 1) and so an audit can
            // see whether the binding compared anything at all.
            "work_binding_comparison_count": report.workBindingComparisonCount,
            "max_top2_logit_delta": report.maxTop2LogitDelta,
            "mean_top2_logit_delta": report.meanTop2LogitDelta,
            "p50_top2_logit_delta": report.p50Top2LogitDelta,
            "p99_top2_logit_delta": report.p99Top2LogitDelta,
            "max_top2_logit_relative_delta": report.maxTop2LogitRelativeDelta,
            "p99_top2_logit_relative_delta": report.p99Top2LogitRelativeDelta,
            "work_binding_tolerance_absolute":
                report.workBindingToleranceAbsolute,
            "work_binding_tolerance_relative":
                report.workBindingToleranceRelative,
        ]
        if let stall = report.maxOverMedianRoundLatency {
            payload["max_over_median_round_latency"] = stall
        }
        // Per-comparison gaps are calibration output, not run output: on a ranked
        // run they would be a per-row proximity trace against a hidden prompt,
        // which contract layer L6 keeps out of any published artifact. The
        // widened tolerance that enables them is refused on the official path.
        if tolerance.absolute != toleranceDefaults.absolute
            || tolerance.relative != toleranceDefaults.relative
        {
            payload["work_binding_logit_deltas"] = report.workBindingLogitDeltas
            // Index-for-index block width behind each gap. The absolute arm is
            // calibrated per width and a run's schedule mixes widths, so the
            // gaps alone cannot reproduce the derivation.
            payload["work_binding_comparison_widths"] =
                report.workBindingComparisonWidths
        }
        let data = try JSONSerialization.data(
            withJSONObject: payload,
            options: [.prettyPrinted, .sortedKeys]
        )
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data("\n".utf8))
        let outputPath = options.value(for: "--output", default: "")
        if !outputPath.isEmpty {
            try data.write(to: URL(fileURLWithPath: outputPath))
        }
    }

    /// `dflash-reference`: generate the pinned-baseline reference golden
    /// (contract layer L1). Run this AFTER the timed phase, from the pinned
    /// baseline tree, over organizer-transformed weights.
    private static func runDFlashReference(_ options: ParsedOptions) throws {
        try options.validate(
            valueOptions: [
                "--weights", "--drafter", "--emitted", "--output",
                // Chain generation: the reference produces the decode chain
                // itself instead of replaying a hand-written `emitted` array.
                "--generate", "--seed-generate", "--block-size",
                "--schedule-seed", "--plan-output",
            ]
        )
        let weightsPath = options.value(
            for: "--weights",
            default: environmentValue(
                "MLXFAST_WEIGHTS_PATH",
                fallback: MLXFastConstants.defaultWeightsPath
            )
        )
        let drafterPath = options.value(
            for: "--drafter",
            default: environmentValue("MLXFAST_DFLASH_DRAFTER_DIR", fallback: "")
        )
        guard !drafterPath.isEmpty else {
            throw MLXFastError.invalidInput(
                "dflash-reference requires --drafter PATH (or "
                    + "MLXFAST_DFLASH_DRAFTER_DIR)"
            )
        }
        let emittedPath = options.value(for: "--emitted", default: "")
        let generateCount = try optionalCount(
            options.value(for: "--generate", default: ""),
            name: "--generate"
        )
        guard !emittedPath.isEmpty else {
            throw MLXFastError.invalidInput(
                "dflash-reference requires --emitted PATH (the emitted plan to "
                    + "replay, or a seed-only plan when --generate is used)"
            )
        }
        let outputPath = options.value(for: "--output", default: "")
        guard !outputPath.isEmpty else {
            throw MLXFastError.invalidInput(
                "dflash-reference requires --output PATH (the golden to write)"
            )
        }
        guard let workerOptions = try runtimeWorkerOptions() else {
            throw MLXFastError.invalidInput(
                "dflash-reference requires the participant runtime worker"
            )
        }

        let plan = try JSONDecoder().decode(
            DFlashEmittedPlan.self,
            from: try Data(contentsOf: URL(fileURLWithPath: emittedPath))
        )

        // `--generate N` makes the reference produce the decode chain itself.
        // `--seed-generate M` first extends the SEED by M reference-generated
        // tokens, which is the only way to build a long seed here: the trusted
        // binary links no tokenizer, so seed length is dialled in tokens, not
        // text. That is what makes the 512-slot ring seam reachable.
        var chain: DFlashReferenceChainOptions?
        if let generateCount {
            let planOutput = options.value(for: "--plan-output", default: "")
            chain = DFlashReferenceChainOptions(
                seedExtensionSteps: try optionalCount(
                    options.value(for: "--seed-generate", default: ""),
                    name: "--seed-generate"
                ) ?? 0,
                generateTokenCount: generateCount,
                roundBlockSize: try positiveInteger(
                    options.value(for: "--block-size", default: "1"),
                    name: "--block-size"
                ),
                scheduleSeed: UInt64(
                    options.value(for: "--schedule-seed", default: "0")
                ) ?? 0,
                planOutputPath: planOutput.isEmpty
                    ? outputPath + ".plan.json"
                    : planOutput
            )
        }

        let result = try Gemma4Runtime.experimentalDFlashReferenceGolden(
            plan: plan,
            chain: chain,
            targetWeightsPath: weightsPath,
            drafterPath: drafterPath,
            outputPath: outputPath,
            workerOptions: workerOptions
        )

        print(
            "dflash-reference: rows=\(result.rowCount) "
                + "seed_tokens=\(result.seedTokenCount) "
                + "reference_seed_token=\(result.referenceSeedToken) "
                + "recorded_frame_widths="
                + "\(result.recordedFrameWidths.map(String.init).joined(separator: ","))"
                + " plan_output=\(result.planOutputPath ?? "-") "
                + "reference_self_consistent=\(result.selfConsistent) "
                + "replayed_rows=\(result.selfConsistencyRowCount) "
                + "chain_row_contradictions="
                + "\(result.chainRowContradictionCount) "
                + "detail=\(result.selfConsistencyDetail)"
        )
        guard result.selfConsistent else {
            // OPERATOR fault, not a submission fault: the admissible sets this
            // golden defines are undefined if the reference is not
            // deterministic, so refuse to hand it onward.
            throw MLXFastError.invalidInput(
                "DFlash reference failed its self-consistency replay "
                    + "(\(result.selfConsistencyDetail)); this is an OPERATOR "
                    + "fault -- the reference build is nondeterministic -- not a "
                    + "participant failure. The golden was written with "
                    + "reference_self_consistent=false and must not be scored "
                    + "against."
            )
        }
    }

    private static func positiveInteger(
        _ text: String,
        name: String
    ) throws -> Int {
        guard let value = Int(text), value > 0 else {
            throw MLXFastError.invalidInput("\(name) requires a positive integer")
        }
        return value
    }

    /// A non-negative count that is absent when the flag was not passed.
    private static func optionalCount(
        _ text: String,
        name: String
    ) throws -> Int? {
        guard !text.isEmpty else { return nil }
        guard let value = Int(text), value >= 0 else {
            throw MLXFastError.invalidInput(
                "\(name) requires a non-negative integer"
            )
        }
        return value
    }

    private static func runtimeWorkerOptions(
        blockedGoldenPath: String? = nil,
        forwardsWorkerStderr: Bool = false
    ) throws -> RuntimeWorkerOptions? {
        // The trusted binary has no in-process model target. Disabling the worker
        // therefore fails closed in every mode rather than selecting an editable
        // model path inside the timer/gate/score process.
        let officialRun = environmentValue("MLXFAST_OFFICIAL_BENCHMARK_RUN", fallback: "0") == "1"
        let enabled = environmentValue("MLXFAST_USE_RUNTIME_WORKER", fallback: "1")
        guard enabled != "0" && enabled.lowercased() != "false" else {
            throw MLXFastError.invalidInput(
                "mlxfast-swift requires the participant runtime worker; unset MLXFAST_USE_RUNTIME_WORKER"
            )
        }
        if officialRun, environmentValue("MLXFAST_NO_SANDBOX", fallback: "0") == "1" {
            throw MLXFastError.invalidInput(
                "official benchmark runs require the runtime worker sandbox; unset MLXFAST_NO_SANDBOX"
            )
        }
        let configuredExecutable = environmentValue(
            "MLXFAST_RUNTIME_WORKER_EXECUTABLE",
            fallback: ""
        )
        let executable = configuredExecutable.isEmpty
            ? try siblingParticipantWorkerExecutablePath()
            : configuredExecutable
        let executablePath: String
        if executable.hasPrefix("/") {
            executablePath = executable
        } else {
            executablePath = URL(
                fileURLWithPath: executable,
                relativeTo: URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            ).standardizedFileURL.path
        }
        guard FileManager.default.isExecutableFile(atPath: executablePath) else {
            throw MLXFastError.invalidInput(
                "participant runtime worker is not executable at \(executablePath)"
            )
        }
        // The metallib the worker will load must correspond to the vendored
        // Metal sources on disk: official runs fail closed on any stale or
        // unverifiable metallib (a cached artifact must never mask kernel
        // edits); local runs warn so an edit-loop checkout that predates the
        // fingerprint sidecar keeps working until ./setup.sh is rerun.
        try enforceMetallibFingerprint(
            workerExecutablePath: executablePath,
            officialRun: officialRun
        )
        // The editable-surface static review — the kernel-bypass LLM judge and
        // the deterministic byte-cap backstop that mirrored it — is owned by the
        // benchd benchmarker now, along with the `benchmark.json` contract that
        // declared the editable surface. Neither is present in this engine tree,
        // so no byte-budget check is enforced at worker launch here.
        var sandboxProfile = environmentValue("MLXFAST_RUNTIME_WORKER_SANDBOX_PROFILE", fallback: "")
        if sandboxProfile.isEmpty,
           environmentValue("MLXFAST_NO_SANDBOX", fallback: "0") != "1",
           let blockedGoldenPath,
           !blockedGoldenPath.isEmpty
        {
            sandboxProfile = try writeRuntimeWorkerSandboxProfile(
                blockedGoldenPath: blockedGoldenPath,
                allowedExecutablePath: executablePath
            )
        }
        if officialRun, sandboxProfile.isEmpty {
            throw MLXFastError.invalidInput(
                "official benchmark runs require a runtime worker sandbox profile; none was configured or derivable"
            )
        }
        return RuntimeWorkerOptions(
            executablePath: executablePath,
            sandboxProfilePath: sandboxProfile.isEmpty ? nil : sandboxProfile,
            // Fail closed: live worker-stderr forwarding is a local-edit-loop
            // convenience only. Official runs keep today's behavior where
            // worker stderr surfaces solely through the sanitized exit
            // diagnostic, so submitted code cannot stream hidden-prompt
            // content into CI logs.
            forwardsWorkerStderr: forwardsWorkerStderr && !officialRun
        )
    }

    /// Verify the metallib next to the participant worker against the
    /// vendored Metal sources before any worker is spawned. Official runs
    /// fail closed on a stale, missing, or unverifiable metallib; local runs
    /// warn and continue (the sidecar may simply predate this check).
    private static func enforceMetallibFingerprint(
        workerExecutablePath: String,
        officialRun: Bool
    ) throws {
        let configuredMetallib = environmentValue("MLXFAST_MLX_METALLIB", fallback: "")
        let metallibPath = configuredMetallib.isEmpty
            ? URL(fileURLWithPath: workerExecutablePath)
                .deletingLastPathComponent()
                .appendingPathComponent("mlx.metallib").path
            : absolutePath(configuredMetallib)
        let cmlxRoot = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(VendoredMetalFingerprint.defaultCmlxRelativePath)
            .path
        switch verifyMetallibFingerprintRecord(
            metallibPath: metallibPath,
            cmlxRoot: cmlxRoot
        ) {
        case .verified:
            return
        case .skipped(let reason):
            if officialRun {
                throw MLXFastError.invalidInput(
                    "official benchmark runs require the metallib fingerprint check; "
                        + reason
                )
            }
        case .mismatch(let reason):
            if officialRun {
                throw MLXFastError.invalidInput(
                    "refusing to spawn the participant worker: " + reason
                )
            }
            FileHandle.standardError.write(Data(
                ("mlxfast-swift: warning: " + reason + "\n").utf8
            ))
        }
    }

    private static func siblingParticipantWorkerExecutablePath() throws -> String {
        // The participant worker builds under its own SwiftPM scratch root, so
        // a trusted binary at <root>/.build/<config>/mlxfast-swift finds its
        // worker at <root>/.build-worker/<config>/mlxfast-runtime-worker. The
        // worker-root twin wins over a same-directory sibling so a stale
        // pre-split worker lingering next to the trusted binary is never
        // silently preferred over the current worker build.
        let executableDirectory = URL(fileURLWithPath: try currentExecutablePath())
            .deletingLastPathComponent()
        var workerRootComponents = executableDirectory.pathComponents
        if let buildIndex = workerRootComponents.lastIndex(of: ".build") {
            workerRootComponents[buildIndex] = ".build-worker"
            let workerTwin = URL(
                fileURLWithPath: NSString.path(withComponents: workerRootComponents)
            ).appendingPathComponent("mlxfast-runtime-worker").path
            if FileManager.default.isExecutableFile(atPath: workerTwin) {
                return workerTwin
            }
            let sibling = executableDirectory
                .appendingPathComponent("mlxfast-runtime-worker").path
            if FileManager.default.isExecutableFile(atPath: sibling) {
                return sibling
            }
            // Neither exists; report the canonical worker-root location.
            return workerTwin
        }
        return executableDirectory
            .appendingPathComponent("mlxfast-runtime-worker")
            .path
    }

    private static func currentExecutablePath() throws -> String {
        if let executableURL = Bundle.main.executableURL {
            let path = executableURL.standardizedFileURL
                .resolvingSymlinksInPath().path
            if FileManager.default.isExecutableFile(atPath: path) {
                return path
            }
        }

        var requiredSize: UInt32 = 0
        _ = _NSGetExecutablePath(nil, &requiredSize)
        if requiredSize > 0 {
            var buffer = [CChar](
                repeating: 0,
                count: Int(requiredSize)
            )
            if _NSGetExecutablePath(&buffer, &requiredSize) == 0 {
                let executableBytes = buffer
                    .prefix { $0 != 0 }
                    .map { UInt8(bitPattern: $0) }
                let path = URL(
                    fileURLWithPath: String(
                        decoding: executableBytes,
                        as: UTF8.self
                    )
                ).standardizedFileURL.resolvingSymlinksInPath().path
                if FileManager.default.isExecutableFile(atPath: path) {
                    return path
                }
            }
        }

        if let rawExecutable = CommandLine.arguments.first,
           !rawExecutable.isEmpty
        {
            if rawExecutable.contains("/") {
                let path = absolutePath(rawExecutable)
                if FileManager.default.isExecutableFile(atPath: path) {
                    return path
                }
            } else {
                let searchPath = ProcessInfo.processInfo.environment[
                    "PATH"
                ] ?? ""
                for directory in searchPath.split(
                    separator: ":",
                    omittingEmptySubsequences: false
                ) {
                    let root = directory.isEmpty
                        ? FileManager.default.currentDirectoryPath
                        : String(directory)
                    let path = URL(fileURLWithPath: root)
                        .appendingPathComponent(rawExecutable).path
                    if FileManager.default.isExecutableFile(atPath: path) {
                        return URL(fileURLWithPath: path)
                            .standardizedFileURL
                            .resolvingSymlinksInPath().path
                    }
                }
            }
        }

        throw MLXFastError.invalidInput(
            "mlxfast-swift could not resolve its actual executable path "
                + "from Bundle.main, _NSGetExecutablePath, argv[0], or PATH"
        )
    }

    // Confine the `transform` and `attach-gpqa-gates` command paths behind a
    // Seatbelt profile before they touch any input. Unlike `correctness`/
    // `benchmark`, these subcommands run the submission-built binary directly
    // (they do not spawn the separately sandboxed runtime worker), so on the
    // ranked box they execute as an UNSANDBOXED bench parent that reads the raw
    // hidden golden + GPQA answer key. This re-executes the current process
    // under `/usr/bin/sandbox-exec` with a profile that denies network,
    // process-fork, process-exec (of anything but this binary), and DNS
    // resolver mach-lookup -- the same guarantees the retired run-offline.sh
    // wrapper gave the transform, plus the mDNSResponder mach-lookup deny the
    // operator worker profile also carries. Reads/writes stay default-allowed
    // (transform legitimately reads the reference checkpoint and writes
    // weights/; a read allowlist would break dyld/Metal/tokenizer loading), so
    // the uid, workspace-write-confinement, and PF-egress layers remain the
    // filesystem boundary.
    //
    // Trigger + fail-closed policy: the trusted workflow sets
    // MLXFAST_SANDBOX_PARENT_TOOLS=1 on exactly the transform/attach steps (and
    // MLXFAST_OFFICIAL_BENCHMARK_RUN=1 also arms it). When armed, a missing
    // sandbox-exec or MLXFAST_NO_SANDBOX=1 aborts the run rather than executing
    // unsandboxed. Local invocations set neither, so participant workflows are
    // unchanged. MLXFAST_PARENT_SANDBOX_ACTIVE=1 is set on the re-exec so the
    // sandboxed child does not recurse.
    private static func reexecUnderParentToolSandboxIfRequested(subcommand: String) throws {
        if environmentValue("MLXFAST_PARENT_SANDBOX_ACTIVE", fallback: "0") == "1" {
            return
        }
        let officialRun = environmentValue("MLXFAST_OFFICIAL_BENCHMARK_RUN", fallback: "0") == "1"
        let requested = officialRun
            || environmentValue("MLXFAST_SANDBOX_PARENT_TOOLS", fallback: "0") == "1"
        guard requested else {
            return
        }
        if environmentValue("MLXFAST_NO_SANDBOX", fallback: "0") == "1" {
            throw MLXFastError.invalidInput(
                "\(subcommand) in a benchmark context requires the parent-tool sandbox; unset MLXFAST_NO_SANDBOX"
            )
        }
        let sandboxExecutable = "/usr/bin/sandbox-exec"
        guard FileManager.default.isExecutableFile(atPath: sandboxExecutable) else {
            throw MLXFastError.invalidInput(
                "\(subcommand) in a benchmark context requires sandbox-exec for the parent-tool sandbox"
            )
        }
        let executablePath = try currentExecutablePath()
        guard FileManager.default.isExecutableFile(atPath: executablePath) else {
            throw MLXFastError.invalidInput(
                "\(subcommand) parent-tool sandbox resolved a non-executable self path: \(executablePath)"
            )
        }
        let profilePath = try writeParentToolSandboxProfile(allowedExecutablePath: executablePath)
        let argv = [sandboxExecutable, "-f", profilePath, executablePath]
            + Array(CommandLine.arguments.dropFirst())
        setenv("MLXFAST_PARENT_SANDBOX_ACTIVE", "1", 1)
        var cArgs: [UnsafeMutablePointer<CChar>?] = argv.map { strdup($0) }
        cArgs.append(nil)
        defer {
            for pointer in cArgs {
                if let pointer {
                    free(pointer)
                }
            }
        }
        _ = sandboxExecutable.withCString { pathPointer in
            execv(pathPointer, cArgs)
        }
        // execv only returns on failure.
        throw MLXFastError.invalidInput(
            "\(subcommand) failed to re-exec under sandbox-exec (errno=\(errno))"
        )
    }

    private static func writeParentToolSandboxProfile(
        allowedExecutablePath: String
    ) throws -> String {
        let profileURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlxfast-parent-tool-\(UUID().uuidString).sb")
        let absoluteExecutablePath = absolutePath(allowedExecutablePath)
        let profile = """
        (version 1)
        (allow default)
        (deny network*)
        (deny process-fork)
        (deny process-exec*)
        (allow process-exec (literal "\(seatbeltEscaped(absoluteExecutablePath))"))
        (deny mach-lookup (global-name "com.apple.mDNSResponder"))
        (deny mach-lookup (global-name "com.apple.system.mDNSResponder"))
        (deny mach-lookup (global-name-prefix "com.apple.mDNSResponder"))
        """
        try profile.write(to: profileURL, atomically: true, encoding: .utf8)
        return profileURL.path
    }

    private static func writeRuntimeWorkerSandboxProfile(
        blockedGoldenPath: String,
        allowedExecutablePath: String
    ) throws -> String {
        let sandboxExecutable = "/usr/bin/sandbox-exec"
        guard FileManager.default.isExecutableFile(atPath: sandboxExecutable) else {
            throw MLXFastError.invalidInput("sandbox-exec not found for runtime worker sandbox")
        }
        let profileURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlxfast-runtime-worker-\(UUID().uuidString).sb")
        let absoluteGoldenPath = absolutePath(blockedGoldenPath)
        let absoluteExecutablePath = absolutePath(allowedExecutablePath)
        var deniedReadRules = [
            "(deny file-read* (literal \"\(seatbeltEscaped(absoluteGoldenPath))\"))",
        ]
        let privateDir = environmentValue("MLXFAST_PRIVATE_DIR", fallback: "")
        if !privateDir.isEmpty {
            deniedReadRules.append(
                "(deny file-read* (subpath \"\(seatbeltEscaped(absolutePath(privateDir)))\"))"
            )
        }
        // `(deny network*)` blocks the worker's OWN sockets, but getaddrinfo(3)
        // resolves via IPC to mDNSResponder, which egresses from ITS uid -- so a
        // uid/socket-scoped block never sees the DNS query and submitted code
        // could exfiltrate over DNS. Deny the worker's mach-lookup of the
        // resolver (canonical name, legacy alias, and the
        // com.apple.mDNSResponder.* family), matching the parent-tool profile in
        // writeParentToolSandboxProfile and the operator-layer worker profile
        // used on the ranked box. This keeps the harness-generated FALLBACK
        // profile (local runs / any path where the operator template is not
        // injected via MLXFAST_RUNTIME_WORKER_SANDBOX_PROFILE) resolver-safe too.
        let profile = """
        (version 1)
        (allow default)
        (deny network*)
        (deny process-fork)
        (deny process-exec*)
        (allow process-exec (literal "\(seatbeltEscaped(absoluteExecutablePath))"))
        (deny mach-lookup (global-name "com.apple.mDNSResponder"))
        (deny mach-lookup (global-name "com.apple.system.mDNSResponder"))
        (deny mach-lookup (global-name-prefix "com.apple.mDNSResponder"))
        (deny file-write*)
        (allow file-write* (literal "/dev/null"))
        \(deniedReadRules.joined(separator: "\n"))
        """
        try profile.write(to: profileURL, atomically: true, encoding: .utf8)
        return profileURL.path
    }

    private static func absolutePath(_ path: String) -> String {
        let url: URL
        if path.hasPrefix("/") {
            url = URL(fileURLWithPath: path)
        } else {
            url = URL(
                fileURLWithPath: path,
                relativeTo: URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            )
        }
        return url.standardizedFileURL.resolvingSymlinksInPath().path
    }

    private static func seatbeltEscaped(_ value: String) -> String {
        value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
    }

    private static func runCheckpointShards(_ options: ParsedOptions) throws {
        try options.validate(valueOptions: ["--index"])
        let indexPath = options.value(for: "--index", default: "")
        guard !indexPath.isEmpty else {
            throw MLXFastError.invalidInput("checkpoint-shards requires --index PATH")
        }
        for shard in try CheckpointIndexTools.safetensorShardNames(from: indexPath) {
            print(shard)
        }
    }

    private static func printUsage() {
        print(
            """
            Usage:
              mlxfast-swift transform [--reference PATH] [--output PATH]
              mlxfast-swift verify-transform [--reference PATH] [--weights PATH] [--tmp-parent PATH] [--max-bytes N]
              mlxfast-swift correctness [--weights PATH] [--golden PATH]
              mlxfast-swift correctness-trace [--weights PATH] [--golden PATH] [--case NAME] --step N [--top-k N]
              mlxfast-swift preflight [--weights PATH] [--golden PATH]
              mlxfast-swift benchmark [--local-submit|--local-iterate] [--weights PATH] [--golden PATH] [--score-path PATH]
              mlxfast-swift attach-gpqa-gates [--golden PATH] --gpqa PATH [--tokenizer PATH] [--output PATH] [--case-count N] [--max-new-tokens N]
              mlxfast-swift attach-free-run-gate [--golden PATH] [--weights PATH] [--output PATH] [--name NAME] [--steps N] [--allow-partial] [--case NAME | --prompt-file PATH [--tokenizer PATH]] [--exact-prefix N]
              mlxfast-swift attach-benchmark-oracle [--golden PATH] [--output PATH]
              mlxfast-swift generate-golden --prompt-file PATH [--weights PATH] [--tokenizer PATH] --output PATH --name NAME --steps N
              mlxfast-swift record-reference-tape --prompt-file PATH [--weights PATH] [--tokenizer PATH] --output PATH [--steps N] [--recording-backend cbv2|legacy] [--allow-short]
              mlxfast-swift analyze-ngram-similarity --golden PATH [--case NAME] [--orders 1,2,3] [--max-hit-rate RATE]
              mlxfast-swift generate-gpqa-answers --gpqa PATH [--weights PATH] [--tokenizer PATH] --output PATH [--case-count N] [--max-new-tokens N]
              mlxfast-swift checkpoint-shards --index PATH
              mlxfast-swift dflash-benchmark --drafter PATH --golden PATH [--weights PATH] [--block-size N] [--tokens N] [--schedule-seed N] [--output PATH]
              mlxfast-swift dflash-probe --drafter PATH --golden PATH [--weights PATH] [--tokens N] [--schedule-seed N] [--output PATH]
              mlxfast-swift dflash-reference --drafter PATH --emitted PATH --output PATH [--weights PATH]
              mlxfast-swift mtp-verify  (deferred: refuses; returns with the follow-up increment)

            Swift-only Gemma 4 26B A4B 4-bit harness entrypoint (the DFlash
            subcommands still drive the Laguna target and its pinned drafter).
            """
        )
    }

    private static func environmentValue(_ name: String, fallback: String) -> String {
        let value = ProcessInfo.processInfo.environment[name] ?? ""
        return value.isEmpty ? fallback : value
    }

    private static func defaultCorrectnessGoldenPath() -> String {
        if FileManager.default.fileExists(atPath: MLXFastConstants.defaultGoldenPath) {
            return MLXFastConstants.defaultGoldenPath
        }
        let publicPath = environmentValue(
            "MLXFAST_PUBLIC_CORRECTNESS_GOLDEN_PATH",
            fallback: MLXFastConstants.defaultPublicCorrectnessGoldenPath
        )
        if FileManager.default.fileExists(atPath: publicPath) {
            return publicPath
        }
        return MLXFastConstants.defaultGoldenPath
    }

    private static func trimmedNonEmpty(_ value: String?) -> String? {
        let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return trimmed.isEmpty ? nil : trimmed
    }

}

private struct NGramSimilarityAnalysisOutput: Codable {
    let targetID: String
    let source: String
    let maximumHitRate: Double
    let passed: Bool
    let report: NGramSelfSimilarityReport

    enum CodingKeys: String, CodingKey {
        case targetID = "target_id"
        case source
        case maximumHitRate = "maximum_hit_rate"
        case passed
        case report
    }
}

private struct GPQAReferenceDocument: Decodable {
    let cases: [GPQAReferenceCase]
}

private struct GPQAReferenceCase: Decodable {
    let id: String?
    let prompt: String
    let expectedResponse: String?
    let answerKey: String?
    let acceptedTokenSequences: [[Int]]?
    let acceptedResponses: [String]?
    let domain: String?
    let subdomain: String?

    enum CodingKeys: String, CodingKey {
        case id
        case prompt
        case expectedResponse = "expected_response"
        case answerKey = "answer_key"
        case acceptedTokenSequences = "accepted_token_sequences"
        case acceptedResponses = "accepted_responses"
        case domain
        case subdomain
    }

    var identifier: String {
        let trimmed = id?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return trimmed.isEmpty ? "gpqa-private" : trimmed
    }

}

private struct SemanticGPQAAnswerDocument: Encodable {
    let version: Int
    let cases: [SemanticGPQAAnswerCase]
}

private struct SemanticGPQAAnswerCase: Encodable {
    let id: String
    let domain: String?
    let subdomain: String?
    let prompt: String
    let answerKey: String?
    let referenceAnswer: String
    let candidateAnswer: String
    let candidateTokens: [Int]
    let maxNewTokens: Int

    enum CodingKeys: String, CodingKey {
        case id
        case domain
        case subdomain
        case prompt
        case answerKey = "answer_key"
        case referenceAnswer = "reference_answer"
        case candidateAnswer = "candidate_answer"
        case candidateTokens = "candidate_tokens"
        case maxNewTokens = "max_new_tokens"
    }
}

private struct ParsedOptions {
    private var values: [String: String] = [:]
    private var flags: Set<String> = []
    private var positionals: [String] = []
    private var duplicates: Set<String> = []

    init(_ arguments: [String]) {
        var index = 0
        while index < arguments.count {
            let argument = arguments[index]
            if argument.hasPrefix("--") {
                if let separator = argument.firstIndex(of: "=") {
                    let key = String(argument[..<separator])
                    let value = String(argument[argument.index(after: separator)...])
                    recordOption(key)
                    values[key] = value
                    index += 1
                } else if index + 1 < arguments.count && !arguments[index + 1].hasPrefix("--") {
                    recordOption(argument)
                    values[argument] = arguments[index + 1]
                    index += 2
                } else {
                    recordOption(argument)
                    flags.insert(argument)
                    index += 1
                }
            } else {
                positionals.append(argument)
                index += 1
            }
        }
    }

    private mutating func recordOption(_ name: String) {
        if values[name] != nil || flags.contains(name) {
            duplicates.insert(name)
        }
    }

    func value(for name: String, default defaultValue: String) -> String {
        values[name] ?? defaultValue
    }

    func hasFlag(_ name: String) -> Bool {
        flags.contains(name)
    }

    func validate(
        valueOptions: Set<String>,
        flagOptions: Set<String> = [],
        allowPositionals: Bool = false
    ) throws {
        if let duplicate = duplicates.first {
            throw MLXFastError.invalidInput("duplicate option \(duplicate)")
        }
        for name in values.keys where !valueOptions.contains(name) {
            throw MLXFastError.invalidInput("unknown option \(name)")
        }
        for (name, value) in values where value.isEmpty {
            throw MLXFastError.invalidInput("\(name) requires a non-empty value")
        }
        for flag in flags {
            if valueOptions.contains(flag) {
                throw MLXFastError.invalidInput("\(flag) requires a value")
            }
            if !flagOptions.contains(flag) {
                throw MLXFastError.invalidInput("unknown option \(flag)")
            }
        }
        if !allowPositionals, let positional = positionals.first {
            throw MLXFastError.invalidInput("unexpected argument \(positional)")
        }
    }
}
