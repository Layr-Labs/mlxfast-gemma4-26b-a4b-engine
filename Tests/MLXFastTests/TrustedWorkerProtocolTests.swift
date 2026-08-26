import Foundation
@testable import MLXFastCore
@testable import MLXFastHarness
import Testing

@Test
func trustedPreflightUsesOneShotWorkerProtocolAndPropagatesFailure() throws {
    let root = try trustedWorkerTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let argumentLog = root.appendingPathComponent("arguments.txt")
    let successWorker = root.appendingPathComponent("success-worker")
    try writeExecutable(
        """
        #!/bin/bash
        printf '%s\n' "$*" > \(shellQuote(argumentLog.path))
        printf '%s\n' '{"ok":true}'
        """,
        to: successWorker
    )
    try Gemma4Runtime.runPreflightWithWorker(
        weightsPath: "/tmp/weights",
        worker: RuntimeWorkerOptions(
            executablePath: successWorker.path,
            requestTimeoutSeconds: 5
        )
    )
    let arguments = try String(
        contentsOf: argumentLog,
        encoding: .utf8
    )
    #expect(arguments.contains("preflight --weights /tmp/weights"))

    let failingWorker = root.appendingPathComponent("failing-worker")
    try writeExecutable(
        """
        #!/bin/bash
        printf '%s\n' '{"ok":false,"error":"invalid tensor metadata"}'
        exit 1
        """,
        to: failingWorker
    )
    #expect(throws: MLXFastError.self) {
        try Gemma4Runtime.runPreflightWithWorker(
            weightsPath: "/tmp/weights",
            worker: RuntimeWorkerOptions(
                executablePath: failingWorker.path,
                requestTimeoutSeconds: 5
            )
        )
    }
}

@Test
func trustedTracePreservesLargeTopKAndOutOfSubsetExpectedDiagnostics() throws {
    let root = try trustedWorkerTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let requestLog = root.appendingPathComponent("request.json")
    let worker = root.appendingPathComponent("trace-worker")
    let topLogits = (0..<12).map {
        #"{"token":\#($0),"logit":\#(100 - $0)}"#
    }.joined(separator: ",")
    try writeExecutable(
        """
        #!/bin/bash
        printf '%s\n' '{"id":0,"nonce":"trace-nonce","ok":true}'
        IFS= read -r request
        printf '%s\n' "$request" > \(shellQuote(requestLog.path))
        printf '%s\n' '{"id":1,"nonce":"trace-nonce","ok":true,"token":0,"top_logits":[\(topLogits)],"expected_token_logit":1,"expected_token_rank":20,"top_logit_margin":1}'
        """,
        to: worker
    )

    let prompt = Array(
        repeating: 1,
        count: MLXFastConstants.correctnessPromptTokens
    )
    var expected = Array(
        repeating: 19,
        count: MLXFastConstants.correctnessSteps
    )
    expected[0] = 19
    let golden = root.appendingPathComponent("golden.json")
    try """
    {
      "version": 1,
      "model_type": "gemma4_text",
      "cases": [
        {
          "name": "trace",
          "prompt_tokens": \(jsonArray(prompt)),
          "expected_tokens": \(jsonArray(expected))
        }
      ]
    }
    """.write(to: golden, atomically: true, encoding: .utf8)

    let report = try Gemma4Runtime.traceCorrectness(
        CorrectnessTraceOptions(
            weightsPath: "/tmp/weights",
            goldenPath: golden.path,
            caseName: "trace",
            step: 0,
            topK: 12
        ),
        worker: RuntimeWorkerOptions(
            executablePath: worker.path,
            requestTimeoutSeconds: 5
        )
    )
    #expect(report.topLogits.count == 12)
    #expect(report.expectedToken == 19)
    #expect(report.expectedTokenLogit == 1)
    #expect(report.expectedTokenRank == 20)
    #expect(!report.topLogits.contains { $0.token == 19 })

    let request = try String(
        contentsOf: requestLog,
        encoding: .utf8
    )
    #expect(request.contains(#""top_k":12"#))
    #expect(request.contains(#""expected_token":19"#))
}

@Test
func trustedSandboxRebindsExecPermissionToTheRuntimeWorker() throws {
    let sandboxExecutable = "/usr/bin/sandbox-exec"
    guard FileManager.default.isExecutableFile(atPath: sandboxExecutable) else {
        return
    }
    let root = try trustedWorkerTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let staleExecutable = "/usr/bin/false"
    let runtimeWorkerExecutable = "/usr/bin/true"
    let staleProfile = root.appendingPathComponent("stale-worker.sb")
    try """
    (version 1)
    (allow default)
    (deny network*)
    (deny process-fork)
    (deny process-exec*)
    (allow process-exec (literal "\(staleExecutable)"))
    """.write(to: staleProfile, atomically: true, encoding: .utf8)

    let reboundProfilePath = try runtimeWorkerSandboxProfile(
        rebinding: staleProfile.path,
        toExecutableAt: runtimeWorkerExecutable
    )
    defer { try? FileManager.default.removeItem(atPath: reboundProfilePath) }
    let reboundProfile = try String(
        contentsOfFile: reboundProfilePath,
        encoding: .utf8
    )
    #expect(reboundProfile.contains("(deny network*)"))
    #expect(!reboundProfile.contains(
        "(allow process-exec (literal \"\(staleExecutable)\"))"
    ))
    #expect(reboundProfile.hasSuffix(
        """
        (deny process-exec*)
        (allow process-exec (literal "\(runtimeWorkerExecutable)"))
        """
    ))

    let allowed = Process()
    allowed.executableURL = URL(fileURLWithPath: sandboxExecutable)
    allowed.arguments = ["-f", reboundProfilePath, runtimeWorkerExecutable]
    allowed.standardError = Pipe()
    try allowed.run()
    allowed.waitUntilExit()
    #expect(allowed.terminationStatus == 0)

    let denied = Process()
    denied.executableURL = URL(fileURLWithPath: sandboxExecutable)
    denied.arguments = ["-f", reboundProfilePath, staleExecutable]
    denied.standardError = Pipe()
    try denied.run()
    denied.waitUntilExit()
    // sandbox-exec returns EX_OSERR (71) when its profile rejects exec. If the
    // stale allow still won, /usr/bin/false would run and return 1 instead.
    #expect(denied.terminationStatus == 71)
}

// Revert-proof for the wrapper adoption at the RANKED BENCHMARK WORKER path.
//
// That branch is deliberately the one this test binds to, because it is the one
// with nothing behind it: `benchmarkWithWorker` skips `BenchmarkPreflight.check`
// on purpose (it would run editable model code in the trusted, unsandboxed
// parent), and the `checkWorkerBenchmarkInputs` call above the load is
// existence-only -- it stats config.json, the safetensors index and the golden
// and reads none of them. So `loadQwenGoldenFixture` there is the ONLY thing
// standing between a foreign-model golden and a scored run.
//
// It is NOT bound to the correctness path on purpose either: those call sites
// have `BenchmarkPreflight.checkCorrectnessArtifacts` immediately behind them,
// which already loads through the Qwen wrapper, so a correctness-side assertion
// would still pass with the wrapper reverted -- vacuous as a revert-proof.
@Test
func rankedWorkerBenchmarkRejectsAForeignModelGoldenBeforeTheWorkerStarts() throws {
    let root = try trustedWorkerTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    // Only what checkWorkerBenchmarkInputs stats: it never opens these.
    let weights = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    for name in ["config.json", "model.safetensors.index.json"] {
        try "{}".write(
            to: weights.appendingPathComponent(name),
            atomically: true,
            encoding: .utf8
        )
    }

    // Records the fact that the worker process was started at all, then exits so
    // the client fails fast instead of waiting out the hello timeout.
    let startLog = root.appendingPathComponent("worker-started.txt")
    let workerExecutable = root.appendingPathComponent("worker")
    try writeExecutable(
        """
        #!/bin/bash
        printf 'started\n' >> \(shellQuote(startLog.path))
        exit 1
        """,
        to: workerExecutable
    )

    // One golden authored twice, identical in every field except `model_type`,
    // so nothing but the model identity can separate the two runs. The oracle is
    // derived so the document gets past the "must contain a benchmark oracle"
    // guard that follows the load.
    func benchmark(modelType: String) throws -> ScorePayload {
        let document = try goldenDocumentAttachingDerivedBenchmarkOracle(
            GoldenDocument(
                modelType: modelType,
                cases: [
                    GoldenCase(
                        name: "worker-path-model-identity",
                        promptTokens: Array(
                            repeating: 11,
                            count: MLXFastConstants.correctnessPromptTokens
                        ),
                        expectedTokens: (0..<(MLXFastConstants.benchmarkDecodeSteps * 2))
                            .map { 900 + $0 }
                    )
                ],
                benchmark: nil
            )
        )
        let goldenPath = root.appendingPathComponent("golden-\(modelType).json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(document).write(to: goldenPath)
        return Gemma4Runtime.benchmark(
            BenchmarkOptions(
                weightsPath: weights.path,
                goldenPath: goldenPath.path
            ),
            worker: RuntimeWorkerOptions(
                executablePath: workerExecutable.path,
                helloTimeoutSeconds: 5,
                requestTimeoutSeconds: 5
            )
        )
    }

    let foreign = try benchmark(modelType: "gemma_text")
    #expect(foreign.passed == false)
    #expect(foreign.score == nil)
    // Asserted as the reference's exact string, not merely "something failed":
    // with the wrapper reverted to the model-agnostic loader this golden LOADS,
    // and the run walks on to the worker with a different error entirely.
    #expect(
        foreign.metrics.error
            == "correctness golden file model_type=Optional(\"gemma_text\") "
                + "expected gemma4_text"
    )
    #expect(!FileManager.default.fileExists(atPath: startLog.path))

    // Control: the same document under the pinned identity gets PAST the loader
    // and does start the worker. So the rejection above is the identity check
    // deciding, not some unrelated defect in the fixture.
    let qwen = try benchmark(modelType: MLXFastConstants.requiredGoldenModelType)
    #expect(qwen.passed == false)
    #expect(qwen.metrics.error != foreign.metrics.error)
    #expect(FileManager.default.fileExists(atPath: startLog.path))
}

private func trustedWorkerTemporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(
        at: url,
        withIntermediateDirectories: true
    )
    return url
}

private func writeExecutable(_ contents: String, to url: URL) throws {
    try contents.write(to: url, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: url.path
    )
}

private func shellQuote(_ value: String) -> String {
    "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
}

private func jsonArray(_ values: [Int]) -> String {
    "[\(values.map(String.init).joined(separator: ","))]"
}
