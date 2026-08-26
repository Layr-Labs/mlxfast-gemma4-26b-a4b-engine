import Darwin
import Foundation
import MLXFastCore
import MLXFastModel
import MLXFastRuntimeWorkerSupport

let exitCode = ParticipantWorkerCLI.run(
    arguments: Array(CommandLine.arguments.dropFirst())
)
exit(Int32(exitCode))

private enum ParticipantWorkerCLI {
    static func run(arguments: [String]) -> Int {
        do {
            guard let command = arguments.first else {
                printUsage()
                return 0
            }
            if ["help", "--help", "-h"].contains(command) {
                printUsage()
                return 0
            }
            if Array(arguments.dropFirst()) == ["--help"]
                || Array(arguments.dropFirst()) == ["-h"]
            {
                printUsage()
                return 0
            }
            let options = try WorkerOptions(
                Array(arguments.dropFirst())
            )
            switch command {
            case "runtime-worker":
                // The generic benchd-facing worker. `--weights` is required.
                //
                // `--mtp-head <DIR>` RESTORED 2026-08-25 (it went with the MTP
                // arm 2026-08-22; the arm is back and so is its spawn channel).
                // benchd's measure-job spawns EVERY worker leg
                // `runtime-worker --weights <W> --mtp-head <H>
                // [--speculative-protocol v1.1]` (benchd @ c2327d15,
                // measure_job.rs `timed_leg_base_args` / `leg_spawn_args`,
                // fenced by `RUNTIME_WORKER_ACCEPTED_FLAGS`), so refusing the
                // flag killed every leg pre-hello. The accepted option surface
                // is the shared `runtimeWorkerAcceptedOptionFlags` constant —
                // pinned equal to benchd's fence — and `requireOnly` still
                // refuses anything outside it: an unknown option is a hard
                // error before the hello, never a silently ignored flag.
                //
                // ARGV-ONLY: the head directory arrives exclusively via this
                // flag. The engine does not read `QMTP_HEAD_DIR` /
                // `QMTP_CANDIDATE_HEAD_DIR` — those are benchd's OWN inputs,
                // resolved benchd-side into this flag's value, and benchd's
                // allowlisted child env drops them before the worker starts
                // (see `runtimeWorkerMTPHeadFlag`'s doc comment). When the
                // flag is absent the worker keeps the CWD `./mtp-head/`
                // staging default (the native trusted CLI's flow); when
                // present the head loads from exactly the named directory,
                // fail-closed if it is not a loadable head.
                try options.requireOnly(
                    values: runtimeWorkerAcceptedOptionFlags
                )
                let weightsPath = options.value(
                    for: "--weights",
                    default: ProcessInfo.processInfo.environment[
                        "MLXFAST_WEIGHTS_PATH"
                    ] ?? MLXFastConstants.defaultWeightsPath
                )
                let advertise = try runtimeWorkerAdvertisesSpeculativeProtocol(
                    flagValue: options.value(
                        for: runtimeWorkerSpeculativeProtocolFlag
                    )
                )
                try Gemma4Runtime.runWorker(
                    weightsPath: weightsPath,
                    mtpHeadPath: options.optionalValue(
                        for: runtimeWorkerMTPHeadFlag
                    ),
                    // `--dflash-head <DIR>` (David ruling 2026-08-26) — the
                    // DFlash drafter's own PER-LEG channel, added when DFlash
                    // became a first-class scored mode. Same ARGV-ONLY rule as
                    // `--mtp-head`: benchd resolves QMTP_DFLASH_HEAD_DIR /
                    // QMTP_CANDIDATE_DFLASH_HEAD_DIR into this flag's value and
                    // passes the PINNED drafter to the serial control leg and
                    // the candidate's own to the candidate leg. Absent keeps
                    // the CWD `./dflash-head/` default, so an MTP-only spawn is
                    // unchanged; present loads from exactly that directory,
                    // fail-closed.
                    dflashHeadPath: options.optionalValue(
                        for: runtimeWorkerDFlashHeadFlag
                    ),
                    advertisesSpeculativeProtocol: advertise
                )

            case "dflash-runtime-worker":
                // DFLASH ARM DEFERRED (2026-08-22, vendored adoption of
                // mlx-swift-lm main). The verb is RETAINED and REFUSES rather
                // than being deleted, because the arm is deferred, not
                // abandoned: keeping the name reserved means the follow-up lane
                // restores behaviour behind an entry point that already exists,
                // and any caller still invoking it gets told what happened
                // instead of "unknown verb".
                //
                // What went: upstream deleted the v1 batching/compiled-decode
                // engine at ffede00, and the vendored DFlash runtime
                // (Libraries/MLXSpeculative, DFlashTarget, DFlashVerifyLinear)
                // is written against it -- BatchKVCache specifically. Retaining
                // that substrate would have meant carrying the v1 engine
                // forward, which the adoption ruling excluded. The trusted
                // DFlash driver/protocol logic (Gemma4RuntimeDFlash.swift,
                // Gemma4RuntimeDFlashDriver.swift) is MLX-free and STAYS in
                // tree, so only the model-side worker is missing.
                //
                // See docs/gemma4-port-notes.md OQ-3 for the follow-up lane's
                // obligations -- in particular re-deriving the sliding-window
                // rollback seam fix on CBv2 rather than assuming CBv2 never
                // had the bug.
                throw MLXFastError.invalidInput(
                    "dflash-runtime-worker is not runnable on this engine: the "
                        + "DFlash arm is deferred pending its port onto the "
                        + "CBv2 engine. See docs/gemma4-port-notes.md OQ-3."
                )

            case "mtp-runtime-worker":
                // MTP ARM DEFERRED (2026-08-22, Gemma 4 26B A4B harness port).
                // The verb is RETAINED and REFUSES, exactly like
                // `dflash-runtime-worker` above: the arm is deferred, not
                // abandoned, so the name stays reserved and a caller still
                // invoking it is told what happened instead of "unknown verb".
                //
                // What went: the speculative surface was written against the
                // Qwen tower end to end -- `Qwen36MTPTarget` conformed
                // `Qwen35TextModel` and `MLXLLM.Qwen35Model` and nothing else,
                // `Qwen36MTPHeadAttachment` merged a Qwen head into a Qwen
                // backbone, and the block session's cache reasoning is the
                // gated-delta tower's. None of that transfers to Gemma 4's
                // 25-sliding/5-global stack by renaming, so it was deleted
                // rather than left compiling-but-unrunnable.
                //
                // See docs/gemma4-port-notes.md section 8 for the follow-up
                // increment's obligations.
                throw MLXFastError.invalidInput(
                    "mtp-runtime-worker is not runnable on this engine: the "
                        + "MTP arm lands with the Gemma harness port's "
                        + "follow-up increment. See "
                        + "docs/gemma4-port-notes.md section 8."
                )

            case "width-probe":
                // OPERATOR-ONLY diagnostic (exactness round three,
                // 2026-08-25): forward-width divergence localization.
                // Teacher-forces a pinned reference tape's own chain through
                // width-1, width-L window, and batch-width forwards over the
                // SAME tokens, bit-comparing per-layer K/V, MoE router
                // scores/selections, and final logits to report the FIRST
                // divergent tensor per position — the (i) LM-head-only vs
                // (ii) mid-network-router question the exactness-anchor
                // design decision hinges on. Never spawned by benchd (its
                // argv fence covers only `runtime-worker`); never on a wire
                // or scored path. See Gemma4RuntimeWidthProbe.swift.
                let weights = options.value(for: "--weights")
                let tapePath = options.value(for: "--tape")
                guard !weights.isEmpty, !tapePath.isEmpty else {
                    throw MLXFastError.invalidInput(
                        "width-probe requires --weights <transformed-weights> "
                            + "and --tape <timed-prompt-tape.json>")
                }
                let steps = Int(options.value(for: "--steps", default: "24")) ?? 24
                let widths = options.value(for: "--widths", default: "2,3")
                    .split(separator: ",").compactMap { Int($0) }
                let batchValue = options.value(for: "--batch", default: "8")
                let batch = Int(batchValue).flatMap { $0 >= 2 ? $0 : nil }
                let outputPath = options.optionalValue(for: "--out")
                // Phase-1 fidelity-gate calibration knobs. --topk defaults to 2
                // (the pre-extension top-2 behavior); set e.g. 16 to record the
                // top-N ranked reference logits + the within-envelope depth.
                // --rel-envelope is the relative near-tie threshold the box run
                // sweeps (default 0.05). Both drive the reference-side (post-
                // softcap) characterization only; nothing scored or armed.
                let logitTopK = max(
                    1, Int(options.value(for: "--topk", default: "2")) ?? 2)
                let relEnvelope = Double(
                    options.value(for: "--rel-envelope", default: "0.05")) ?? 0.05
                try Gemma4Runtime.runWidthProbe(
                    WidthProbeOptions(
                        weightsPath: weights,
                        tapePath: tapePath,
                        steps: steps,
                        windowWidths: widths,
                        batchWidth: batch,
                        outputPath: outputPath,
                        logitTopK: logitTopK,
                        relEnvelope: relEnvelope))
                return 0

            case "preflight":
                try options.requireOnly(
                    values: ["--weights"]
                )
                let weightsPath = options.value(
                    for: "--weights",
                    default: ProcessInfo.processInfo.environment[
                        "MLXFAST_WEIGHTS_PATH"
                    ] ?? MLXFastConstants.defaultWeightsPath
                )
                try Gemma4Runtime.runPreflightWorker(
                    weightsPath: weightsPath
                )

            default:
                throw MLXFastError.invalidInput(
                    "unknown participant worker command '\(command)'"
                )
            }
            return 0
        } catch {
            fputs("mlxfast-runtime-worker: \(error)\n", stderr)
            return 1
        }
    }

    private static func printUsage() {
        print(
            """
            Usage:
              mlxfast-runtime-worker runtime-worker [--weights PATH] [--mtp-head DIR] [--speculative-protocol v1.1]
              mlxfast-runtime-worker dflash-runtime-worker  (deferred: refuses)
              mlxfast-runtime-worker mtp-runtime-worker     (deferred: refuses)
              mlxfast-runtime-worker preflight [--weights PATH]

            --mtp-head DIR loads the assistant head from DIR (fail-closed when DIR
            is not a loadable head). Omit it to use the CWD ./mtp-head/ staging
            default, where an absent head is the normal serial-only case.

            --speculative-protocol v1.1 opts the hello into the v1.1 speculative
            surface (spec_modes / capabilities / head_provenance + effective_spec
            echoes). Omit it for the v1-only surface the native trusted CLI expects.

            Participant-side MLX runtime worker for mlxfast-swift.
            """
        )
    }
}

private struct WorkerOptions {
    private let values: [String: String]

    init(_ arguments: [String]) throws {
        var values: [String: String] = [:]
        var index = 0
        while index < arguments.count {
            let option = arguments[index]
            guard option.hasPrefix("--") else {
                throw MLXFastError.invalidInput(
                    "unexpected participant worker argument '\(option)'"
                )
            }
            guard values[option] == nil else {
                throw MLXFastError.invalidInput(
                    "duplicate participant worker option \(option)"
                )
            }
            guard index + 1 < arguments.count else {
                throw MLXFastError.invalidInput(
                    "participant worker option \(option) requires a value"
                )
            }
            values[option] = arguments[index + 1]
            index += 2
        }
        self.values = values
    }

    func value(for option: String, default defaultValue: String = "") -> String {
        values[option] ?? defaultValue
    }

    /// Present-vs-absent, preserved: `--mtp-head`'s two channels (explicit
    /// directory vs CWD staging default) hang on exactly this distinction,
    /// which `value(for:default:)`'s empty-string default would erase.
    func optionalValue(for option: String) -> String? {
        values[option]
    }

    func requireOnly(values allowedValues: Set<String>) throws {
        if let unexpected = Set(values.keys).subtracting(allowedValues).sorted().first {
            throw MLXFastError.invalidInput(
                "unexpected participant worker option \(unexpected)"
            )
        }
    }
}
