// PROMPT-GLUE3 (pg3): the prompt router's finalists32 + weights kernel in
// issue-bound-optimal form, output words unchanged.
//
// On the prompt pass (8 x 1024 tokens) every streaming glue kernel now runs
// at the roof of its traffic pattern; the one prompt-plane kernel left with
// headroom moves almost no bytes at all:
//   * `gemma4_router_finalists32_weights_bf16_v1` (0.14 ms per layer, 18 GB/s)
//     is bound by the instruction count of its compare-exchange step, not by
//     its bytes, its barriers or its threadgroup count: the first selection
//     network alone is 80% of the dispatch, and every one of its 15 steps
//     re-decodes two bf16 scores, tests both for NaN, branches and compares
//     twice (`gemma4_finalists_before`). Rows per threadgroup and a
//     barrier-free tail each measured within a few percent of the incumbent.
// The twin in this arm keeps the incumbent's networks step for step and its
// softmax tail statement for statement; it changes the FORM of the
// compare-exchange decision (one unsigned compare of a per-item key computed
// once at load, see the kernel) and runs the four group networks interleaved
// on one simdgroup, so nothing waits on threadgroup memory. Admitted only for
// prompt-width score planes (`minRows` token rows and up); decode and the
// verify rectangles keep the incumbent dispatch byte for byte.
//
// Kill switch: `DARKBLOOM_GEMMA4_PROMPT_GLUE3=0` (the incumbent dispatch
// unchanged). Engage mark: `prompt-glue3`.
// `DARKBLOOM_GEMMA4_PROMPT_GLUE3_XCHECK=1` evaluates the incumbent kernel
// beside every pg3 dispatch and counts differing words per output
// (diagnostic only; forces evaluation).

import Foundation
import MLX

public enum Gemma4PromptGlue3V1 {
    public static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_PROMPT_GLUE3"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// Token-row floor. The scored prompt plane carries 8192 token rows;
    /// decode and verify rectangles carry 8..256.
    public static let minRows: Int = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_PROMPT_GLUE3_MIN_ROWS"],
            let value = Int(raw), value > 0
        else { return 1024 }
        return value
    }()

    public static let xcheck: Bool =
        ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_PROMPT_GLUE3_XCHECK"] == "1"

    public static func mark() {
        CBv2EngageMark.once("prompt-glue3")
    }

    // MARK: - diagnostics (never on a timed run)

    /// Counts words that differ between a pg3 kernel's output and the
    /// incumbent's, evaluating both.
    public static func report(_ candidate: MLXArray, reference: MLXArray, site: String) {
        guard candidate.shape == reference.shape, candidate.dtype == reference.dtype else {
            FileHandle.standardError.write(
                Data(
                    ("[xcheck] prompt-glue3 \(site): shape/dtype mismatch "
                        + "\(candidate.shape) \(candidate.dtype) vs "
                        + "\(reference.shape) \(reference.dtype)\n").utf8))
            return
        }
        let wordType: DType
        switch candidate.dtype {
        case .bfloat16, .float16, .int16, .uint16: wordType = .uint16
        case .float32, .int32, .uint32: wordType = .uint32
        default: wordType = candidate.dtype
        }
        let differs = candidate.view(dtype: wordType) .!= reference.view(dtype: wordType)
        let differing = MLX.sum(differs, stream: .default)
        eval(candidate, reference, differing)
        FileHandle.standardError.write(
            Data(
                ("[xcheck] prompt-glue3 \(site) shape \(candidate.shape) "
                    + "words \(candidate.size) differing \(differing.item(Int32.self))\n").utf8))
    }
}
