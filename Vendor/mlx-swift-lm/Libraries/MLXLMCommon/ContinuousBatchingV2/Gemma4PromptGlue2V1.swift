// PROMPT-GLUE2 (pg2): three prompt-plane glue kernels re-hosted in
// bandwidth-optimal form, output words unchanged.
//
// On the prompt pass (8 x 1024 tokens) every glue kernel between the GEMMs
// is bandwidth-bound, and most already stream at the copy roof. Three do
// not, for reasons that have nothing to do with their arithmetic:
//   * `cbv2_prefill_sdpa_softmax_vec_bf16_v1` runs one 128..1024-column row
//     per threadgroup -- 16384 threadgroups of 32..256 threads per query
//     block -- and is bound by threadgroup residency, not by its bytes: its
//     time per dispatch barely moves with the row length.
//   * `gemma4_qkv_rms_norm_head_major_sliding_pack_pg1` loads each input
//     word twice through scalar loads, stages the row through two
//     threadgroup arrays and meets five barriers per row.
//   * `mlx_lm_route_csort128_scan_v3` walks the 256-block histogram twice
//     in sequence from one 256-thread threadgroup.
// The twins in this arm keep every per-element expression, its order, its
// dtype and every reduction tree; they change only the memory access form:
// rows per threadgroup (softmax), vector loads and stores plus one staging
// array and a register-held combine (QKV), and a range-split integer sum
// (scan). Each is admitted only for prompt-width rectangles (`minRows`
// token rows and up), so decode keeps its dispatches byte for byte.
//
// Kill switch: `DARKBLOOM_GEMMA4_PROMPT_GLUE2=0` (every twin off, the
// incumbent dispatches unchanged). Engage mark: `prompt-glue2`.
// `DARKBLOOM_GEMMA4_PROMPT_GLUE2_XCHECK=1` evaluates the incumbent kernel
// beside every pg2 dispatch and counts differing words (diagnostic only;
// forces evaluation).

import Foundation
import MLX

public enum Gemma4PromptGlue2V1 {
    public static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_PROMPT_GLUE2"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// Token-row floor. The scored prompt plane carries 8192 token rows;
    /// decode and verify rectangles carry 8..256.
    public static let minRows: Int = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_PROMPT_GLUE2_MIN_ROWS"],
            let value = Int(raw), value > 0
        else { return 1024 }
        return value
    }()

    public static let xcheck: Bool =
        ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_PROMPT_GLUE2_XCHECK"] == "1"

    public static func mark() {
        CBv2EngageMark.once("prompt-glue2")
    }

    // MARK: - diagnostics (never on a timed run)

    /// Counts words that differ between a pg2 kernel's output and the
    /// incumbent's, evaluating both. `mask` (broadcastable to the arrays)
    /// restricts the count to words the incumbent defines.
    public static func report(
        _ candidate: MLXArray, reference: MLXArray, site: String, mask: MLXArray? = nil
    ) {
        guard candidate.shape == reference.shape, candidate.dtype == reference.dtype else {
            FileHandle.standardError.write(
                Data(
                    ("[xcheck] prompt-glue2 \(site): shape/dtype mismatch "
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
        var differs = candidate.view(dtype: wordType) .!= reference.view(dtype: wordType)
        if let mask {
            differs = differs .&& mask
        }
        let differing = MLX.sum(differs, stream: .default)
        eval(candidate, reference, differing)
        FileHandle.standardError.write(
            Data(
                ("[xcheck] prompt-glue2 \(site) shape \(candidate.shape) "
                    + "words \(candidate.size) differing \(differing.item(Int32.self))\n").utf8))
    }
}
