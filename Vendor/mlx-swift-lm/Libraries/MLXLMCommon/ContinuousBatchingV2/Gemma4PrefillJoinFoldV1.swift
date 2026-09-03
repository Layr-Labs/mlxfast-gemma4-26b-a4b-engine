// PREFILL-JOIN-FOLD (jf1): the prompt attention's per-block P.V products
// are stored straight into the token-major rectangle the output projection
// reads, so the join dispatch that used to assemble that rectangle from the
// eight block outputs (`cbv2_prefill_join_v1_nb8`: one read and one write of
// the whole rectangle per layer, 4.4 GB per ranked prompt pass) is not
// issued.
//
// The block loop already hands every block's P.V product to `addMM` with the
// PREFILL-ATTN-TRAFFIC statistics carrier as its out-source operand. Here
// the statistics of all blocks and the output rectangle share ONE bf16
// carrier buffer:
//
//   [ slot 0 | slot 1 | ... | slot nBlocks-1 ][ B, L, H, D rectangle ]
//
// with a slot holding a block's `[B, nKV, rep, BS, 4]` row statistics. The
// first block's statistics kernel allocates the carrier as its output; every
// later block's statistics kernel writes its slot in place (its output is the
// carrier itself, see `_aliaslast_` in custom_kernel.cpp), so each block's
// carrier depends on the previous one. Each block's `addMM` then names its
// slot through the same column view the at1 call uses, with `alpha = -L`
// (the chunk's token count) beside the at1 loader beta: on that signature
// the host (matmul.cpp, `addmm_join_fold`) binds the GEMM's output as the
// strided view of the block's rows of the rectangle and dispatches with
// per-dimension output batch strides; the kernel receives the at1 scalars
// and runs the at1 dispatch, tile for tile, storing every word to its
// rectangle address instead of to a fresh block buffer. A final no-op fence
// kernel, aliasing the carrier and taking every block output as an input,
// is the array the caller slices the rectangle from: it carries the
// dependency on every GEMM.
//
// EXACTNESS. The statistics kernels are the at1 text with a slot offset on
// the store; the GEMMs are the at1 dispatches (same loader transform, tiles,
// K order, accumulators, epilogue) with different store addresses; the fence
// moves no bytes; the rectangle views move no bytes. The join was a copy, so
// the rectangle holds, word for word, what the join wrote.
//
// Prompt width only: admitted when the composed path takes the at1 fused
// call for EVERY block of a uniformly blocked chunk (the same guards, block
// by block, ahead of the loop); decode (L = 1) and the MTP verify widths
// never reach the blocked prompt path. Kill switch:
// `DARKBLOOM_GEMMA4_PREFILL_JOIN_FOLD=0` (the at1 calls and the join dispatch,
// byte for byte). Engage mark: `prefill-join-fold`.
// `DARKBLOOM_GEMMA4_PREFILL_JOIN_FOLD_XCHECK=1` evaluates the incumbent
// blocks and join beside every folded chunk and counts differing output words
// (diagnostic only; forces evaluation).

import Foundation
import MLX
import MLXFast

/// One chunk's fold state: the carrier chain head and the delivered blocks.
final class CBv2PrefillJoinFoldContext {
    let B: Int
    let nKV: Int
    let rep: Int
    let H: Int
    let BS: Int
    let L: Int
    let D: Int
    let nBlocks: Int
    /// Words per statistics slot: `B * H * BS * 4`.
    let slotElems: Int
    /// Words before the rectangle: `nBlocks * slotElems`.
    let statsElems: Int
    /// Words of the rectangle: `B * L * H * D`.
    let rectElems: Int
    let totalElems: Int

    /// The carrier chain head (the whole buffer, rank 1), nil before block 0.
    var carrier: MLXArray?
    /// Blocks whose product has been stored into the rectangle, in order.
    var delivered = 0
    /// Set by the block loop when any block did not take the folded call.
    var broken = false

    init(B: Int, nKV: Int, rep: Int, BS: Int, L: Int, D: Int, nBlocks: Int) {
        self.B = B
        self.nKV = nKV
        self.rep = rep
        self.H = nKV * rep
        self.BS = BS
        self.L = L
        self.D = D
        self.nBlocks = nBlocks
        self.slotElems = B * H * BS * 4
        self.statsElems = nBlocks * slotElems
        self.rectElems = B * L * H * D
        self.totalElems = statsElems + rectElems
    }

    func slotOffset(block: Int) -> Int { block * slotElems }
}

public enum Gemma4PrefillJoinFoldV1 {
    public static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_PREFILL_JOIN_FOLD"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    public static let xcheck: Bool =
        ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_PREFILL_JOIN_FOLD_XCHECK"] == "1"

    public static func mark() {
        CBv2EngageMark.once("prefill-join-fold")
    }

    /// Fence inputs are one buffer per block plus the carrier; the output
    /// makes 31 Metal buffer bindings at most.
    private static let maxBlocks = 29

    /// The fold context for a uniformly blocked prompt chunk whose every
    /// block takes the at1 fused call, or nil (the block loop keeps the
    /// established calls and the join dispatch). Mirrors, block by block,
    /// the admissions of `CBv2ComposedPrefillSDPAV1.attend` and
    /// `CBv2PrefillAttnTrafficV1.attend` so no admitted block can fall back.
    static func admit(
        queries: MLXArray, keys: MLXArray, values: MLXArray, queryPlane: MLXArray,
        newTokenCount: Int, blockSize: Int, historyCount: Int, window: Int?,
        hasSpanContext: Bool
    ) -> CBv2PrefillJoinFoldContext? {
        guard enabled, CBv2PrefillAttnTrafficV1.foldCapable else { return nil }
        guard !hasSpanContext, blockSize > 8, newTokenCount > blockSize,
            newTokenCount % blockSize == 0
        else { return nil }
        let nBlocks = newTokenCount / blockSize
        guard nBlocks >= 2, nBlocks <= maxBlocks else { return nil }
        guard queries.ndim == 4, keys.ndim == 4, values.ndim == 4, queryPlane.ndim == 5,
            queries.dtype == .bfloat16, keys.dtype == .bfloat16,
            values.dtype == .bfloat16, queryPlane.dtype == .bfloat16
        else { return nil }
        let B = queries.dim(0)
        let H = queries.dim(1)
        let nKV = keys.dim(1)
        let queryDim = queries.dim(3)
        let D = values.dim(3)
        guard B >= 1, nKV >= 1, H % nKV == 0 else { return nil }
        let rep = H / nKV
        guard rep > 1, values.dim(1) == nKV, keys.dim(3) == queryDim,
            keys.dim(0) == B, values.dim(0) == B,
            keys.dim(2) == values.dim(2), keys.dim(2) == historyCount + newTokenCount,
            queryPlane.dim(0) == B, queryPlane.dim(1) == nKV, queryPlane.dim(2) == rep,
            queryPlane.dim(3) == newTokenCount, queryPlane.dim(4) == queryDim
        else { return nil }
        let BS = blockSize
        // Every steel tile divides the block and the head dim (the MN-aligned
        // path, the only one carrying the loader twin).
        guard BS % 128 == 0, D % 128 == 0 else { return nil }
        guard B * H * BS >= Gemma4PromptGlue2V1.minRows else { return nil }
        for block in 0 ..< nBlocks {
            let bounds = CBv2AttentionV1.queryBlockBounds(
                historyCount: historyCount, offset: block * BS, count: BS, window: window)
            let kL = bounds.visibleEnd - bounds.visibleStart
            guard kL >= BS else { return nil }
            // Symbolic causal only: an array mask keeps the stock call.
            if let window, kL > window { return nil }
            guard CBv2PrefillAttnTrafficV1.admitsKeyLength(kL) else { return nil }
        }
        let context = CBv2PrefillJoinFoldContext(
            B: B, nKV: nKV, rep: rep, BS: BS, L: newTokenCount, D: D, nBlocks: nBlocks)
        guard context.totalElems < (1 << 31), context.statsElems < (1 << 32) else { return nil }
        return context
    }

    nonisolated(unsafe) private static var fenceKernels: [Int: MLXFast.MLXFastKernel] = [:]
    private static let fenceKernelLock = NSLock()

    /// The fence for `blockCount` blocks: a one-thread kernel with no body
    /// whose output aliases its last input (the carrier) and whose other
    /// inputs are the block outputs. It exists for the graph edge, not for
    /// any byte it moves; the name carries the aliasing convention.
    private static func fenceKernel(blockCount: Int) -> MLXFast.MLXFastKernel {
        fenceKernelLock.lock()
        defer { fenceKernelLock.unlock() }
        if let hit = fenceKernels[blockCount] { return hit }
        var inputNames = (0 ..< blockCount).map { "d\($0)" }
        inputNames.append("carrier")
        let kernel = MLXFast.metalKernel(
            name: "cbv2_prefill_join_fence_jf1_nb\(blockCount)_aliaslast_",
            inputNames: inputNames,
            outputNames: ["joined"],
            source: "",
            ensureRowContiguous: false)
        fenceKernels[blockCount] = kernel
        return kernel
    }

    /// The token-major rectangle `[B, L, H, D]` of a fully delivered chunk:
    /// the fence over the carrier and the block outputs, then the rectangle
    /// as a view of it. nil when the chain is incomplete.
    static func finish(
        _ fold: CBv2PrefillJoinFoldContext, blockOutputs: [MLXArray]
    ) -> MLXArray? {
        guard !fold.broken, let carrier = fold.carrier,
            fold.delivered == fold.nBlocks, blockOutputs.count == fold.nBlocks,
            carrier.ndim == 1, carrier.dim(0) == fold.totalElems
        else { return nil }
        var inputs: [any ScalarOrArray] = blockOutputs.map { $0 as any ScalarOrArray }
        inputs.append(carrier)
        let joined = fenceKernel(blockCount: fold.nBlocks)(
            inputs,
            grid: (1, 1, 1),
            threadGroup: (1, 1, 1),
            outputShapes: [[fold.totalElems]],
            outputDTypes: [.bfloat16]
        )[0]
        let rectangle = joined[fold.statsElems ..< (fold.statsElems + fold.rectElems)]
            .reshaped([fold.B, fold.L, fold.H, fold.D])
        mark()
        return rectangle
    }

    // MARK: - diagnostics (never on a timed run)

    /// Counts words that differ between the folded rectangle and the
    /// incumbent join's, evaluating both.
    public static func report(_ candidate: MLXArray, reference: MLXArray, site: String) {
        guard candidate.shape == reference.shape, candidate.dtype == reference.dtype else {
            FileHandle.standardError.write(
                Data(
                    ("[xcheck] prefill-join-fold \(site): shape/dtype mismatch "
                        + "\(candidate.shape) \(candidate.dtype) vs "
                        + "\(reference.shape) \(reference.dtype)\n").utf8))
            return
        }
        let differs = candidate.view(dtype: .uint16) .!= reference.view(dtype: .uint16)
        let differing = MLX.sum(differs, stream: .default)
        eval(candidate, reference, differing)
        FileHandle.standardError.write(
            Data(
                ("[xcheck] prefill-join-fold \(site) shape \(candidate.shape) "
                    + "words \(candidate.size) differing \(differing.item(Int32.self))\n").utf8))
    }
}
