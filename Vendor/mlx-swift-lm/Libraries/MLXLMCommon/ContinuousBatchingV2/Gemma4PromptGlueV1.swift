// PROMPT-GLUE (pg1): the prompt plane's GeLU-gated products as one vector
// kernel.
//
// On the prompt pass (8 x 1024 tokens) the two `gelu(gate) * up` sites run
// as MLX compiled elementwise kernels at ONE element per thread. The dense
// site reads two contiguous `[8192, 2112]` planes; the routed-expert site
// reads the two halves of the fused `[65536, 1408]` gate|up gather output
// through strided views, which selects the compiled kernel's 2-D "strided"
// variant (`elem_to_loc` per input per element, scalar bf16 loads). Per
// layer that is 277 MB (experts) + 104 MB (dense) of traffic through the
// slowest access pattern MLX has -- 11 GB per prompt pass.
//
// This kernel computes the identical expression per element, in the
// identical order and at the identical dtype, but reads and writes four
// consecutive columns per thread (`vec<T, 4>`), and takes the fused expert
// plane directly (column offset for the up half) instead of the strided
// views. Same bytes, coalesced 8-byte accesses, no per-element index math.
//
// EXACTNESS. The compiled kernel's tape (`Compiled::eval_gpu`, read off the
// kernel name) is, with B = gate and F = up and every op at bfloat16:
//     G = T(0.5)   J = T(1)   L = T(0.7978846)   N = T(0.044715)
//     I = G * B
//     P = N * B;  Q = P * B;  R = Q * B;  S = B + R;  T_ = L * S
//     U = tanh(T_);  V = J + U;  W = I * V;  X = W * F
// The constants are printed by `print_float_constant` at seven significant
// digits and cast float -> bfloat16 (the `1` is an int32 cast); the same
// literals and casts appear below. Every binary op is a native `bfloat`
// op on two `bfloat` operands (fast math is disabled for every JIT library,
// so nothing is contracted or reassociated), and `Tanh` is
// `metal::precise::tanh` on the `bfloat` operand with the result stored as
// `bfloat` -- the expressions below are the same statements. IEEE
// multiplication and addition are commutative, so the operand order inside
// one op does not matter; the association is the tape's.
//
// SCOPE. Admitted only for rectangles of at least `minRows` rows (1024: the
// scored prompt plane carries 8192 rows dense and 65536 rows routed). Decode
// (`[8, 1, N]`, `[64, 1, N]`), the MTP verify rectangles and every other
// geometry never reach the row floor and keep their incumbent dispatches
// byte for byte. Kill switch: `DARKBLOOM_GEMMA4_PROMPT_GLUE=0`. Engage mark:
// `prompt-glue`. `DARKBLOOM_GEMMA4_PROMPT_GLUE_XCHECK=1` evaluates the
// incumbent compiled kernel beside every admitted dispatch and counts
// differing bf16 words (diagnostic only; forces evaluation).

import Foundation
import MLX
import MLXFast

public enum Gemma4PromptGlueV1 {
    public static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_PROMPT_GLUE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// Activation-row floor. Prompt planes carry >= 1024 rows; decode and
    /// verify rectangles carry 8..256.
    public static let minRows: Int = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_PROMPT_GLUE_MIN_ROWS"],
            let value = Int(raw), value > 0
        else { return 1024 }
        return value
    }()

    public static let xcheck: Bool =
        ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_PROMPT_GLUE_XCHECK"] == "1"

    private static let geluProductKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "gemma4_prompt_gelu_product_bf16_pg1",
        inputNames: ["gate", "up"],
        outputNames: ["out"],
        source: """
            typedef vec<T, 4> T4;
            constexpr uint cols4 = N / 4;
            const uint tid = thread_position_in_grid.x;
            const uint row = tid / cols4;
            if (row >= ROWS) return;
            const uint c4 = tid - row * cols4;
            const size_t base = size_t(row) * PITCH + size_t(c4) * 4;
            const T4 g4 = *reinterpret_cast<const device T4*>(gate + base);
            const T4 u4 = *reinterpret_cast<const device T4*>(up + base + UP_OFF);

            // The compiled kernel's constants: printed at seven significant
            // digits, cast to float, then to T (the `1` is an int32 cast).
            const T c_half = static_cast<T>(static_cast<float>(0.5));
            const T c_one = static_cast<T>(static_cast<int32_t>(1));
            const T c_k = static_cast<T>(static_cast<float>(0.7978846));
            const T c_a = static_cast<T>(static_cast<float>(0.044715));

            T4 o4;
            for (uint i = 0; i < 4; ++i) {
                const T g = g4[i];
                const T u = u4[i];
                const T hg = c_half * g;
                T t = c_a * g;
                t = t * g;
                t = t * g;
                t = g + t;
                t = c_k * t;
                t = metal::precise::tanh(t);
                t = c_one + t;
                t = hg * t;
                o4[i] = t * u;
            }
            *reinterpret_cast<device T4*>(out + size_t(row) * N + size_t(c4) * 4) = o4;
            """,
        ensureRowContiguous: true
    )

    /// `gelu(gate) * up` over two same-shaped contiguous bf16 planes whose
    /// last dimension is a multiple of 4, with at least `minRows` rows.
    /// nil keeps the caller on its incumbent closure.
    public static func geluProduct(gate: MLXArray, up: MLXArray) -> MLXArray? {
        guard enabled,
            gate.dtype == .bfloat16, up.dtype == .bfloat16,
            gate.ndim >= 2, gate.shape == up.shape
        else { return nil }
        let n = gate.dim(-1)
        guard n > 0, n % 4 == 0 else { return nil }
        let rows = gate.size / n
        guard rows >= minRows else { return nil }
        return dispatch(
            gate: gate, up: up, rows: rows, n: n, pitch: n, upOffset: 0,
            outputShape: gate.shape)
    }

    /// The routed-expert form: one fused `[..., 2 * hidden]` plane whose
    /// first `hidden` columns are the gate projection and whose last
    /// `hidden` columns are the up projection. The result has the plane's
    /// shape with `hidden` as its last dimension -- exactly what the
    /// incumbent closure returns from the two sliced views.
    public static func geluProductFusedPlane(_ plane: MLXArray, hidden n: Int) -> MLXArray? {
        guard enabled,
            plane.dtype == .bfloat16, plane.ndim >= 2,
            n > 0, n % 4 == 0, plane.dim(-1) == 2 * n
        else { return nil }
        let rows = plane.size / (2 * n)
        guard rows >= minRows else { return nil }
        var shape = plane.shape
        shape[shape.count - 1] = n
        return dispatch(
            gate: plane, up: plane, rows: rows, n: n, pitch: 2 * n, upOffset: n,
            outputShape: shape)
    }

    /// GEGLU-PG1-DECODE: the fused-plane form, admitted at the DECODE row
    /// counts.
    ///
    /// `geluProductFusedPlane` above is unreachable on every plane it was
    /// written for. Its own doc calls it "the routed-expert form", and the
    /// `[..., 2 * hidden]` layout it describes is exactly what the decode
    /// dense (`[8, 1, 4224]`) and routed-expert (`[64, 1, 1408]`) gate|up
    /// joins produce today -- but `minRows` is 1024 and those planes carry 8
    /// and 64 rows, so it has never had a caller.
    ///
    /// The floor is inherited from `Gemma4PrefillDeqGEMMV1`, where it is
    /// load-bearing: that mechanism trades plane BYTES for plane WORK, and
    /// the trade only pays above a row count (the arithmetic intensity of a
    /// weight plane is exactly the number of activation rows multiplied
    /// against it). THIS mechanism moves no weight plane and does no
    /// dequantization. It reads and writes byte-for-byte the traffic the
    /// incumbent compiled closure reads and writes; the only difference is
    /// that it reads a contiguous plane with `vec<T, 4>` accesses at one
    /// computed base, where the incumbent walks two strided views of pitch
    /// `2 * hidden` one element per thread. A row floor prices nothing here.
    ///
    /// Bit-identity is the design of `geluProductKernel` above and is not
    /// assumed: it reproduces the compiled closure's constants at seven
    /// significant digits and every bf16 temporary in order, and
    /// `DARKBLOOM_GEMMA4_PROMPT_GLUE_XCHECK=1` counts differing bf16 words
    /// against the exact incumbent this replaces.
    ///
    /// `DARKBLOOM_GEMMA4_PROMPT_GLUE_DECODE=0` returns nil here and nowhere
    /// else, which restores the incumbent closure on the decode planes while
    /// leaving every prompt-width admission above untouched.
    public static let decodePlaneEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_PROMPT_GLUE_DECODE"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// Admits exactly the two pinned decode rectangles and nothing else, so
    /// no prefill or verify geometry can reach this entry point.
    public static func geluProductFusedPlaneDecode(
        _ plane: MLXArray, hidden n: Int
    ) -> MLXArray? {
        guard enabled, decodePlaneEnabled,
            plane.dtype == .bfloat16, plane.ndim >= 2,
            n > 0, n % 4 == 0, plane.dim(-1) == 2 * n
        else { return nil }
        let rows = plane.size / (2 * n)
        guard rows == 8 || rows == 64 else { return nil }
        var shape = plane.shape
        shape[shape.count - 1] = n
        CBv2EngageMark.once("prompt-glue-decode")
        return dispatch(
            gate: plane, up: plane, rows: rows, n: n, pitch: 2 * n, upOffset: n,
            outputShape: shape)
    }

    private static func dispatch(
        gate: MLXArray, up: MLXArray, rows: Int, n: Int, pitch: Int, upOffset: Int,
        outputShape: [Int]
    ) -> MLXArray {
        CBv2EngageMark.once("prompt-glue")
        let threads = rows * (n / 4)
        return geluProductKernel(
            [gate, up],
            template: [
                ("T", DType.bfloat16), ("N", n), ("ROWS", rows),
                ("PITCH", pitch), ("UP_OFF", upOffset),
            ],
            grid: (threads, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [outputShape],
            outputDTypes: [.bfloat16]
        )[0]
    }

    // MARK: - PACK-IN-NORMROPE: pre-packed q4 KV mirrors

    /// The sliding layers' prompt q4 mirror pack
    /// (`cbv2_kvq4g64_pack_pair_chunk_batch_d256_v1`) re-reads the 67 MB of
    /// bf16 K/V the norm+RoPE kernel has just written. The norm+RoPE twin
    /// with the pack folded in (`gemma4_qkv_rms_norm_head_major_sliding_pack_pg1`,
    /// Gemma4Text.swift) emits each row's mirror from the rounded values it
    /// holds in threadgroup memory, with the pack kernel's per-lane
    /// arithmetic and lane mapping. The attention layer registers the
    /// mirrors here against the K/V arrays it hands the cache; the batched
    /// fresh-chunk commit takes them by array identity and skips its pack
    /// dispatch. Any mismatch leaves the pack dispatch in place.
    /// Kill switch: `DARKBLOOM_GEMMA4_PROMPT_GLUE_PACK=0` (also off under the
    /// arm's own switch). Engage mark: `prompt-glue-pack`.
    public static let packEnabled: Bool = {
        guard enabled else { return false }
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_PROMPT_GLUE_PACK"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    nonisolated(unsafe) private static var packedMirrors:
        (keys: MLXArray, values: MLXArray, mirrors: [MLXArray])? = nil
    private static let packedLock = NSLock()

    public static func registerPackedMirrors(
        keys: MLXArray, values: MLXArray, mirrors: [MLXArray]
    ) {
        packedLock.lock()
        packedMirrors = (keys, values, mirrors)
        packedLock.unlock()
    }

    /// The mirrors registered for exactly these K/V arrays, one `[2, heads,
    /// n, words]` uint32 plane per row; nil (and the pack dispatch) for any
    /// other array pair or geometry. The entry is consumed either way.
    public static func takePackedMirrors(
        keys: MLXArray, values: MLXArray, rows: Int, heads: Int, n: Int, words: Int
    ) -> [MLXArray]? {
        packedLock.lock()
        let entry = packedMirrors
        packedMirrors = nil
        packedLock.unlock()
        guard let entry, entry.keys === keys, entry.values === values,
            entry.mirrors.count == rows,
            entry.mirrors.allSatisfy({ $0.dtype == .uint32 && $0.shape == [2, heads, n, words] })
        else { return nil }
        CBv2EngageMark.once("prompt-glue-pack")
        return entry.mirrors
    }

    // MARK: - diagnostics (never on a timed run)

    /// Counts bf16 words that differ between the kernel's output and the
    /// incumbent's, evaluating both. `site` names the caller in the log.
    public static func report(_ candidate: MLXArray, reference: MLXArray, site: String) {
        let differing = MLX.sum(
            candidate.view(dtype: .uint16) .!= reference.view(dtype: .uint16),
            stream: .default)
        eval(candidate, reference, differing)
        FileHandle.standardError.write(
            Data(
                ("[xcheck] prompt-glue \(site) shape \(candidate.shape) "
                    + "differing \(differing.item(Int32.self))\n").utf8))
    }

    nonisolated(unsafe) private static var exhaustiveDone = false
    private static let exhaustiveLock = NSLock()

    /// Every bfloat16 bit pattern as the gate operand (with `up == 1`, an
    /// exact identity on the product) through both roads; reports the
    /// differing-word count over all 65536 inputs and over the finite ones.
    /// Runs once per process; `stock` is the incumbent closure.
    public static func exhaustiveCheck(stock: (MLXArray, MLXArray) -> MLXArray) {
        exhaustiveLock.lock()
        let fresh = !exhaustiveDone
        exhaustiveDone = true
        exhaustiveLock.unlock()
        guard fresh else { return }
        let bits = MLXArray((0 ..< 65536).map { UInt16($0) }).reshaped(1024, 64)
        let gate = bits.view(dtype: .bfloat16)
        let up = MLXArray.ones([1024, 64], dtype: .bfloat16)
        guard let candidate = geluProduct(gate: gate, up: up) else {
            FileHandle.standardError.write(
                Data("[xcheck] prompt-glue exhaustive: not admitted\n".utf8))
            return
        }
        let reference = stock(gate, up)
        let differs = candidate.view(dtype: .uint16) .!= reference.view(dtype: .uint16)
        let gate32 = gate.asType(.float32)
        let finite = (gate32 .<= Float(3.3895314e38)) .&& (gate32 .>= Float(-3.3895314e38))
        let total = MLX.sum(differs, stream: .default)
        let finiteDiffs = MLX.sum(differs .&& finite, stream: .default)
        eval(total, finiteDiffs)
        FileHandle.standardError.write(
            Data(
                ("[xcheck] prompt-glue exhaustive: differing \(total.item(Int32.self)) "
                    + "of 65536, finite-input differing \(finiteDiffs.item(Int32.self))\n").utf8))
    }
}
