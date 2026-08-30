import Foundation
import MLX
import MLXFastCore
import MLXFastModel
import MLXLLM
import MLXLMCommon

// WIDTH-DIVERGENCE LOCALIZATION PROBE (exactness round three, 2026-08-25).
//
// The box evidence established a forward-WIDTH family of token divergences on
// the production tuple: width-1 forwards match the pinned tapes, while
// width-8 serial cohorts, width-(1+k) verify forwards, and width-8(1+k)
// batched MTP all flip near-tie argmaxes at deterministic prompt-specific
// steps (trap 3.1 physics — shape-dependent quantized-kernel reduction
// paths). The design question this probe answers, per position and
// layer-by-layer: WHERE does the width divergence FIRST enter the network?
//
//   (i)  Only at the final LM-head logits (accumulation-order noise in the
//        last matmul) → a width-stable head computation could restore full
//        bit-exactness across widths — the best design outcome.
//   (ii) At an MoE ROUTER selection mid-network (near-tie expert flips →
//        the entire downstream computation differs) → width-stability is
//        effectively unfixable and the correctness criterion must change.
//
// Method: teacher forcing along a pinned reference tape's own token chain,
// so every compared forward consumes IDENTICAL tokens and positions — any
// tensor difference is width-induced numerics, never context drift. Three
// stacks per tape:
//
//   * width-1 reference: seed prefill `[1, S]`, then one `[1, 1]` forward
//     per chain position;
//   * width-L windows: an independent stack fed the SAME chain as `[1, L]`
//     windows (L = 2, 3, ... — the MTP verify rectangle geometry);
//   * batch-width: an independent stack prefilled `[B, S]` with B identical
//     rows and stepped `[B, 1]` (the serial-cohort geometry); row 0 is
//     compared.
//
// At every chain position the probe bit-compares, in network order: each
// layer's newly-computed K/V (`captureHook`), each MoE router's
// pre-selection expert scores and its top-K selection (`Gemma4RouterProbe`),
// and the final logits — recording the FIRST divergent tensor, its
// magnitude, and the local margins (router k-th vs (k+1)-th score; logits
// top-1 vs top-2).
//
// OPERATOR-ONLY diagnostic: a separate CLI verb, never spawned by benchd
// (whose argv fence covers only `runtime-worker`), never on any scored or
// wire path. Laptop runs are expected to report bit-identity everywhere
// (the flip is box physics); the box invocation is the arbiter.
//
// FIDELITY-GATE PHASE-1 EXTENSION (2026-08-25). The exactness probe above
// reports, per position, only the top-2 tokens/logits and the top-2 margin.
// The fidelity-gate calibration (challenger token-fidelity ladder reuse) needs
// a DEEPER, RELATIVE reference-side characterization to answer a question
// top-2 is structurally blind to: "do top-3+ WITHIN-ENVELOPE near-ties occur
// over the full pool?" This extension ADDS (never removes) a top-N ranked
// readout of the reference logits per position, the ranked absolute gaps from
// top-1, the RELATIVE ratio of each gap to max(1, |top1|), and a
// WITHIN-ENVELOPE DEPTH = the count of ranks whose relative ratio is within a
// configurable envelope. Depth >= 3 is the falsifiable "top-3+ near-ties
// occur" signal. It arms nothing — it is a read-only measurement.
//
// LOGIT PROVENANCE (pinned). The relative form divides by |top1|, so pre- vs
// post-softcap is a real numerical choice. The probe reads
// `widthProbeForward`, which applies the LM head AND gemma's final-logit
// softcap (`applyLMHead`, cap = 30.0) — the exact tensor the sampler sees. We
// PIN post-softcap (matching emission), stamp `logit_provenance=post_softcap`
// into the output, and document it so a box sweep is unambiguous.

// MARK: - Options / tape

/// Logit provenance for the ranked/relative characterization: the width probe
/// reads `widthProbeForward`, whose `applyLMHead` applies the gemma final-logit
/// softcap (cap = 30.0), so these logits are post-softcap — the same tensor the
/// sampler consumes. Pinned so the relative form has a consistent, documented
/// reference.
let widthProbeLogitProvenance = "post_softcap"

public struct WidthProbeOptions {
    public let weightsPath: String
    public let tapePath: String
    public let steps: Int
    public let windowWidths: [Int]
    public let batchWidth: Int?
    public let outputPath: String?
    /// Number of ranked reference logits/tokens to read per position. Default 2
    /// keeps the pre-extension top-2 behavior; the box sweep raises it (e.g. 16)
    /// to characterize the near-tie pool depth.
    public let logitTopK: Int
    /// Within-envelope relative-ratio threshold: a rank counts toward the
    /// within-envelope depth when (top1 - rank_i) / max(1, |top1|) <= this.
    /// Default 0.05 is a generous cover over the contract's EPS_REL (~0.4-0.8%)
    /// with a safety factor; the box sweeps it.
    public let relEnvelope: Double

    public init(
        weightsPath: String, tapePath: String, steps: Int,
        windowWidths: [Int], batchWidth: Int?, outputPath: String?,
        logitTopK: Int = 2, relEnvelope: Double = 0.05
    ) {
        self.weightsPath = weightsPath
        self.tapePath = tapePath
        self.steps = steps
        self.windowWidths = windowWidths
        self.batchWidth = batchWidth
        self.outputPath = outputPath
        self.logitTopK = logitTopK
        self.relEnvelope = relEnvelope
    }
}

struct WidthProbeTape {
    let seedTokens: [Int]
    /// chain[0] = reference_seed_token; chain[i+1] = rows[i].sequential_argmax.
    let chain: [Int]

    static func load(path: String) throws -> WidthProbeTape {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let seed = root["seed_tokens"] as? [Int],
            let seedToken = root["reference_seed_token"] as? Int,
            let rows = root["rows"] as? [[String: Any]]
        else {
            throw MLXFastError.invalidInput(
                "width-probe tape \(path) is not a timed-prompt tape document "
                    + "(seed_tokens / reference_seed_token / rows required)")
        }
        let chain = try [seedToken] + rows.map { row -> Int in
            guard let token = row["sequential_argmax"] as? Int else {
                throw MLXFastError.invalidInput(
                    "width-probe tape row missing sequential_argmax")
            }
            return token
        }
        return WidthProbeTape(seedTokens: seed, chain: chain)
    }
}

// MARK: - Per-forward capture

/// Everything one probed forward records, per fed position: layer K/V, MoE
/// router scores/selections (layer execution order), and final logits.
struct ForwardCapture {
    /// [layerIdx: (keys, values)] for the fed window — shapes [1, H, L, D].
    var layerKV: [Int: (MLXArray, MLXArray)] = [:]
    /// Router events in layer execution order: (expertScores [.., E],
    /// topKIndices [.., K]) — flattened token-major by the router.
    var routerEvents: [(scores: MLXArray, indices: MLXArray)] = []
    var logits: MLXArray?
}

/// Reference-type box so the @escaping capture hook can accumulate without
/// capturing an inout struct.
private final class CaptureBox {
    var layerKV: [Int: (MLXArray, MLXArray)] = [:]
}

private final class RouterBox {
    var events: [(scores: MLXArray, indices: MLXArray)] = []
}

func capturedForward(
    model: Gemma4TextModel, tokens: [Int], batch: Int, cache: [KVCache]
) -> ForwardCapture {
    var capture = ForwardCapture()
    let routerBox = RouterBox()
    Gemma4RouterProbe.recorder = { scores, indices in
        routerBox.events.append((scores, indices))
    }
    defer { Gemma4RouterProbe.recorder = nil }
    let row = tokens.map(Int32.init)
    let input = MLXArray(Array(repeating: row, count: batch).flatMap { $0 })
        .reshaped([batch, tokens.count])
    let box = CaptureBox()
    let logits = model.widthProbeForward(
        input, cache: cache,
        captureHook: { idx, kvPair in box.layerKV[idx] = kvPair })
    capture.layerKV = box.layerKV
    capture.routerEvents = routerBox.events
    capture.logits = logits
    eval(logits)
    return capture
}

// MARK: - Comparison

private func probeBitEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
    a.shape == b.shape && allClose(a, b, rtol: 0, atol: 0).item(Bool.self)
}

private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Double {
    guard a.shape == b.shape else { return .infinity }
    return Double(abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self))
}

/// Slice one position's column out of a window capture. K/V are
/// [B, H, L, D] (position axis 2); router tensors are token-major on their
/// leading flattened axis; logits are [B, L, V] (position axis 1).
private func kvColumn(_ kv: MLXArray, batchRow: Int, position: Int) -> MLXArray {
    kv[batchRow ..< batchRow + 1, 0..., position ..< position + 1, 0...]
}

// MARK: - Ranked reference-side (Phase-1 fidelity-gate) characterization

/// The top-N ranked, RELATIVE near-tie characterization of one reference
/// position's post-softcap logits — the deeper readout Phase-1 needs beyond
/// top-2. All fields are reference-only (a property of the teacher-forced
/// reference forward at that position), independent of any probe width.
struct RankedReferenceCharacterization {
    /// Top-N token ids, rank-0 first (descending logit).
    var tokens: [Int]
    /// Top-N logit values (post-softcap), rank-0 first.
    var logits: [Double]
    /// Absolute gap from top-1 to each rank: `top1 - rank_i` (>= 0, rank 0 = 0).
    var logitGaps: [Double]
    /// Relative ratio of each gap: `(top1 - rank_i) / max(1, |top1|)`.
    var relativeGaps: [Double]
    /// Count of ranks whose relative ratio <= the envelope (top-1 included, so
    /// >= 1). Depth >= 3 == "top-3+ within-envelope near-ties occur here".
    var withinEnvelopeDepth: Int
}

/// Rank one reference position's full post-softcap logit vector and derive the
/// relative near-tie characterization. `logits1D` is any shape that flattens to
/// the position's `[V]` logits. Materializes the vector off the device, then
/// defers to the pure-Swift core below.
func rankedReferenceCharacterization(
    logits1D: MLXArray, logitTopK: Int, relEnvelope: Double
) -> RankedReferenceCharacterization {
    rankedReferenceCharacterization(
        logits: logits1D.asType(.float32).flattened().asArray(Float.self),
        logitTopK: logitTopK, relEnvelope: relEnvelope)
}

/// Pure-Swift core over an already-materialized logit vector — the GPU-free
/// seam tests exercise (no MLX device needed). The relative form divides gaps
/// by `max(1, |top1|)`; the within-envelope depth counts ranks (top-1 included,
/// so >= 1) at relative ratio <= `relEnvelope`.
func rankedReferenceCharacterization(
    logits flat: [Float], logitTopK: Int, relEnvelope: Double
) -> RankedReferenceCharacterization {
    let order = flat.indices.sorted { flat[$0] > flat[$1] }
    let n = max(0, min(logitTopK, order.count))
    let idx = Array(order.prefix(n))
    let logitValues = idx.map { Double(flat[$0]) }
    let top1 = logitValues.first ?? 0
    let denominator = max(1.0, abs(top1))
    let gaps = logitValues.map { top1 - $0 }
    let relativeGaps = gaps.map { $0 / denominator }
    let depth = relativeGaps.filter { $0 <= relEnvelope }.count
    return RankedReferenceCharacterization(
        tokens: idx, logits: logitValues, logitGaps: gaps,
        relativeGaps: relativeGaps, withinEnvelopeDepth: depth)
}

/// Depth histogram keyed by decimal depth string (JSON object keys are strings).
func withinEnvelopeDepthHistogram(_ depths: [Int]) -> [String: Int] {
    var histogram: [String: Int] = [:]
    for depth in depths { histogram[String(depth), default: 0] += 1 }
    return histogram
}

struct PositionReport {
    var step: Int
    var firstDivergentKind: String?  // kv | router_scores | router_selection | logits
    var firstDivergentLayer: Int?
    var firstDivergentMaxAbsDiff: Double?
    var routerSelectionFlipped = false
    var routerMarginAtFlip: Double?
    var logitsBitEqual = true
    var logitsMaxAbsDiff: Double = 0
    var logitArgmaxA: Int = -1
    var logitArgmaxB: Int = -1
    var logitTop2MarginA: Double = 0
    // Phase-1 fidelity-gate additions (reference-only, post-softcap): the
    // top-N ranked readout and the relative within-envelope characterization.
    var rankedTokensRef: [Int] = []
    var rankedLogitsRef: [Double] = []
    var rankedLogitGapsRef: [Double] = []
    var rankedRelativeGapsRef: [Double] = []
    var withinEnvelopeDepth: Int = 0

    var asJSON: [String: Any] {
        var object: [String: Any] = [
            "step": step,
            "logits_bit_equal": logitsBitEqual,
            "logits_max_abs_diff": logitsMaxAbsDiff,
            "logit_argmax_ref": logitArgmaxA,
            "logit_argmax_probe": logitArgmaxB,
            "logit_argmax_flipped": logitArgmaxA != logitArgmaxB,
            "logit_top2_margin_ref": logitTop2MarginA,
            "router_selection_flipped": routerSelectionFlipped,
            // Additive Phase-1 fields — existing consumers ignore them.
            "within_envelope_depth": withinEnvelopeDepth,
            "ranked_tokens_ref": rankedTokensRef,
            "ranked_logits_ref": rankedLogitsRef,
            "ranked_logit_gaps_ref": rankedLogitGapsRef,
            "ranked_relative_gaps_ref": rankedRelativeGapsRef,
            "logit_provenance": widthProbeLogitProvenance,
        ]
        if let firstDivergentKind { object["first_divergent_kind"] = firstDivergentKind }
        if let firstDivergentLayer { object["first_divergent_layer"] = firstDivergentLayer }
        if let firstDivergentMaxAbsDiff {
            object["first_divergent_max_abs_diff"] = firstDivergentMaxAbsDiff
        }
        if let routerMarginAtFlip { object["router_margin_at_flip"] = routerMarginAtFlip }
        return object
    }
}

/// Compare one chain position between the width-1 reference capture (always
/// a [1,1] forward) and a probe capture's column `position` (row `batchRow`).
/// Layer order defines "first": for each layer index ascending, K/V first,
/// then that layer's router event (execution order pairs router events with
/// MoE layers ascending).
func comparePosition(
    step: Int,
    reference: ForwardCapture,
    probe: ForwardCapture,
    probePosition: Int,
    probeBatchRow: Int,
    referencePosition: Int = 0,
    topK: Int,
    logitTopK: Int,
    relEnvelope: Double
) -> PositionReport {
    var report = PositionReport(step: step)

    let layers = reference.layerKV.keys.sorted()
    // Router events per capture: one per MoE layer, execution order. Map
    // layer -> ordinal by pairing sorted MoE-event ordinals.
    let refRouterCount = reference.routerEvents.count
    let probeRouterCount = probe.routerEvents.count
    var routerOrdinal = 0

    func recordFirst(_ kind: String, _ layer: Int, _ diff: Double) {
        if report.firstDivergentKind == nil {
            report.firstDivergentKind = kind
            report.firstDivergentLayer = layer
            report.firstDivergentMaxAbsDiff = diff
        }
    }

    // The model interleaves: layer i's K/V projection, then (if MoE) layer
    // i's router. Router events count == MoE layer count; K/V capture covers
    // every layer. Walk layers ascending, consuming router events for MoE
    // layers (identified by matching counts — every layer is MoE on this
    // target when enable_moe_block, else zero events).
    let routerEventsPerLayer =
        layers.isEmpty ? 0 : (refRouterCount == layers.count ? 1 : 0)

    for layer in layers {
        if let refKV = reference.layerKV[layer], let probeKV = probe.layerKV[layer] {
            let refK = kvColumn(refKV.0, batchRow: 0, position: referencePosition)
            let refV = kvColumn(refKV.1, batchRow: 0, position: referencePosition)
            let probeK = kvColumn(probeKV.0, batchRow: probeBatchRow, position: probePosition)
            let probeV = kvColumn(probeKV.1, batchRow: probeBatchRow, position: probePosition)
            if !probeBitEqual(refK, probeK) {
                recordFirst("kv", layer, maxAbsDiff(refK, probeK))
            }
            if !probeBitEqual(refV, probeV) {
                recordFirst("kv", layer, maxAbsDiff(refV, probeV))
            }
        }
        if routerEventsPerLayer == 1, routerOrdinal < refRouterCount,
            routerOrdinal < probeRouterCount
        {
            let ref = reference.routerEvents[routerOrdinal]
            let probeEvent = probe.routerEvents[routerOrdinal]
            routerOrdinal += 1
            // Token-major flatten: reference position 0 row; probe row =
            // batchRow * L + position for [B, L] inputs flattened [B*L, E]
            // (router receives [B, S, E]-shaped scores; slice generically on
            // the leading axes by reshaping to [-1, E]).
            let expertCount = ref.scores.dim(-1)
            let refScores = ref.scores.reshaped([-1, expertCount])[
                referencePosition ..< referencePosition + 1, 0...]
            let probeRow = probeBatchRow * (probe.logits?.dim(1) ?? 1) + probePosition
            let probeScores = probeEvent.scores.reshaped([-1, expertCount])[
                probeRow ..< probeRow + 1, 0...]
            let scoresEqual = probeBitEqual(refScores, probeScores)

            let refSelection = Set(
                ref.indices.reshaped([-1, topK])[
                    referencePosition ..< referencePosition + 1, 0...
                ].asArray(Int32.self))
            let probeSelection = Set(
                probeEvent.indices.reshaped([-1, topK])[
                    probeRow ..< probeRow + 1, 0...
                ].asArray(Int32.self))
            if refSelection != probeSelection {
                report.routerSelectionFlipped = true
                // Margin between the k-th selected and the first excluded
                // score in the REFERENCE ordering — how near the tie was.
                let sorted = refScores.asType(.float32).flattened()
                    .asArray(Float.self).sorted(by: >)
                if sorted.count > topK {
                    report.routerMarginAtFlip = Double(sorted[topK - 1] - sorted[topK])
                }
                recordFirst("router_selection", layer, maxAbsDiff(refScores, probeScores))
            } else if !scoresEqual {
                recordFirst("router_scores", layer, maxAbsDiff(refScores, probeScores))
            }
        }
    }

    if let refLogits = reference.logits, let probeLogits = probe.logits {
        let ref = refLogits[0 ..< 1, referencePosition ..< referencePosition + 1, 0...]
        let probe = probeLogits[
            probeBatchRow ..< probeBatchRow + 1, probePosition ..< probePosition + 1, 0...]
        report.logitsBitEqual = probeBitEqual(ref, probe)
        report.logitsMaxAbsDiff = maxAbsDiff(ref, probe)
        report.logitArgmaxA = ref.flattened().argMax().item(Int.self)
        report.logitArgmaxB = probe.flattened().argMax().item(Int.self)
        let sorted = ref.asType(.float32).flattened().asArray(Float.self).sorted(by: >)
        if sorted.count >= 2 {
            report.logitTop2MarginA = Double(sorted[0] - sorted[1])
        }
        // Phase-1: the deeper ranked / relative within-envelope readout of the
        // SAME reference logits (post-softcap, `applyLMHead` output). Widens the
        // K of the existing top-logits readout for the probe path only.
        let characterization = rankedReferenceCharacterization(
            logits1D: ref, logitTopK: logitTopK, relEnvelope: relEnvelope)
        report.rankedTokensRef = characterization.tokens
        report.rankedLogitsRef = characterization.logits
        report.rankedLogitGapsRef = characterization.logitGaps
        report.rankedRelativeGapsRef = characterization.relativeGaps
        report.withinEnvelopeDepth = characterization.withinEnvelopeDepth
        if !report.logitsBitEqual {
            recordFirst("logits", -1, report.logitsMaxAbsDiff)
        }
    }
    return report
}

// MARK: - The probe

/// What `runWidthProbeCore` returns: the per-width family summaries (unchanged)
/// plus the per-run reference-side Phase-1 characterization (a single canonical
/// copy of the top-N ranked / within-envelope-depth readout, reference-only).
struct WidthProbeResult {
    var families: [[String: Any]]
    var referenceEnvelope: [String: Any]
}

extension Gemma4Runtime {

    public static func runWidthProbe(_ options: WidthProbeOptions) throws {
        try validateRuntimeWorkerPinnedConfiguration(weightsPath: options.weightsPath)
        let config = try Gemma4A4BConfig.load(from: options.weightsPath)
        let loader = try Gemma4A4BWeightLoader(weightsPath: options.weightsPath)
        let weightCache = Gemma4A4BRuntimeWeightCache(loader: loader, config: config)
        let model = try weightCache.requireLibraryModelAtDrainFencedBoundary()
        let tape = try WidthProbeTape.load(path: options.tapePath)
        let result = runWidthProbeCore(
            model: model, tape: tape, steps: options.steps,
            windowWidths: options.windowWidths, batchWidth: options.batchWidth,
            logitTopK: options.logitTopK, relEnvelope: options.relEnvelope)
        let report: [String: Any] = [
            "probe": "forward-width-divergence-localization",
            "weights": options.weightsPath,
            "tape": options.tapePath,
            "seed_tokens": tape.seedTokens.count,
            "steps": min(options.steps, tape.chain.count - 1),
            "logit_topk": options.logitTopK,
            "rel_envelope": options.relEnvelope,
            "logit_provenance": widthProbeLogitProvenance,
            "families": result.families,
            "reference_envelope": result.referenceEnvelope,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: report, options: [.sortedKeys, .prettyPrinted])
        if let outputPath = options.outputPath {
            try data.write(to: URL(fileURLWithPath: outputPath))
            FileHandle.standardError.write(
                Data("width-probe: report written to \(outputPath)\n".utf8))
        } else {
            FileHandle.standardOutput.write(data)
            FileHandle.standardOutput.write(Data("\n".utf8))
        }
    }

    /// The probe body over an already-loaded model — the seam laptop tests
    /// exercise with a weight-free fixture (expected all-bit-identical off
    /// the box; the machinery, indexing, and report shapes are what the
    /// test pins).
    static func runWidthProbeCore(
        model: Gemma4TextModel, tape: WidthProbeTape, steps requestedSteps: Int,
        windowWidths: [Int], batchWidth: Int?,
        logitTopK: Int = 2, relEnvelope: Double = 0.05
    ) -> WidthProbeResult {
        let steps = min(requestedSteps, tape.chain.count - 1)
        let topK = model.configuration.topKExperts ?? 1

        FileHandle.standardError.write(
            Data(
                ("width-probe: seed=\(tape.seedTokens.count) chain=\(tape.chain.count) "
                    + "steps=\(steps) widths=\(windowWidths) "
                    + "batch=\(batchWidth.map(String.init) ?? "off") "
                    + "topk=\(logitTopK) rel_envelope=\(relEnvelope)\n").utf8))

        // WIDTH-1 REFERENCE: prefill + one [1,1] forward per chain position,
        // capturing everything per step.
        var referenceCaptures: [ForwardCapture] = []
        do {
            let cache = model.newCache(parameters: nil)
            let prefill = model(
                MLXArray(tape.seedTokens.map(Int32.init))
                    .reshaped([1, tape.seedTokens.count]),
                cache: cache)
            eval(prefill)
            eval(cache)
            for step in 0 ..< steps {
                let capture = capturedForward(
                    model: model, tokens: [tape.chain[step]], batch: 1, cache: cache)
                referenceCaptures.append(capture)
                if (step + 1) % 8 == 0 {
                    FileHandle.standardError.write(
                        Data("width-probe: reference step \(step + 1)/\(steps)\n".utf8))
                }
            }
        }

        var families: [[String: Any]] = []

        // WIDTH-L WINDOW FAMILIES: an independent stack fed the same chain
        // as [1, L] windows.
        for width in windowWidths where width >= 2 {
            var reports: [PositionReport] = []
            let cache = model.newCache(parameters: nil)
            let prefill = model(
                MLXArray(tape.seedTokens.map(Int32.init))
                    .reshaped([1, tape.seedTokens.count]),
                cache: cache)
            eval(prefill)
            eval(cache)
            var position = 0
            while position < steps {
                let take = min(width, steps - position)
                let window = Array(tape.chain[position ..< position + take])
                let capture = capturedForward(
                    model: model, tokens: window, batch: 1, cache: cache)
                for offset in 0 ..< take {
                    reports.append(
                        comparePosition(
                            step: position + offset,
                            reference: referenceCaptures[position + offset],
                            probe: capture,
                            probePosition: offset,
                            probeBatchRow: 0,
                            topK: topK,
                            logitTopK: logitTopK,
                            relEnvelope: relEnvelope))
                }
                position += take
            }
            families.append(familySummary(name: "b1-width\(width)", reports: reports))
            FileHandle.standardError.write(
                Data("width-probe: family b1-width\(width) done\n".utf8))
        }

        // BATCH-WIDTH FAMILY: B identical rows, [B, S] prefill + [B, 1]
        // steps — the serial-cohort geometry; compare row 0.
        if let batch = batchWidth, batch >= 2 {
            var reports: [PositionReport] = []
            let cache = model.newCache(parameters: nil)
            let row = tape.seedTokens.map(Int32.init)
            let prefill = model(
                MLXArray(Array(repeating: row, count: batch).flatMap { $0 })
                    .reshaped([batch, tape.seedTokens.count]),
                cache: cache)
            eval(prefill)
            eval(cache)
            for step in 0 ..< steps {
                let capture = capturedForward(
                    model: model, tokens: [tape.chain[step]], batch: batch, cache: cache)
                reports.append(
                    comparePosition(
                        step: step,
                        reference: referenceCaptures[step],
                        probe: capture,
                        probePosition: 0,
                        probeBatchRow: 0,
                        topK: topK,
                        logitTopK: logitTopK,
                        relEnvelope: relEnvelope))
                if (step + 1) % 8 == 0 {
                    FileHandle.standardError.write(
                        Data("width-probe: batch step \(step + 1)/\(steps)\n".utf8))
                }
            }
            families.append(familySummary(name: "batch\(batch)-width1", reports: reports))
        }

        // PER-RUN REFERENCE ENVELOPE (Phase-1): the canonical single-copy
        // reference-side characterization. Independent of any probe width — it
        // ranks each teacher-forced reference position's own post-softcap
        // logits and reports the within-envelope depth over the whole run. This
        // is the falsifiable answer to "do top-3+ near-ties occur over the pool
        // tape" that the top-2 fields cannot give.
        var referencePositions: [[String: Any]] = []
        var referenceDepths: [Int] = []
        for (step, capture) in referenceCaptures.enumerated() {
            guard let logits = capture.logits else { continue }
            let characterization = rankedReferenceCharacterization(
                logits1D: logits[0 ..< 1, 0 ..< 1, 0...],
                logitTopK: logitTopK, relEnvelope: relEnvelope)
            referenceDepths.append(characterization.withinEnvelopeDepth)
            referencePositions.append([
                "step": step,
                "within_envelope_depth": characterization.withinEnvelopeDepth,
                "ranked_tokens": characterization.tokens,
                "ranked_logits": characterization.logits,
                "ranked_logit_gaps": characterization.logitGaps,
                "ranked_relative_gaps": characterization.relativeGaps,
            ])
        }
        let referenceEnvelope: [String: Any] = [
            "logit_provenance": widthProbeLogitProvenance,
            "topk": logitTopK,
            "rel_envelope": relEnvelope,
            "positions": referencePositions,
            "max_within_envelope_depth": referenceDepths.max() ?? 0,
            "within_envelope_depth_histogram": withinEnvelopeDepthHistogram(
                referenceDepths),
        ]

        return WidthProbeResult(families: families, referenceEnvelope: referenceEnvelope)
    }

    private static func familySummary(
        name: String, reports: [PositionReport]
    ) -> [String: Any] {
        let firstDivergent = reports.first(where: { $0.firstDivergentKind != nil })
        var summary: [String: Any] = [
            "family": name,
            "positions": reports.map(\.asJSON),
            "bit_identical_everywhere": firstDivergent == nil,
        ]
        if let firstDivergent {
            summary["first_divergent_step"] = firstDivergent.step
            summary["entry_kind"] = firstDivergent.firstDivergentKind ?? "?"
            summary["entry_layer"] = firstDivergent.firstDivergentLayer ?? -1
        }
        let flips = reports.filter { $0.logitArgmaxA != $0.logitArgmaxB }
        summary["logit_argmax_flip_steps"] = flips.map(\.step)
        let selectionFlips = reports.filter(\.routerSelectionFlipped)
        summary["router_selection_flip_steps"] = selectionFlips.map(\.step)
        // Phase-1 fidelity-gate: the family's within-envelope depth rollup. The
        // depths are reference-only, so they match the run-level reference
        // envelope, but each family carries its own copy for a self-describing
        // per-family view.
        let depths = reports.map(\.withinEnvelopeDepth)
        summary["max_within_envelope_depth"] = depths.max() ?? 0
        summary["within_envelope_depth_histogram"] = withinEnvelopeDepthHistogram(depths)
        return summary
    }
}
