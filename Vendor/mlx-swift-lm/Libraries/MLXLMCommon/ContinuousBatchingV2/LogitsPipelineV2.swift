// LogitsPipelineV2.swift
//
// Ordered, vectorized logits-transform pipeline for the ContinuousBatchingV2
// engine (workstream E). Operates on [B, vocab] logits, one row per running
// request, in the vLLM sampler order (research report 08 §6):
//
//   (0) capture RAW logprobs (log_softmax) when any row requests them —
//       BEFORE any transform, so reported logprobs reflect the untampered
//       distribution; top-k gather is lazy (`CBv2Logprobs`).
//   (1) logit-bias scatter-add.
//   (2) penalties — repetition (multiply/divide on presence over the
//       prompt ∪ output window), frequency (α·count over output), presence
//       (β·mask over output). Counts are per-row [B, vocab] tensors
//       maintained INCREMENTALLY via scatter-add each step; they are never
//       re-binned from scratch in the step path.
//   (3) temperature divide (greedy rows use 1.0 — argmax-invariant).
//   (4) top-k / top-p / min-p via a single descending vocab sort + cumsum
//       threshold, per-row parameters broadcast as [B, 1] tensors.
//
// Batch-composition invariance: every transform is row-independent (per-row
// reductions along the vocab axis only), so a row's transformed logits
// depend only on that row's raw logits, parameters, and history — never on
// batchmates.
//
// Threading: `process`/`commit` are engine-step-thread ops that only build
// graph nodes (no eval, no `.item()`, no host syncs). `setRows` runs at
// batch-membership change (join/leave) and may loop over rows on the host —
// that is the ONLY place per-row Swift loops are allowed.

import Foundation
import MLX

// MARK: - Row description (membership-change input)

public struct CBv2SamplerRow {
    public var id: CBv2RequestID
    public var params: CBv2SamplingParams
    public var promptTokens: [Int]
    public var outputTokens: [Int]
    public var tokenConstraint: (any CBv2TokenConstraint)?
    public var maxTokens: Int

    public init(
        id: CBv2RequestID, params: CBv2SamplingParams,
        promptTokens: [Int], outputTokens: [Int] = [],
        tokenConstraint: (any CBv2TokenConstraint)? = nil,
        maxTokens: Int = .max
    ) {
        self.id = id
        self.params = params
        self.promptTokens = promptTokens
        self.outputTokens = outputTokens
        self.tokenConstraint = tokenConstraint
        self.maxTokens = maxTokens
    }
}

// MARK: - Pipeline

public final class LogitsPipelineV2 {

    public static let greedyEpsilon: Float = 1e-5

    public struct Output {
        public let sampling: MLXArray
        public let rawLogprobs: MLXArray?
    }

    public let vocabSize: Int

    public private(set) var allGreedy = true
    public private(set) var wantsLogprobs = false
    public private(set) var logprobBuildCount = 0

    private var rowCount = 0
    private var anyBias = false
    private var anyRepetition = false
    private var anyFrequencyPresence = false
    private var anyTemperature = false
    private var anyTopKPMinP = false

    private var temperature: MLXArray?  // greedy rows pinned to 1.0
    private var repetitionPenalty: MLXArray?
    private var frequencyPenalty: MLXArray?
    private var presencePenalty: MLXArray?
    private var topK: MLXArray?  // [B, 1] int32; disabled rows = vocabSize
    private var topP: MLXArray?  // [B, 1]; disabled rows = 2.0 sentinel
    private var minP: MLXArray?  // [B, 1]; disabled rows = 0.0
    private var biasMatrix: MLXArray?  // [B, vocab] float32

    private var outputCounts: MLXArray?
    private var repWindowCounts: MLXArray?
    private var repWindow: MLXArray?
    private var ringPos: MLXArray?
    private var windowSizes: MLXArray?
    private var rowIndices: MLXArray?

    public init(vocabSize: Int) {
        precondition(vocabSize > 0, "vocabSize must be positive")
        self.vocabSize = vocabSize
    }

    // MARK: Membership change (host loops allowed here, and only here)

    public func setRows(_ rows: [CBv2SamplerRow]) {
        rowCount = rows.count
        guard !rows.isEmpty else {
            clearState()
            return
        }

        let b = rows.count
        var temps = [Float](repeating: 1, count: b)
        var repPs = [Float](repeating: 1, count: b)
        var freqPs = [Float](repeating: 0, count: b)
        var presPs = [Float](repeating: 0, count: b)
        var topKs = [Int32](repeating: Int32(vocabSize), count: b)
        var topPs = [Float](repeating: 2, count: b)
        var minPs = [Float](repeating: 0, count: b)
        var windows = [Int32](repeating: 1, count: b)

        allGreedy = true
        wantsLogprobs = false
        anyBias = false
        anyRepetition = false
        anyFrequencyPresence = false
        anyTemperature = false
        anyTopKPMinP = false

        for (i, row) in rows.enumerated() {
            let p = row.params
            let greedy = p.temperature < Self.greedyEpsilon
            if !greedy {
                allGreedy = false
                temps[i] = p.temperature
                if p.temperature != 1 { anyTemperature = true }
                if p.topK > 0, p.topK < vocabSize {
                    topKs[i] = Int32(p.topK)
                    anyTopKPMinP = true
                }
                if p.topP > 0, p.topP < 1 {
                    topPs[i] = p.topP
                    anyTopKPMinP = true
                } else if p.topP <= 0 {
                    topPs[i] = 0
                    anyTopKPMinP = true
                }
                if p.minP > 0 {
                    minPs[i] = min(p.minP, 1)
                    anyTopKPMinP = true
                }
            }
            if p.topLogprobs > 0 { wantsLogprobs = true }
            if !p.logitBias.isEmpty { anyBias = true }
            let windowedRepetition = p.repetitionPenalty != 1 && p.repetitionContextSize > 0
            if windowedRepetition {
                repPs[i] = p.repetitionPenalty
                anyRepetition = true
                windows[i] = Int32(max(1, p.repetitionContextSize))
            }
            if p.frequencyPenalty != 0 || p.presencePenalty != 0 {
                freqPs[i] = p.frequencyPenalty
                presPs[i] = p.presencePenalty
                anyFrequencyPresence = true
            }
        }

        temperature = anyTemperature ? MLXArray(temps).reshaped([b, 1]) : nil
        repetitionPenalty = anyRepetition ? MLXArray(repPs).reshaped([b, 1]) : nil
        frequencyPenalty = anyFrequencyPresence ? MLXArray(freqPs).reshaped([b, 1]) : nil
        presencePenalty = anyFrequencyPresence ? MLXArray(presPs).reshaped([b, 1]) : nil
        topK = anyTopKPMinP ? MLXArray(topKs).reshaped([b, 1]) : nil
        topP = anyTopKPMinP ? MLXArray(topPs).reshaped([b, 1]) : nil
        minP = anyTopKPMinP ? MLXArray(minPs).reshaped([b, 1]) : nil

        biasMatrix = anyBias ? Self.buildBiasMatrix(rows: rows, vocabSize: vocabSize) : nil

        rowIndices = MLXArray(Array(0 ..< Int32(b)))
        if anyFrequencyPresence {
            outputCounts = Self.buildCounts(
                tokenLists: rows.map(\.outputTokens), vocabSize: vocabSize)
        } else {
            outputCounts = nil
        }
        if anyRepetition {
            windowSizes = MLXArray(windows)
            let (window, positions) = Self.buildRepetitionWindow(rows: rows, windows: windows)
            repWindow = window
            ringPos = positions
            let windowTokens = rows.enumerated().map { i, row in
                Array((row.promptTokens + row.outputTokens).suffix(Int(windows[i])))
            }
            repWindowCounts = Self.buildCounts(
                tokenLists: windowTokens, vocabSize: vocabSize)
        } else {
            windowSizes = nil
            repWindow = nil
            ringPos = nil
            repWindowCounts = nil
        }
    }

    private func clearState() {
        allGreedy = true
        wantsLogprobs = false
        anyBias = false
        anyRepetition = false
        anyFrequencyPresence = false
        anyTemperature = false
        anyTopKPMinP = false
        temperature = nil
        repetitionPenalty = nil
        frequencyPenalty = nil
        presencePenalty = nil
        topK = nil
        topP = nil
        minP = nil
        biasMatrix = nil
        outputCounts = nil
        repWindowCounts = nil
        repWindow = nil
        ringPos = nil
        windowSizes = nil
        rowIndices = nil
    }

    // MARK: Step path — graph-build only, no host syncs, no per-row loops

    public func process(
        _ logits: MLXArray,
        rawLogprobsFrom rawLogits: MLXArray? = nil,
        hardMask: ((MLXArray) -> MLXArray)? = nil
    ) -> Output {
        precondition(
            logits.dim(0) == rowCount,
            "logits rows (\(logits.dim(0))) != configured rows (\(rowCount)) — call setRows")

        if allGreedy && !wantsLogprobs && !anyBias && !anyRepetition
            && !anyFrequencyPresence && !anyTemperature && !anyTopKPMinP
            && hardMask == nil
        {
            return Output(sampling: logits, rawLogprobs: nil)
        }

        var x = logits.asType(.float32)

        let rawLogprobs: MLXArray?
        if wantsLogprobs {
            logprobBuildCount += 1
            let raw = (rawLogits ?? logits).asType(.float32)
            rawLogprobs = raw - logSumExp(raw, axis: -1, keepDims: true)
        } else {
            rawLogprobs = nil
        }

        if let biasMatrix {
            x = x + biasMatrix
        }

        if let repetitionPenalty, let repWindowCounts {
            let present = repWindowCounts .> 0
            let penalized = which(x .> 0, x / repetitionPenalty, x * repetitionPenalty)
            x = which(present, penalized, x)
        }
        if let frequencyPenalty, let presencePenalty, let outputCounts {
            x = x - frequencyPenalty * outputCounts
            x = x - presencePenalty * (outputCounts .> 0).asType(.float32)
        }

        if let temperature {
            x = x / temperature
        }

        if let hardMask {
            x = hardMask(x)
        }

        if anyTopKPMinP, let topK, let topP, let minP {
            x = Self.applyTopKTopPMinP(x, topK: topK, topP: topP, minP: minP)
        }

        return Output(sampling: x, rawLogprobs: rawLogprobs)
    }

    public func commit(sampledTokens: MLXArray) {
        guard rowCount > 0, anyRepetition || anyFrequencyPresence else { return }
        precondition(
            sampledTokens.dim(0) == rowCount,
            "sampledTokens rows (\(sampledTokens.dim(0))) != configured rows (\(rowCount))")
        guard let rowIndices else { return }
        let tokens = sampledTokens.asType(.int32)

        if let counts = outputCounts {
            outputCounts = counts.at[rowIndices, tokens].add(Float(1))
        }

        if let window = repWindow, let positions = ringPos, let windowSizes,
            let counts = repWindowCounts
        {
            let slot = positions.expandedDimensions(axis: 1)  // [B, 1]
            let evicted = takeAlong(window, slot, axis: 1).squeezed(axis: 1)  // [B]
            let valid = (evicted .>= 0).asType(.float32)  // [B]
            let evictedIndex = maximum(evicted, MLXArray(Int32(0)))
            var updated = counts.at[rowIndices, evictedIndex].add(-valid)
            updated = updated.at[rowIndices, tokens].add(Float(1))
            repWindowCounts = updated
            repWindow = putAlong(
                window, slot, values: tokens.expandedDimensions(axis: 1), axis: 1)
            ringPos = remainder(positions + 1, windowSizes)
        }
    }

    // MARK: - Vectorized top-k/top-p/min-p

    static func applyTopKTopPMinP(
        _ logits: MLXArray, topK: MLXArray, topP: MLXArray, minP: MLXArray
    ) -> MLXArray {
        let vocab = logits.dim(-1)
        let sortedIndices = argSort(-logits, axis: -1)
        let sortedLogits = takeAlong(logits, sortedIndices, axis: -1)
        let probs = softmax(sortedLogits, axis: -1)
        let cumulative = cumsum(probs, axis: -1)

        let ranks = MLXArray(Array(0 ..< Int32(vocab))).reshaped([1, vocab])
        var keep = ranks .< topK
        keep = keep .&& ((cumulative - probs) .<= topP)
        let maxProb = probs[0..., ..<1]
        keep = keep .&& (probs .>= (minP * maxProb))

        let masked = which(keep, sortedLogits, MLXArray(-Float.infinity))

        let inverse = putAlong(
            MLXArray.zeros(like: sortedIndices),
            sortedIndices,
            values: broadcast(ranks.asType(sortedIndices.dtype), to: sortedIndices.shape),
            axis: -1
        )
        return takeAlong(masked, inverse, axis: -1)
    }

    // MARK: - Membership-change tensor builders

    private static func buildBiasMatrix(rows: [CBv2SamplerRow], vocabSize: Int) -> MLXArray {
        var rowIdx = [Int32]()
        var colIdx = [Int32]()
        var values = [Float]()
        for (i, row) in rows.enumerated() {
            for (token, bias) in row.params.logitBias where token >= 0 && token < vocabSize {
                rowIdx.append(Int32(i))
                colIdx.append(Int32(token))
                values.append(bias)
            }
        }
        let zeros = MLXArray.zeros([rows.count, vocabSize], type: Float.self)
        guard !rowIdx.isEmpty else { return zeros }
        return zeros.at[MLXArray(rowIdx), MLXArray(colIdx)].add(MLXArray(values))
    }

    private static func buildCounts(tokenLists: [[Int]], vocabSize: Int) -> MLXArray {
        var rowIdx = [Int32]()
        var colIdx = [Int32]()
        for (i, tokens) in tokenLists.enumerated() {
            for token in tokens where token >= 0 && token < vocabSize {
                rowIdx.append(Int32(i))
                colIdx.append(Int32(token))
            }
        }
        let zeros = MLXArray.zeros([tokenLists.count, vocabSize], type: Float.self)
        guard !rowIdx.isEmpty else { return zeros }
        return zeros.at[MLXArray(rowIdx), MLXArray(colIdx)].add(Float(1))
    }

    private static func buildRepetitionWindow(
        rows: [CBv2SamplerRow], windows: [Int32]
    ) -> (window: MLXArray, ringPos: MLXArray) {
        let b = rows.count
        let maxWindow = Int(windows.max() ?? 1)
        var flat = [Int32](repeating: -1, count: b * maxWindow)
        var positions = [Int32](repeating: 0, count: b)
        for (i, row) in rows.enumerated() {
            let w = Int(windows[i])
            let history = row.promptTokens + row.outputTokens
            let recent = history.suffix(w)
            for (j, token) in recent.enumerated() {
                flat[i * maxWindow + j] = token >= 0 ? Int32(token) : -1
            }
            positions[i] = recent.count == w ? 0 : Int32(recent.count)
        }
        return (MLXArray(flat, [b, maxWindow]), MLXArray(positions))
    }
}

// MARK: - Lazy logprob gather

public enum CBv2Logprobs {

    public struct Gathered {
        public let chosen: MLXArray
        public let topValues: MLXArray
        public let topIndices: MLXArray
    }

    public static func gather(
        rawLogprobs: MLXArray, sampledTokens: MLXArray, k: Int
    ) -> Gathered {
        let tokens = sampledTokens.asType(.int32).expandedDimensions(axis: 1)
        let chosen = takeAlong(rawLogprobs, tokens, axis: -1).squeezed(axis: 1)
        let vocab = rawLogprobs.dim(-1)
        let kEff = max(0, min(k, vocab))
        guard kEff > 0 else {
            let empty = MLXArray.zeros([rawLogprobs.dim(0), 0], type: Float.self)
            return Gathered(chosen: chosen, topValues: empty, topIndices: empty.asType(.int32))
        }
        let sortedIndices = argSort(-rawLogprobs, axis: -1)[0..., ..<kEff]
        let sortedValues = takeAlong(rawLogprobs, sortedIndices, axis: -1)
        return Gathered(
            chosen: chosen, topValues: sortedValues, topIndices: sortedIndices.asType(.int32))
    }

    public static func assemble(
        _ gathered: Gathered, sampledTokens: [Int], topLogprobsPerRow: [Int]
    ) -> [CBv2TokenLogprob?] {
        let chosen = gathered.chosen.asArray(Float.self)
        let k = gathered.topValues.dim(-1)
        let values = gathered.topValues.asArray(Float.self)
        let indices = gathered.topIndices.asArray(Int32.self)
        var out = [CBv2TokenLogprob?]()
        out.reserveCapacity(sampledTokens.count)
        for row in 0 ..< sampledTokens.count {
            let want = min(topLogprobsPerRow[row], k)
            guard want > 0 else {
                out.append(nil)
                continue
            }
            var top = [(token: Int, logprob: Float)]()
            top.reserveCapacity(want)
            for j in 0 ..< want {
                top.append((token: Int(indices[row * k + j]), logprob: values[row * k + j]))
            }
            out.append(
                CBv2TokenLogprob(
                    token: sampledTokens[row], logprob: chosen[row], topLogprobs: top))
        }
        return out
    }
}

// MARK: - Per-step logprob handoff (sampler → engine loop)

public struct CBv2StepLogprobs {
    public let rows: [CBv2RequestID]
    public let topLogprobsPerRow: [Int]
    public let gathered: CBv2Logprobs.Gathered

    public init(
        rows: [CBv2RequestID], topLogprobsPerRow: [Int], gathered: CBv2Logprobs.Gathered
    ) {
        self.rows = rows
        self.topLogprobsPerRow = topLogprobsPerRow
        self.gathered = gathered
    }

    public var evalTargets: [MLXArray] {
        [gathered.chosen, gathered.topValues, gathered.topIndices]
    }
}

// MARK: - Order-only logits (SOFTCAP-SKIP) [r2]

public enum CBv2OrderOnlyLogits {
    nonisolated(unsafe) private static var flag = false

    public static var engaged: Bool { flag }

    public static func set(_ value: Bool) { flag = value }

    public static func orderOnly(_ params: [CBv2SamplingParams]) -> Bool {
        guard !params.isEmpty else { return false }
        return params.allSatisfy { p in
            p.temperature < LogitsPipelineV2.greedyEpsilon
                && p.topLogprobs == 0
                && p.logitBias.isEmpty
                && !(p.repetitionPenalty != 1 && p.repetitionContextSize > 0)
                && p.frequencyPenalty == 0
                && p.presencePenalty == 0
        }
    }

    public static func withOrderOnly<T>(
        _ params: [CBv2SamplingParams], _ build: () -> T
    ) -> T {
        set(orderOnly(params))
        defer { set(false) }
        return build()
    }
}
