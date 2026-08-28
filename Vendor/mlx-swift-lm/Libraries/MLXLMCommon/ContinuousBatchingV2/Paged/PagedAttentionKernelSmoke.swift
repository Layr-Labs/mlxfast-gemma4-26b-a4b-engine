// PagedAttentionKernelSmoke.swift
//
// Pre-traffic compilation/dispatch coverage for paged Metal kernels.

import Foundation
import MLX
import MLXRandom

public struct PagedAttentionKernelSmokeShape: Hashable, Sendable {
    public let headDim: Int
    public let kvHeads: Int
    public let queryHeads: Int
    public let hasSinks: Bool
    public let hasWrite: Bool

    public init(
        headDim: Int,
        kvHeads: Int,
        queryHeads: Int,
        hasSinks: Bool,
        hasWrite: Bool
    ) {
        self.headDim = headDim
        self.kvHeads = kvHeads
        self.queryHeads = queryHeads
        self.hasSinks = hasSinks
        self.hasWrite = hasWrite
    }

    public var argumentValue: String {
        [
            headDim,
            kvHeads,
            queryHeads,
            hasSinks ? 1 : 0,
            hasWrite ? 1 : 0,
        ].map(String.init).joined(separator: ":")
    }

    public init(argumentValue: String) throws {
        let fields = argumentValue.split(
            separator: ":",
            omittingEmptySubsequences: false)
        guard fields.count == 5,
            let headDim = Int(fields[0]),
            let kvHeads = Int(fields[1]),
            let queryHeads = Int(fields[2]),
            let sinks = Int(fields[3]),
            let write = Int(fields[4]),
            (sinks == 0 || sinks == 1),
            (write == 0 || write == 1)
        else {
            throw PagedAttentionKernelSmokeError.invalidShape(argumentValue)
        }
        self.init(
            headDim: headDim,
            kvHeads: kvHeads,
            queryHeads: queryHeads,
            hasSinks: sinks == 1,
            hasWrite: write == 1)
    }
}

public enum PagedAttentionKernelSmokeError: Error, CustomStringConvertible {
    case invalidShape(String)
    case ineligibleShape(String)

    public var description: String {
        switch self {
        case .invalidShape(let value):
            return "invalid paged-kernel smoke shape \(value)"
        case .ineligibleShape(let reason):
            return "ineligible paged-kernel smoke shape: \(reason)"
        }
    }
}

/// Partition rungs (`PagedAttentionKernel.partitionTokenLadder` values) the
/// runtime smoke actually dispatched, keyed by the shape that dispatched
/// them. Each entry is recorded at the dispatch site from the
/// `maxAttendLength` that dispatch handed to `decode`, so it reports what
/// was compiled rather than what was planned.
public typealias PagedAttentionKernelSmokeCoverage =
    [PagedAttentionKernelSmokeShape: Set<Int>]

extension PagedAttentionKernel {
    /// Canonical production GPT-OSS shape. Full and sliding layers share
    /// the same compiled kernel specialization.
    public static let gptOSSRuntimeSmokeShapes = [
        PagedAttentionKernelSmokeShape(
            headDim: 64,
            kvHeads: 8,
            queryHeads: 64,
            hasSinks: true,
            hasWrite: true)
    ]

    public static func smokeShapes(
        layerKinds: [CBv2LayerKind]
    ) -> [PagedAttentionKernelSmokeShape] {
        Array(Set(layerKinds.map {
            PagedAttentionKernelSmokeShape(
                headDim: $0.headDim,
                kvHeads: $0.kvHeads,
                queryHeads: $0.queryHeads,
                hasSinks: $0.hasSinks,
                hasWrite: $0.sharesKVWithLayer == nil)
        })).sorted {
            (
                $0.headDim,
                $0.kvHeads,
                $0.queryHeads,
                $0.hasSinks ? 1 : 0,
                $0.hasWrite ? 1 : 0
            ) < (
                $1.headDim,
                $1.kvHeads,
                $1.queryHeads,
                $1.hasSinks ? 1 : 0,
                $1.hasWrite ? 1 : 0
            )
        }
    }

    /// Catchable resource preflight used by backend construction and
    /// packaged-artifact verification. A packaged process searches only
    /// its sealed app resources.
    public static func validateRuntimeResources() throws {
        _ = try PagedAttentionResources.loadSourceForCurrentProcess()
    }

    /// Pre-JIT every paged kernel specialization a model can dispatch:
    /// prefill/adoption bulk-write, decode part (fused-write or borrowing),
    /// and decode merge (with the model's sink specialization). Every output
    /// is evaluated so compilation and dispatch complete before traffic.
    ///
    /// Decode is dispatched once per PARTITION RUNG the sizer can select
    /// for the shape (`smokeAttendLengths`). PTOK is a kernel TEMPLATE
    /// parameter, so a rung the smoke skips is a Metal JIT charged to the
    /// first production request whose context bucket selects it — a TTFT
    /// spike on a real request, which is the whole thing this exists to
    /// prevent.
    ///
    /// - Returns: the rungs actually dispatched, per shape.
    @discardableResult
    public static func runtimeSmoke(
        shapes: [PagedAttentionKernelSmokeShape] = gptOSSRuntimeSmokeShapes
    ) throws -> PagedAttentionKernelSmokeCoverage {
        let source = try PagedAttentionResources.loadSourceForCurrentProcess()
        return try runtimeSmoke(source: source, shapes: shapes)
    }

    @discardableResult
    static func runtimeSmokeForTesting(
        searchRoots: [URL],
        shapes: [PagedAttentionKernelSmokeShape] = gptOSSRuntimeSmokeShapes
    ) throws -> PagedAttentionKernelSmokeCoverage {
        let source = try PagedAttentionResources.loadSource(roots: searchRoots)
        return try runtimeSmoke(source: source, shapes: shapes)
    }

    /// Attended lengths that walk `partitionTokensForDispatch` across every
    /// partition rung it can return for one dispatch shape, ascending.
    ///
    /// The ladder is NOT enumerated here and the sizer's selection is NOT
    /// reimplemented. Candidates are generated as `rung * partitions` —
    /// that is the definition of a range split into partitions, not a
    /// selection rule — and every candidate is CLASSIFIED by calling the
    /// sizer. Add a rung to `partitionTokenLadder`, or change how a rung is
    /// picked, and this plan follows with no edit here. Rungs the sizer
    /// cannot reach are correctly absent: a page size that filters a rung,
    /// or `DARKBLOOM_CBV2_PAGED_PTOK_TARGET=0` pinning every dispatch to
    /// `partitionTokens`, means production cannot dispatch them either, so
    /// compiling them would cost JIT for a variant nothing will use.
    ///
    /// Exhaustive by construction: the sizer aims at `target` threadgroups
    /// and a dispatch already launches at least one per (kv head, head
    /// split, row), so the partition count it wants never exceeds that
    /// target — sweeping the multiplier over `1...target` therefore probes
    /// `rung * wantedPartitions`, which lands inside every reachable rung's
    /// band. Ascending multiplier order makes the length recorded for a
    /// rung the SHORTEST probe that selects it, which is what keeps the
    /// extra dispatches cheap.
    ///
    /// The sweep is bounded ONLY by the target, with no separate cap of its
    /// own: a cap here that the production sizer does not share is a rung
    /// the smoke cannot reach and traffic can, which is the JIT spike this
    /// exists to prevent. `PagedAttentionKernel.partitionTargetLimit` bounds
    /// both sides at the one place the operator's value enters the process.
    static func smokeAttendLengths(
        kvHeads: Int, headSplits: Int, pageSize: Int,
        target: Int = partitionTargetThreadgroups
    ) -> [Int] {
        let maxPartitions = max(1, target)
        var lengthByRung: [Int: Int] = [:]
        for partitions in 1 ... maxPartitions {
            for rung in partitionTokenLadder {
                let length = rung * partitions
                let ptok = partitionTokensForDispatch(
                    maxAttendLength: length, batch: 1, kvHeads: kvHeads,
                    headSplits: headSplits, pageSize: pageSize, target: target)
                if lengthByRung[ptok] == nil { lengthByRung[ptok] = length }
            }
            if lengthByRung.count == partitionTokenLadder.count { break }
        }
        return lengthByRung.values.sorted()
    }

    /// Physical layout of one smoke decode probe: a history table that maps
    /// the whole attended range onto a single zeroed page, plus a fused
    /// write that lands on a page the table never names.
    ///
    /// The separation is REQUIRED, not tidiness. The part kernel's
    /// single-writer argument (pagedattention.metal, "In-place slab writes")
    /// holds only because no threadgroup reads the bytes the fused write
    /// stores. A probe that both maps every logical page to physical page 0
    /// and writes its tile to page 0 slot 0 breaks that: every absolute
    /// position that is a multiple of the page size resolves to exactly the
    /// written address, and the earlier partitions reading it run
    /// concurrently with the last partition's store. The values are all
    /// zero, but zeros are not ordering — the race is real and it is the
    /// production invariant the smoke is supposed to rehearse.
    struct SmokeProbe {
        /// `[1, columns]` block table, every entry the shared history page.
        let table: [Int32]
        /// `[1, 8]` row info `{attendStart, attendLen, tableLen, writePage,
        /// writeSlot, 0…}`.
        let seqinfo: [Int32]
        var writePage: Int32 { seqinfo[3] }
        var writeSlot: Int32 { seqinfo[4] }
    }

    /// The single physical page every probe's history table points at. Its
    /// contents are zeros, so every gathered row is in bounds and the
    /// attention arithmetic is meaningless but safe.
    static let smokeHistoryPage: Int32 = 0

    /// Scratch page for the fused decode write — absent from every probe
    /// table, so no threadgroup in the dispatch reads what it stores.
    static let smokeWritePage: Int32 = 1

    /// Physical pages a probe's slabs must hold.
    static let smokeSlabPages = Int(max(smokeHistoryPage, smokeWritePage)) + 1

    static func smokeProbe(attendLength: Int, pageSize: Int) -> SmokeProbe {
        // The shader gathers through `table[logicalPage % tableLen]`
        // (pagedattention.metal), so the table needs `tableLen` columns —
        // padded to the 8 the kernel signature demands — and not one more.
        let tableLength = (attendLength + pageSize - 1) / pageSize
        let columns = max(tableLength, 8)
        return SmokeProbe(
            table: [Int32](repeating: smokeHistoryPage, count: columns),
            seqinfo: PagedAttentionKernel.SeqInfoRow(
                attendStart: 0, attendLength: attendLength, tableLength: tableLength,
                writePage: smokeWritePage
            ).packed)
    }

    private static func runtimeSmoke(
        source: String,
        shapes: [PagedAttentionKernelSmokeShape]
    ) throws -> PagedAttentionKernelSmokeCoverage {
        guard !shapes.isEmpty else {
            throw PagedAttentionKernelSmokeError.invalidShape("empty")
        }

        let pageSize = CBv2PagedDefaults.pageSize
        let slots = MLXArray([Int32](repeating: 0, count: 8))
        var coverage: PagedAttentionKernelSmokeCoverage = [:]
        for shape in shapes {
            guard shape.kvHeads > 0,
                shape.queryHeads > 0,
                shape.queryHeads % shape.kvHeads == 0
            else {
                throw PagedAttentionKernelSmokeError.invalidShape(
                    shape.argumentValue)
            }
            let gqa = shape.queryHeads / shape.kvHeads
            if let reason = ineligibilityReason(
                headDim: shape.headDim,
                gqa: gqa)
            {
                throw PagedAttentionKernelSmokeError.ineligibleShape(reason)
            }

            let kSlab = MLXArray.zeros(
                [smokeSlabPages, shape.kvHeads, pageSize, shape.headDim],
                dtype: .float16)
            let vSlab = MLXArray.zeros(
                [smokeSlabPages, shape.kvHeads, pageSize, shape.headDim],
                dtype: .float16)
            let writeTile = MLXArray.zeros(
                [shape.kvHeads, 1, shape.headDim],
                dtype: .float16)
            var fence = bulkWrite(
                kSlab: kSlab,
                vSlab: vSlab,
                keys: writeTile,
                values: writeTile,
                slots: slots,
                prevFence: MLXArray.zeros([1], dtype: .int32),
                pageSize: pageSize,
                kernelSource: source)
            eval(fence)

            let queries = MLXArray.zeros(
                [1, shape.queryHeads, 1, shape.headDim],
                dtype: .float16)
            let decodeTile = MLXArray.zeros(
                [1, shape.kvHeads, shape.headDim],
                dtype: .float16)
            let sinks = shape.hasSinks
                ? MLXRandom.normal(
                    [max(shape.queryHeads, 8)],
                    dtype: .float32)
                : nil
            if let sinks {
                eval(sinks)
            }
            let params = MLXArray([Float(1), 0.125, 0, 0, 0, 0, 0, 0])
            let headSplits =
                gqa / headsPerThreadgroup(headDim: shape.headDim, gqa: gqa)

            for attendLength in smokeAttendLengths(
                kvHeads: shape.kvHeads,
                headSplits: headSplits,
                pageSize: pageSize)
            {
                let probe = smokeProbe(
                    attendLength: attendLength, pageSize: pageSize)
                let tables = MLXArray(probe.table)
                    .reshaped([1, probe.table.count])
                let seqinfo = MLXArray(probe.seqinfo, [1, 8])
                let result = decode(
                    queries: queries,
                    newKeys: shape.hasWrite ? decodeTile : nil,
                    newValues: shape.hasWrite ? decodeTile : nil,
                    kSlab: kSlab,
                    vSlab: vSlab,
                    tables: tables,
                    seqinfo: seqinfo,
                    maxAttendLength: attendLength,
                    sinks: sinks,
                    params: params,
                    softcap: false,
                    pageSize: pageSize,
                    writeFence: fence,
                    kernelSource: source)
                // Read the rung back from the sizer using the exact
                // `maxAttendLength` THIS dispatch passed, so the record
                // tracks what was compiled rather than what was planned.
                coverage[shape, default: []].insert(
                    partitionTokensForDispatch(
                        maxAttendLength: attendLength, batch: 1,
                        kvHeads: shape.kvHeads, headSplits: headSplits,
                        pageSize: pageSize))
                if let nextFence = result.nextWriteFence {
                    // The fused write advanced the group's chain; the next
                    // dispatch must order after it.
                    eval(result.out, nextFence)
                    fence = nextFence
                } else {
                    eval(result.out)
                }
            }
        }
        return coverage
    }
}
