import Foundation
import Testing

@testable import BenchCBv2Core

@Suite("BenchCBv2 optimization provenance")
struct BenchCBv2ProvenanceTests {
    private let exactModel = ModelOptimizationProvenance(
        layer18Requested: true,
        layer18Effective: true,
        layer18Interval: 18,
        weightedUnsortRequested: true,
        weightedUnsortEffective: true,
        safeR1GeometryEligible: true)

    @Test func exactR1GeometryRequiresAHitOnNonNAXHosts() {
        let missing = SafeR1Provenance(
            requested: true, aotAvailable: true, naxAvailable: false,
            exactGeometryExpected: true, attempts: 1, hits: 0,
            fallbackMetallibUnavailable: 1)
        #expect(missing.guardFailure != nil)

        let hit = SafeR1Provenance(
            requested: true, aotAvailable: true, naxAvailable: false,
            exactGeometryExpected: true, attempts: 1, hits: 1)
        #expect(hit.guardFailure == nil)
    }

    @Test func naxPrecedenceRequiresAnObservedNAXRoute() {
        let missing = SafeR1Provenance(
            requested: true, aotAvailable: true, naxAvailable: true,
            exactGeometryExpected: true, attempts: 0, hits: 0)
        #expect(missing.guardFailure != nil)

        let routed = SafeR1Provenance(
            requested: true, aotAvailable: true, naxAvailable: true,
            exactGeometryExpected: true, attempts: 1, hits: 0,
            fallbackNAX: 1)
        #expect(routed.guardFailure == nil)
    }

    @Test func malformedOrUnrequestedR1CountersFailClosed() {
        let inconsistent = SafeR1Provenance(
            requested: true, aotAvailable: true, naxAvailable: false,
            exactGeometryExpected: false, attempts: 2, hits: 1)
        #expect(inconsistent.guardFailure != nil)

        let unrequested = SafeR1Provenance(
            requested: false, aotAvailable: true, naxAvailable: false,
            exactGeometryExpected: false, attempts: 1, hits: 1)
        #expect(unrequested.guardFailure != nil)
    }

    @Test func unarmedR1CountersFailClosed() {
        let unarmed = SafeR1Provenance(
            requested: false, aotAvailable: true, naxAvailable: false,
            armed: false, exactGeometryExpected: false, attempts: 0, hits: 0)
        #expect(unarmed.guardFailure?.contains("not armed") == true)
    }

    @Test func geometryGuardRecognizesOnlyEligibleExactPrefillChunks() {
        #expect(!safeR1ExactGeometryExpected(
            modelEligible: true, promptLengths: [500], prefillChunkSize: 512))
        #expect(safeR1ExactGeometryExpected(
            modelEligible: true, promptLengths: [100, 1500], prefillChunkSize: 512))
        #expect(safeR1ExactGeometryExpected(
            modelEligible: true, promptLengths: [2048], prefillChunkSize: 2048))
        #expect(safeR1ExactGeometryExpected(
            modelEligible: true, promptLengths: [256, 256], prefillChunkSize: 512))
        #expect(!safeR1ExactGeometryExpected(
            modelEligible: true, promptLengths: [512, 512, 512], prefillChunkSize: 512))
        #expect(safeR1ExactGeometryExpected(
            modelEligible: true,
            promptLengths: [512, 1500, 2000],
            prefillChunkSize: 512))
        #expect(safeR1ExactGeometryExpected(
            modelEligible: true,
            promptLengths: [512, 512, 512, 512],
            prefillChunkSize: 512))
        #expect(!safeR1ExactGeometryExpected(
            modelEligible: false, promptLengths: [2048], prefillChunkSize: 2048))
    }

    @Test func layerAndWeightedEngagementAreRequiredOnlyWhenEffective() {
        let r1Off = SafeR1Provenance(
            requested: false, aotAvailable: false, naxAvailable: false,
            exactGeometryExpected: false, attempts: 0, hits: 0)
        let missingLayer = CellOptimizationProvenance(
            model: exactModel,
            layer18IntermediateSubmissions: 0,
            layer18ExpectedMinimumSubmissions: 1,
            weightedUnsortEffectiveCalls: 1,
            safeR1: r1Off)
        #expect(missingLayer.guardFailure?.contains("layer18") == true)

        let missingWeighted = CellOptimizationProvenance(
            model: exactModel,
            layer18IntermediateSubmissions: 1,
            layer18ExpectedMinimumSubmissions: 1,
            weightedUnsortEffectiveCalls: 0,
            safeR1: r1Off)
        #expect(missingWeighted.guardFailure?.contains("weighted") == true)
    }

    @Test func markdownAndJSONExposeEveryR1FallbackAndTopologyState() throws {
        let r1 = SafeR1Provenance(
            requested: true, aotAvailable: false, naxAvailable: false,
            exactGeometryExpected: false, attempts: 7, hits: 0,
            fallbackNAX: 1, fallbackOuterRoute: 1,
            fallbackQuantization: 1, fallbackTopology: 1,
            fallbackAssignmentCount: 1, fallbackGeometry: 1,
            fallbackMetallibUnavailable: 1)
        let provenance = CellOptimizationProvenance(
            model: exactModel,
            layer18IntermediateSubmissions: 2,
            layer18ExpectedMinimumSubmissions: 1,
            weightedUnsortEffectiveCalls: 3,
            safeR1: r1)

        let markdown = provenance.markdown(scope: "perf/v2/B2")
        for field in [
            "weightedUnsort(requested=true, effective=true)",
            "fallbackNAX=1", "fallbackOuterRoute=1",
            "fallbackQuantization=1", "fallbackTopology=1",
            "fallbackAssignmentCount=1", "fallbackGeometry=1",
            "fallbackMetallibUnavailable=1",
        ] {
            #expect(markdown.contains(field))
        }

        let json = try benchmarkJSONString(provenance)
        for key in [
            "weightedUnsortRequested", "weightedUnsortEffective",
            "layer18Engaged", "weightedUnsortEngaged", "armed", "effective", "fallbacks",
            "fallbackNAX", "fallbackOuterRoute", "fallbackQuantization",
            "fallbackTopology", "fallbackAssignmentCount", "fallbackGeometry",
            "fallbackMetallibUnavailable",
        ] {
            #expect(json.contains("\"\(key)\""))
        }
    }

    @Test func missingCellProvenanceFailsClosed() {
        let cell = CellResult(
            engine: "v2", batch: 1, promptMix: [500],
            decodeTPSPerRequest: 1, aggregateTPS: 1,
            ttftP50Ms: 1, itlP50Ms: 1, perRequest: [],
            optimizationProvenance: nil)
        #expect(throws: (any Error).self) {
            _ = try optimizationProvenanceLines(cell, scope: "perf/v2/B1")
        }
    }

    @Test func performanceTableRowsRemainContiguousBeforeProvenance() {
        let rows = [
            "| v2 | 1 | 500 | 1 | 1 | 1 | 1 |",
            "| v2 | 2 | 100/1500 | 2 | 2 | 2 | 2 |",
            "| v2 | 4 | 100/500/1500/500 | 4 | 4 | 4 | 4 |",
        ]
        let provenance = [
            "- optimization provenance [perf/v2/B1]: one",
            "- optimization-cell-json [perf/v2/B1]: {}",
            "- optimization provenance [perf/v2/B2]: two",
            "- optimization-cell-json [perf/v2/B2]: {}",
            "- optimization provenance [perf/v2/B4]: four",
            "- optimization-cell-json [perf/v2/B4]: {}",
        ]

        let rendered = performanceMarkdown(
            rows: rows, provenanceLines: provenance)
        let lines = rendered.split(
            separator: "\n", omittingEmptySubsequences: false).map(String.init)

        #expect(Array(lines[2 ..< 5]) == rows)
        #expect(lines[5].isEmpty)
        #expect(Array(lines[6...]) == provenance)
    }
}
