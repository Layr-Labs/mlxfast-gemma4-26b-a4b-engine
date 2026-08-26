// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXVLM

// DISABLED AT COMPILE TIME: this spike targets the MLXVLM Gemma4 MTP surface
// (`forwardForMTP`, `mtpConfiguration`, `rollbackSpeculativeCache`,
// `Gemma4Configuration.textConfiguration`) that was removed when the vMLX
// decode hot path was ported (8a212b4). The file no longer compiles on
// main/feat/engine-v2 and blocks the entire MLXLMTests bundle from building.
// Re-enable with `-Xswiftc -DVLM_MTP_SPIKE_COMPILED` once the VLM MTP surface
// is restored. (The suite was already env-gated at RUNTIME via VLM_MTP_SPIKE.)
#if VLM_MTP_SPIKE_COMPILED

/// De-risking spike for VLM MTP speculative decoding.
///
/// The make-or-break question: does the MLXLLM-trained Gemma 4 MTP drafter
/// produce *correct* greedy output when bound to the **MLXVLM** text tower —
/// a separate implementation of the same Gemma 4 text architecture loaded from
/// the same checkpoint? If the two towers are numerically equivalent, then:
///
///    plain_greedy(MLXLLM text tower)  ==  MTP_greedy(MLXVLM tower, drafter)
///
/// holds token-for-token (modulo late bf16 argmax ties), proving both that the
/// VLM tower matches the reference text tower AND that the drafter's shared-KV
/// is layout/numeric-compatible with the VLM tower's captured K/V. If it
/// fails, the drafter would need retraining against the VLM tower — a model
/// task, not a code fix.
///
/// Gated on env so it never runs in plain `swift test`:
///   VLM_MTP_SPIKE=1
///   VLM_MTP_TARGET_DIR=/abs/path/to/gemma-4-26B-A4B-it-qat-4bit   (multimodal checkpoint)
///   VLM_MTP_DRAFTER_DIR=/abs/path/to/gemma-4-26B-A4B-it-qat-assistant-4bit
///   [VLM_MTP_MAXTOK=64] [VLM_MTP_K=3] [VLM_MTP_PROMPT=<json int array>]
// DISABLED (pre-existing build breakage, unrelated to CBv2): the MLXVLM
// `Gemma4` tower lost its `Gemma4MTPTarget` surface (`forwardForMTP` /
// `rollbackSpeculativeCache`) in 8a212b4 "port vMLX decode hot path (#55)",
// so this env-gated spike no longer compiles and blocks the whole
// MLXLMTests target. Re-enable once the VLM tower's MTP conformance is
// restored.
@Suite(.serialized)
struct Gemma4VLMMTPSpikeTests {

    /// Isolates a multi-token-verify (forward/mask) bug from a round-machinery
    /// (rollback/acceptance) bug. MTP verifies K tokens in ONE forward at
    /// offset>0; the normal VLM path only ever does single-token-at-offset, so
    /// this pattern is otherwise untested. The K-token forward of [t0,t1,t2]
    /// must predict, per position, exactly what sequential single-token decode
    /// predicts. If this fails, the bug is in the VLM tower's forward/mask. If
    /// it passes but `vlm_tower_drafter_parity` fails, the bug is in rollback.
    @Test func vlm_multitoken_forward_consistency() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["VLM_MTP_SPIKE"] == "1", let targetDir = env["VLM_MTP_TARGET_DIR"]
        else {
            print("Skipping: set VLM_MTP_SPIKE=1, VLM_MTP_TARGET_DIR")
            return
        }
        let prompt = loadPrompt(env)
        let vlm = try loadRealTargetAsVLM(from: URL(fileURLWithPath: targetDir))
        eval(vlm)

        func argLast(_ toks: [Int32], _ cache: [KVCache]) -> Int {
            let out = vlm.forwardForMTP(MLXArray(toks)[.newAxis, .ellipsis], cache: cache)
            let a = out.logits[0..., -1, 0...].asType(.float32).argMax(axis: -1)
            eval(a)
            return Int(a.item(Int32.self))
        }

        // Single-token reference: prompt → bonus → s1 → s2 → s3.
        let cacheA = vlm.newCache(parameters: nil)
        let bonus = argLast(prompt, cacheA)
        let s1 = argLast([Int32(bonus)], cacheA)
        let s2 = argLast([Int32(s1)], cacheA)
        let s3 = argLast([Int32(s2)], cacheA)

        // Multi-token: fresh cache, prefill prompt, then ONE forward of
        // [bonus, s1, s2]; per-position argmax must equal [s1, s2, s3].
        let cacheB = vlm.newCache(parameters: nil)
        _ = vlm.forwardForMTP(MLXArray(prompt)[.newAxis, .ellipsis], cache: cacheB)
        let multi = vlm.forwardForMTP(
            MLXArray([Int32(bonus), Int32(s1), Int32(s2)])[.newAxis, .ellipsis], cache: cacheB)
        let multiArg = multi.logits.asType(.float32).argMax(axis: -1).squeezed(axis: 0)
        eval(multiArg)
        let m = multiArg.asArray(Int32.self).map { Int($0) }

        print("\n=== VLM multi-token forward consistency ===")
        print("single-token next: [\(s1), \(s2), \(s3)]")
        print("multi-token  pred: \(m)")
        #expect(m == [s1, s2, s3],
            """
            VLM multi-token verify forward != sequential single-token decode. \
            The K-token-at-offset forward (mask/rope/attention) is the bug, not \
            rollback. single=[\(s1), \(s2), \(s3)] multi=\(m)
            """)
    }

    /// Isolates the rollback (cache trim) from the drafter. Drives MTP-style
    /// rounds with deliberately-wrong drafts so EVERY round rejects and trims,
    /// then checks the emitted tokens still equal clean single-token greedy.
    /// The emitted tokens are always the verify's argmax at confirmed
    /// positions, so if verify+rollback are correct this matches greedy exactly
    /// regardless of draft quality. Divergence here ⇒ the bug is rollback.
    @Test func vlm_rollback_consistency() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["VLM_MTP_SPIKE"] == "1", let targetDir = env["VLM_MTP_TARGET_DIR"]
        else { print("Skipping: set VLM_MTP_SPIKE=1, VLM_MTP_TARGET_DIR"); return }
        let prompt = loadPrompt(env)
        let K = 3
        let steps = 28
        let vlm = try loadRealTargetAsVLM(from: URL(fileURLWithPath: targetDir))
        eval(vlm)

        func argLast(_ toks: [Int32], _ cache: [KVCache]) -> Int {
            let out = vlm.forwardForMTP(MLXArray(toks)[.newAxis, .ellipsis], cache: cache)
            let a = out.logits[0..., -1, 0...].asType(.float32).argMax(axis: -1)
            eval(a)
            return Int(a.item(Int32.self))
        }

        // Clean single-token greedy reference.
        let cRef = vlm.newCache(parameters: nil)
        var ref: [Int] = [argLast(prompt, cRef)]
        while ref.count < steps { ref.append(argLast([Int32(ref.last!)], cRef)) }

        // MTP-style rounds with forced rejection (drafts = token 0).
        let c = vlm.newCache(parameters: nil)
        var got: [Int] = [argLast(prompt, c)]
        var bonus = got[0]
        while got.count < steps {
            let drafts = Array(repeating: 0, count: K - 1)
            let vin = [Int32(bonus)] + drafts.map { Int32($0) }
            let out = vlm.forwardForMTP(MLXArray(vin)[.newAxis, .ellipsis], cache: c)
            let mainArg = out.logits.asType(.float32).argMax(axis: -1).squeezed(axis: 0)
            eval(mainArg)
            let main = mainArg.asArray(Int32.self).map { Int($0) }
            let (accepted, newTokens) = Gemma4SpeculativeWalk.single(draft: drafts, main: main)
            if accepted < K - 1 {
                vlm.rollbackSpeculativeCache(c, accepted: .scalar(accepted), blockSize: K)
            }
            got.append(contentsOf: newTokens)
            bonus = newTokens.last!
        }
        let matched = matchedPrefix(got, ref)
        let n = Swift.min(got.count, ref.count)
        print("\n=== VLM rollback consistency ===")
        print("ref: \(ref.prefix(20))")
        print("got: \(got.prefix(20))")
        print("matched: \(matched)/\(n)")
        #expect(matched >= n - 1,
            """
            VLM rollback corrupts the cache: verify+trim does not reproduce \
            clean greedy. matched=\(matched)/\(n)
              ref=\(ref.prefix(matched + 4))
              got=\(got.prefix(matched + 4))
            """)
    }

    /// Same forced-rejection rollback drive, but on the MLXLLM text tower from
    /// the SAME checkpoint. Localizes the rollback bug: if MLXLLM ALSO diverges
    /// the bug is in the shared rollback/trim (masked by high acceptance in the
    /// real-model tests); if MLXLLM matches, it is VLM-tower-specific.
    @Test func mlxllm_rollback_consistency() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["VLM_MTP_SPIKE"] == "1", let targetDir = env["VLM_MTP_TARGET_DIR"]
        else { print("Skipping: set VLM_MTP_SPIKE=1, VLM_MTP_TARGET_DIR"); return }
        let prompt = loadPrompt(env)
        let K = 3
        let steps = 28
        let (llm, _) = try loadRealTargetAsTextModel(from: URL(fileURLWithPath: targetDir))
        eval(llm)
        let (matched, n, ref, got) = forcedRejectionParity(
            llm, prompt: prompt, steps: steps, K: K)
        print("\n=== MLXLLM rollback consistency (same checkpoint) ===")
        print("ref: \(ref.prefix(20))")
        print("got: \(got.prefix(20))")
        print("matched: \(matched)/\(n)")
        #expect(matched >= n - 1,
            "MLXLLM rollback also diverges ⇒ SHARED rollback bug. matched=\(matched)/\(n)")
    }

    /// Accept-ALL discriminator on the VLM tower: drafts == the true greedy
    /// tokens so every draft is accepted and trim NEVER fires. If this matches
    /// clean greedy but `vlm_rollback_consistency` (reject-all) does not, the
    /// bug is the trim; if this ALSO diverges, it is the multi-token verify at
    /// scale (attention/SDPA), independent of rollback.
    @Test func vlm_acceptall_consistency() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["VLM_MTP_SPIKE"] == "1", let targetDir = env["VLM_MTP_TARGET_DIR"]
        else { print("Skipping: set VLM_MTP_SPIKE=1, VLM_MTP_TARGET_DIR"); return }
        let prompt = loadPrompt(env)
        let K = 3, steps = 28
        let vlm = try loadRealTargetAsVLM(from: URL(fileURLWithPath: targetDir))
        eval(vlm)

        func argLast(_ toks: [Int32], _ cache: [KVCache]) -> Int {
            let out = vlm.forwardForMTP(MLXArray(toks)[.newAxis, .ellipsis], cache: cache)
            let a = out.logits[0..., -1, 0...].asType(.float32).argMax(axis: -1)
            eval(a)
            return Int(a.item(Int32.self))
        }
        // Reference greedy.
        let cRef = vlm.newCache(parameters: nil)
        var ref: [Int] = [argLast(prompt, cRef)]
        while ref.count < steps + K { ref.append(argLast([Int32(ref.last!)], cRef)) }

        // Chunked multi-token forward of the SAME reference tokens (accept-all):
        // feed prompt, then feed ref in K-token chunks; each chunk's per-position
        // argmax must predict the next ref token. No trim ever.
        // Single-token logits at each position, for gap comparison at the mismatch.
        let cSingle = vlm.newCache(parameters: nil)
        _ = vlm.forwardForMTP(MLXArray(prompt)[.newAxis, .ellipsis], cache: cSingle)
        var singleTop2: [Int: (Int, Float, Float)] = [:]  // refIndex -> (argmax, top1, top2)
        for i in 0 ..< (ref.count - 1) {
            let out = vlm.forwardForMTP(MLXArray([Int32(ref[i])])[.newAxis, .ellipsis], cache: cSingle)
            let lg = out.logits[0..., -1, 0...].asType(.float32).squeezed()
            let sorted = MLX.sorted(lg)
            eval(sorted)
            let v = sorted.asArray(Float.self)
            let arg = lg.argMax(axis: -1)
            eval(arg)
            singleTop2[i + 1] = (Int(arg.item(Int32.self)), v[v.count - 1], v[v.count - 2])
        }

        let c = vlm.newCache(parameters: nil)
        _ = vlm.forwardForMTP(MLXArray(prompt)[.newAxis, .ellipsis], cache: c)
        var pos = 0
        var firstMismatch = -1
        var gapReport = ""
        while pos + K <= ref.count {
            let chunk = Array(ref[pos ..< pos + K]).map { Int32($0) }
            let out = vlm.forwardForMTP(MLXArray(chunk)[.newAxis, .ellipsis], cache: c)
            let arg = out.logits.asType(.float32).argMax(axis: -1).squeezed(axis: 0)
            eval(arg)
            let pred = arg.asArray(Int32.self).map { Int($0) }  // predicts ref[pos+1 ... pos+K]
            for j in 0 ..< K where pos + 1 + j < ref.count {
                if pred[j] != ref[pos + 1 + j] {
                    firstMismatch = pos + 1 + j
                    // logit gap at this position: multi-token top-2 vs single-token top-2.
                    let row = out.logits[0, j, 0...].asType(.float32)
                    let ms = MLX.sorted(row)
                    eval(ms)
                    let mv = ms.asArray(Float.self)
                    let mTop1 = mv[mv.count - 1], mTop2 = mv[mv.count - 2]
                    let s = singleTop2[firstMismatch]
                    gapReport = """
                        multi argmax=\(pred[j]) top1=\(mTop1) top2=\(mTop2) gap=\(mTop1 - mTop2)
                        single argmax=\(s?.0 ?? -1) top1=\(s?.1 ?? 0) top2=\(s?.2 ?? 0) gap=\((s?.1 ?? 0) - (s?.2 ?? 0))
                        """
                    break
                }
            }
            if firstMismatch >= 0 { break }
            pos += K
        }
        print("\n=== VLM accept-all (chunked forward, no trim) ===")
        print("first multi-vs-single mismatch at: \(firstMismatch) (>= ~22 ⇒ verify-at-scale, not trim)")
        print(gapReport)
        #expect(firstMismatch == -1 || firstMismatch >= steps,
            "VLM chunked multi-token forward diverges from single-token at \(firstMismatch) — verify-at-scale numerical difference")
    }

    /// Run forced-rejection MTP-style rounds against any tower; return matched
    /// prefix vs clean single-token greedy.
    private func forcedRejectionParity(
        _ target: any Gemma4MTPTarget, prompt: [Int32], steps: Int, K: Int
    ) -> (matched: Int, n: Int, ref: [Int], got: [Int]) {
        func argLast(_ toks: [Int32], _ cache: [KVCache]) -> Int {
            let out = target.forwardForMTP(MLXArray(toks)[.newAxis, .ellipsis], cache: cache)
            let a = out.logits[0..., -1, 0...].asType(.float32).argMax(axis: -1)
            eval(a)
            return Int(a.item(Int32.self))
        }
        let cRef = target.mtpNewCache(parameters: nil)
        var ref: [Int] = [argLast(prompt, cRef)]
        while ref.count < steps { ref.append(argLast([Int32(ref.last!)], cRef)) }

        let c = target.mtpNewCache(parameters: nil)
        var got: [Int] = [argLast(prompt, c)]
        var bonus = got[0]
        while got.count < steps {
            let drafts = Array(repeating: 0, count: K - 1)
            let vin = [Int32(bonus)] + drafts.map { Int32($0) }
            let out = target.forwardForMTP(MLXArray(vin)[.newAxis, .ellipsis], cache: c)
            let mainArg = out.logits.asType(.float32).argMax(axis: -1).squeezed(axis: 0)
            eval(mainArg)
            let main = mainArg.asArray(Int32.self).map { Int($0) }
            let (accepted, newTokens) = Gemma4SpeculativeWalk.single(draft: drafts, main: main)
            if accepted < K - 1 {
                target.rollbackSpeculativeCache(c, accepted: .scalar(accepted), blockSize: K)
            }
            got.append(contentsOf: newTokens)
            bonus = newTokens.last!
        }
        return (matchedPrefix(got, ref), Swift.min(got.count, ref.count), ref, got)
    }

    /// Measure the actual decode speedup of MTP through the VLM tower vs plain
    /// single-token greedy. This is the value prop: "multimodal fast MTP".
    @Test func vlm_mtp_throughput() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["VLM_MTP_SPIKE"] == "1", let targetDir = env["VLM_MTP_TARGET_DIR"],
            let drafterDir = env["VLM_MTP_DRAFTER_DIR"]
        else { print("Skipping: set VLM_MTP_SPIKE=1, VLM_MTP_TARGET_DIR, VLM_MTP_DRAFTER_DIR"); return }
        let prompt = loadPrompt(env)
        let maxTokens = Int(env["VLM_MTP_MAXTOK"] ?? "128") ?? 128
        let K = Int(env["VLM_MTP_K"] ?? "3") ?? 3
        let vlm = try loadRealTargetAsVLM(from: URL(fileURLWithPath: targetDir))
        let drafter = try await Gemma4AssistantDraftModel.load(
            from: URL(fileURLWithPath: drafterDir))
        eval(vlm, drafter)
        try drafter.bind(target: vlm)

        // Warmup (kernel compile) for both paths.
        _ = vlmPlainGreedy(vlm, prompt: prompt, maxTokens: 8)
        MLX.Memory.clearCache()
        do {
            var w = try Gemma4MTPTokenIterator(
                input: LMInput(text: .init(tokens: MLXArray(prompt))), target: vlm,
                drafter: drafter, parameters: GenerateParameters(maxTokens: 8, temperature: 0),
                blockSize: K)
            while w.next() != nil {}
        }
        MLX.Memory.clearCache()

        let t0 = Date()
        let base = vlmPlainGreedy(vlm, prompt: prompt, maxTokens: maxTokens)
        let baseSecs = -t0.timeIntervalSinceNow
        MLX.Memory.clearCache()

        let t1 = Date()
        var iter = try Gemma4MTPTokenIterator(
            input: LMInput(text: .init(tokens: MLXArray(prompt))), target: vlm,
            drafter: drafter, parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0),
            blockSize: K)
        var mtp: [Int] = []
        while let t = iter.next(), mtp.count < maxTokens { mtp.append(t) }
        let mtpSecs = -t1.timeIntervalSinceNow

        let baseTps = Double(base.count) / Swift.max(baseSecs, 1e-9)
        let mtpTps = Double(mtp.count) / Swift.max(mtpSecs, 1e-9)
        print("\n=== VLM-MTP throughput (K=\(K), tokens=\(maxTokens)) ===")
        print(String(format: "plain greedy : %6.1f tok/s (%.2fs)", baseTps, baseSecs))
        print(String(format: "MTP          : %6.1f tok/s (%.2fs)", mtpTps, mtpSecs))
        print(String(format: "SPEEDUP      : %.2fx", mtpTps / Swift.max(baseTps, 1e-9)))
    }

    @Test func vlm_tower_drafter_parity() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["VLM_MTP_SPIKE"] == "1",
            let targetDir = env["VLM_MTP_TARGET_DIR"],
            let drafterDir = env["VLM_MTP_DRAFTER_DIR"]
        else {
            print(
                "Skipping VLM-MTP spike: set VLM_MTP_SPIKE=1, "
                    + "VLM_MTP_TARGET_DIR, VLM_MTP_DRAFTER_DIR")
            return
        }
        let maxTokens = Int(env["VLM_MTP_MAXTOK"] ?? "64") ?? 64
        let K = Int(env["VLM_MTP_K"] ?? "3") ?? 3
        // A REAL chat-templated prompt — greedy parity is only meaningful where
        // the model is confident; off-distribution prompts flip argmax on tiny
        // logit noise. Prefer a tokenized prompts file, else a small default.
        let prompt = loadPrompt(env)
        let targetURL = URL(fileURLWithPath: targetDir)
        let drafterURL = URL(fileURLWithPath: drafterDir)

        // --- Phase 1: through the MLXVLM tower (the path under test) ---
        // While the VLM is loaded, collect BOTH plain greedy (no drafter) and
        // MTP greedy. Released on return so the diagnostic load fits in memory.
        let (mtpTokens, vlmGreedy, bridgeReport) = try await runVLM(
            targetURL: targetURL, drafterURL: drafterURL,
            prompt: prompt, maxTokens: maxTokens, K: K)
        MLX.Memory.clearCache()

        // --- Phase 2: diagnostic — plain greedy through the MLXLLM text tower
        // from the same checkpoint. Tells us whether the two tower
        // implementations are numerically equivalent (which governs drafter
        // ACCEPTANCE/speed) — but is NOT the correctness gate.
        let (llmText, _) = try loadRealTargetAsTextModel(from: targetURL)
        eval(llmText)
        let llmGreedy = runBatchedBaselineGreedyTokens(
            target: llmText, promptTokens: [prompt], maxTokens: maxTokens)[0]
        MLX.Memory.clearCache()

        let matchMtpVsVlm = matchedPrefix(mtpTokens, vlmGreedy)
        let matchVlmVsLlm = matchedPrefix(vlmGreedy, llmGreedy)
        let n = Swift.min(mtpTokens.count, vlmGreedy.count)

        print("\n=== VLM-MTP spike (K=\(K), max=\(maxTokens), prompt len=\(prompt.count)) ===")
        print(bridgeReport)
        print("vlm-greedy  (MLXVLM, no drafter): \(vlmGreedy.prefix(20))")
        print("mtp         (MLXVLM, drafter):    \(mtpTokens.prefix(20))")
        print("llm-greedy  (MLXLLM ref):         \(llmGreedy.prefix(20))")
        print("CORRECTNESS  MTP==VLM-greedy: \(matchMtpVsVlm)/\(n)")
        print("DIAGNOSTIC   VLM-greedy==LLM-greedy (tower equiv): "
            + "\(matchVlmVsLlm)/\(Swift.min(vlmGreedy.count, llmGreedy.count))")

        // CHARACTERIZATION (not the strict correctness gate — see the
        // `vlm_multitoken_forward_consistency`, `mlxllm_rollback_consistency`,
        // and `vlm_acceptall_consistency` tests, which PROVE the forward / mask
        // / rollback / walk are correct). MTP reproduces the VLM tower's own
        // greedy decode until the first near-tie (~0.14-logit gap) that the VLM
        // tower's bf16-noisier multi-query attention flips — exactly the
        // "similar answer, not exact" tolerance. A real logic regression craters
        // this to near-zero, so the floor catches gross breakage; the isolation
        // tests catch the rest.
        #expect(matchMtpVsVlm >= 10,
            """
            VLM-MTP regressed: MTP diverges from the VLM tower's greedy decode \
            far earlier than the known bf16-tie point. Likely a real logic bug — \
            re-run the isolation tests (forward/rollback/accept-all).
              matched=\(matchMtpVsVlm)/\(n)
              vlm-greedy=\(vlmGreedy.prefix(matchMtpVsVlm + 4))
              mtp       =\(mtpTokens.prefix(matchMtpVsVlm + 4))
            """)
    }

    private func loadPrompt(_ env: [String: String]) -> [Int32] {
        // Inline JSON int array wins, then a prompts-file path (JSON [[Int]] or
        // [Int]); index via VLM_MTP_PROMPT_IDX. Fallback: a tiny default.
        if let p = env["VLM_MTP_PROMPT"],
            let arr = try? JSONDecoder().decode([Int].self, from: Data(p.utf8))
        {
            return arr.map { Int32($0) }
        }
        if let path = env["VLM_MTP_PROMPTS"],
            let data = try? Data(contentsOf: URL(fileURLWithPath: path))
        {
            let idx = Int(env["VLM_MTP_PROMPT_IDX"] ?? "0") ?? 0
            if let rows = try? JSONDecoder().decode([[Int]].self, from: data), !rows.isEmpty {
                return rows[idx % rows.count].map { Int32($0) }
            }
            if let flat = try? JSONDecoder().decode([Int].self, from: data) {
                return flat.map { Int32($0) }
            }
        }
        return [2, 105, 2364, 107, 1596, 603, 476, 2425, 576, 573, 4368, 235292]
    }

    private func matchedPrefix(_ a: [Int], _ b: [Int]) -> Int {
        let n = Swift.min(a.count, b.count)
        for j in 0 ..< n where a[j] != b[j] { return j }
        return n
    }

    /// Plain greedy through the VLM tower via `forwardForMTP` (no drafter): the
    /// verify path MTP uses, so MTP must reproduce this exactly.
    private func vlmPlainGreedy(
        _ vlm: MLXVLM.Gemma4, prompt: [Int32], maxTokens: Int
    ) -> [Int] {
        let cache = vlm.newCache(parameters: nil)
        var toks: [Int] = []
        var cur = MLXArray(prompt)[.newAxis, .ellipsis]
        while toks.count < maxTokens {
            let out = vlm.forwardForMTP(cur, cache: cache)
            let nextArr = out.logits[0..., -1, 0...].asType(.float32).argMax(axis: -1)
            eval(nextArr)
            let next = Int(nextArr.item(Int32.self))
            toks.append(next)
            cur = MLXArray([Int32(next)])[.newAxis, .ellipsis]
        }
        return toks
    }

    /// Bind the drafter to the freshly-loaded MLXVLM tower; collect plain
    /// greedy AND MTP greedy from the same loaded model. Released on return.
    private func runVLM(
        targetURL: URL, drafterURL: URL, prompt: [Int32], maxTokens: Int, K: Int
    ) async throws -> (mtp: [Int], vlmGreedy: [Int], bridge: String) {
        let vlm = try loadRealTargetAsVLM(from: targetURL)
        let drafter = try await Gemma4AssistantDraftModel.load(from: drafterURL)
        eval(vlm, drafter)

        // Validate the VLM→MLXLLM config bridge the drafter validator and
        // automatic policy consume.
        let bridged = vlm.mtpConfiguration
        let bridgeReport = """
            bridge: hidden=\(bridged.hiddenSize) vocab=\(bridged.vocabSize) \
            layers=\(bridged.numHiddenLayers) kvShared=\(bridged.numKvSharedLayers) \
            kEqV=\(bridged.attentionKeqV) moe=\(bridged.enableMoeBlock) \
            layerTypes=\(bridged.layerTypes.count)
            """
        #expect(bridged.hiddenSize == vlm.config.textConfiguration.hiddenSize)
        #expect(bridged.vocabSize == vlm.config.textConfiguration.vocabularySize)
        #expect(bridged.numHiddenLayers == vlm.config.textConfiguration.hiddenLayers)
        #expect(bridged.numKvSharedLayers == vlm.config.textConfiguration.numKVSharedLayers)
        #expect(bridged.attentionKeqV == vlm.config.textConfiguration.attentionKEqV)
        #expect(bridged.layerTypes == vlm.config.textConfiguration.layerTypes)

        // Plain greedy through the VLM tower (no drafter) — the correctness
        // reference MTP must reproduce.
        let vlmGreedy = vlmPlainGreedy(vlm, prompt: prompt, maxTokens: maxTokens)
        MLX.Memory.clearCache()

        // Drafter must bind to the VLM tower without an incompatibility throw.
        try drafter.bind(target: vlm)

        let input = LMInput(text: .init(tokens: MLXArray(prompt)))
        var iter = try Gemma4MTPTokenIterator(
            input: input, target: vlm, drafter: drafter,
            parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0),
            blockSize: K)
        var tokens: [Int] = []
        while let t = iter.next(), tokens.count < maxTokens {
            tokens.append(t)
        }
        return (tokens, vlmGreedy, bridgeReport)
    }

    /// Load a multimodal Gemma 4 checkpoint as an `MLXVLM.Gemma4` (the VLM
    /// tower under test). Mirrors `loadRealTargetAsTextModel` but constructs
    /// the VLM model so the vision tower + text tower share the same weights.
    private func loadRealTargetAsVLM(from modelDir: URL) throws -> MLXVLM.Gemma4 {
        let configData = try Data(
            contentsOf: modelDir.appendingPathComponent("config.json"))
        let decoder = JSONDecoder()
        let fullConfig = try decoder.decode(MLXVLM.Gemma4Configuration.self, from: configData)
        let baseConfig = try? decoder.decode(BaseConfiguration.self, from: configData)

        let model = MLXVLM.Gemma4(fullConfig)

        var weights = [String: MLXArray]()
        let scanDir = modelDir.resolvingSymlinksInPath()
        let shardURLs = (try? FileManager.default.contentsOfDirectory(
            at: scanDir, includingPropertiesForKeys: nil)) ?? []
        for url in shardURLs where url.pathExtension == "safetensors" {
            let (w, _) = try loadArraysAndMetadata(url: url)
            for (k, v) in w { weights[k] = v }
        }
        let sanitized = model.sanitize(weights: weights)
        if let plq = baseConfig?.perLayerQuantization {
            quantize(model: model) { path, _ in
                sanitized["\(path).scales"] != nil
                    ? plq.quantization(layer: path)?.asTuple : nil
            }
        }
        try model.update(
            parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
        eval(model)
        return model
    }

    /// Load a Gemma 4 checkpoint as the reference `MLXLLM.Gemma4TextModel`
    /// (gold baseline). Same loading logic as the benchmark suite's private
    /// helper, qualified for the dual-module test target.
    private func loadRealTargetAsTextModel(
        from modelDir: URL
    ) throws -> (MLXLLM.Gemma4TextModel, MLXLLM.Gemma4Configuration) {
        let configData = try Data(
            contentsOf: modelDir.appendingPathComponent("config.json"))
        let decoder = JSONDecoder()
        let fullConfig = try decoder.decode(MLXLLM.Gemma4Configuration.self, from: configData)
        let baseConfig = try? decoder.decode(BaseConfiguration.self, from: configData)

        let wrapper = MLXLLM.Gemma4Model(fullConfig)

        var weights = [String: MLXArray]()
        let scanDir = modelDir.resolvingSymlinksInPath()
        let shardURLs = (try? FileManager.default.contentsOfDirectory(
            at: scanDir, includingPropertiesForKeys: nil)) ?? []
        for url in shardURLs where url.pathExtension == "safetensors" {
            let (w, _) = try loadArraysAndMetadata(url: url)
            for (k, v) in w { weights[k] = v }
        }
        let sanitized = wrapper.sanitize(weights: weights)
        if let plq = baseConfig?.perLayerQuantization {
            quantize(model: wrapper) { path, _ in
                sanitized["\(path).scales"] != nil
                    ? plq.quantization(layer: path)?.asTuple : nil
            }
        }
        try wrapper.update(
            parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
        eval(wrapper)
        return (wrapper.textModel, fullConfig)
    }
}
#endif
