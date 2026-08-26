# BenchCBv2RealModel report

| | |
|---|---|
| Model | ~/.cache/huggingface/hub/models--mlx-community--gpt-oss-20b-MXFP4-Q8/snapshots/773a7da77e569019bb0fd17a554b263738d669a3/ |
| Chip | Apple M4 Max |
| RAM | 128 GB |
| OS | Version 26.5 (Build 25F71) |
| Host at start | load avg (1m) 5.6 / 16 cores; no darkbloom process |
| Date | 2026-07-10T05:35:00Z |

model class: GPTOSSModel; layers: 24
vocabSize=201088

## Correctness (v2 contiguous, greedy)

prompt tokens: target=79 short=74 long=1600 eos=[199999, 200002, 200012]
[b1-sanity] finish=stop tokens=195 loop=false
[b1-sanity] text: <|channel|>analysis<|message|>We need to explain why the sky is blue in two sentences. So we need to produce a concise explanation. The sky is blue because of Rayleigh scattering of sunlight by air molecules, which scatters shorter wavelengths (blue) more strongly than longer wavelengths (red). So we can say: The sky appears blue because sunlight is scattered by the Earth's atmosphere, and shorter blue wavelengths are scattered more efficiently than longer red wavelengths. That is two sentences. Ensure it's exactly two sentences. Let's craft: "The sky looks blue because sunlight is scattered by the Earth's atmosphere, and the shorter blue wavelengths are scattered more efficiently than the longer red wavelengths. This preferential scattering makes the sky appear blue to our eyes." That's two sentences. Good.<|end|><|start|>assistant<|channel|>final<|message|>The sky looks blue because sunlight is scattered by the Earth's atmosphere, and the shorter blue wavelengths are scattered more efficiently than the longer red wavelengths. This preferential scattering makes the sky appear blue to our eyes.
[invariance] solo tokens=64 finish=length
[invariance] solo-repeat divergence=nil
[invariance] solo text: <|channel|>analysis<|message|>We need to explain why the sky is blue in two sentences. So we need to produce a concise explanation. The sky is blue because of Rayleigh scattering of sunlight by air molecules, which scatters shorter wavelengths (blue) more strongly than longer wavelengths (red). So we can say: The sky appears
[invariance] burst target tokens=64 divergence=nil
[invariance] mid-join: neighbors had (8, 5) tokens at join; target tokens=64 divergence=nil
[invariance] neighbor-invariance (burst vs mid-join at B=3): divergence=nil
[chunked-prefill] chunked=64 tok unchunked=64 tok divergence=nil
[chunked-prefill] chunked text: <|channel|>analysis<|message|>The user asks: "Summarize the following passage in one sentence:" and then repeats the same sentence many times. The passage is basically: "The quick brown fox jumps over the lazy dog while the seasoned cartographer annotates ancient maps with meticulous margin notes about tides, trade winds, and the slow
[compiled] solo divergence=nil compiledSteps=64 fallbacks=[:] warmup: b1=0.04s b2=0.03s b4=0.04s
[compiled] text: <|channel|>analysis<|message|>We need to explain why the sky is blue in two sentences. So we need to produce a concise explanation. The sky is blue because of Rayleigh scattering of sunlight by air molecules, which scatters shorter wavelengths (blue) more strongly than longer wavelengths (red). So we can say: The sky appears
[compiled] burst-vs-solo divergence=nil compiledSteps=99 fallbacks=[:]

- **b1-sanity**: PASS — finish=stop completion=195 repetitionLoop=false
- **invariance**: PASS — solo=64 tok; solo-repeat deterministic; neighbor-invariant at B=3 (burst == mid-join); burst identical to solo; mid-join identical to solo
- **chunked-prefill**: PASS — token-exact over 64 tokens (prompt=1600)
- **compiled-parity**: PASS — compiledSteps=64; token-exact vs eager over 64 tokens
- **compiled-invariance**: PASS — compiled burst identical to compiled solo (compiledSteps=99)

## Performance (maxTokens 128)

| engine | B | prompts (tok) | decode TPS/req | agg TPS | TTFT p50 (ms) | ITL p50 (ms) |
|---|---|---|---|---|---|---|
| v2 | 1 | 500 | 101.8 | 73.7 | 489 | 9.8 |
    [mem after v2 B=1] gpuActive=11.25 GiB gpuPeak=12.44 GiB
| v2 | 2 | 100/1500 | 59.8 | 74.5 | 1176 | 13.4 |
    [mem after v2 B=2] gpuActive=11.25 GiB gpuPeak=12.33 GiB
| v2 | 4 | 100/500/1500/500 | 37.3 | 93.0 | 1720 | 21.1 |
    [mem after v2 B=4] gpuActive=11.25 GiB gpuPeak=13.45 GiB
| v2 | 8 | 500/500/500/500/500/500/500/500 | 21.9 | 108.5 | 3432 | 38.8 |
    [mem after v2 B=8] gpuActive=11.25 GiB gpuPeak=13.81 GiB
    [v2-compiled B=1] warmup 0.10s (b1=0.02s)
    [v2-compiled B=1] compiledSteps=10 rebinds=3 scratchResets=0 fallbacks=[:]
| v2-compiled | 1 | 500 | 96.7 | 68.1 | 565 | 10.3 |
    [mem after v2-compiled B=1] gpuActive=11.25 GiB gpuPeak=12.43 GiB
| v2-compiled | 2 | 100/1500 | 56.0 | 69.2 | 1287 | 14.5 |
    [mem after v2-compiled B=2] gpuActive=11.25 GiB gpuPeak=12.66 GiB
| v2-compiled | 4 | 100/500/1500/500 | 35.3 | 88.5 | 1799 | 22.5 |
    [mem after v2-compiled B=4] gpuActive=11.25 GiB gpuPeak=13.45 GiB
| v2-compiled | 8 | 500/500/500/500/500/500/500/500 | 21.6 | 108.8 | 3325 | 37.7 |
    [mem after v2-compiled B=8] gpuActive=11.25 GiB gpuPeak=14.45 GiB
| v2-paged | 1 | 500 | 88.5 | 63.9 | 570 | 11.3 |
    [mem after v2-paged B=1] gpuActive=11.25 GiB gpuPeak=28.21 GiB
| v2-paged | 2 | 100/1500 | 59.4 | 72.3 | 1266 | 13.6 |
    [mem after v2-paged B=2] gpuActive=11.25 GiB gpuPeak=28.18 GiB
| v2-paged | 4 | 100/500/1500/500 | 38.5 | 93.7 | 1789 | 20.2 |
    [mem after v2-paged B=4] gpuActive=11.25 GiB gpuPeak=28.65 GiB
| v2-paged | 8 | 500/500/500/500/500/500/500/500 | 22.2 | 110.8 | 3330 | 38.6 |
    [mem after v2-paged B=8] gpuActive=11.25 GiB gpuPeak=29.44 GiB

Per-request detail:
  v2 B=1:
    req 0: prompt=500 tokens=128 ttft=489ms decodeTPS=101.8 finish=length
  v2 B=2:
    req 0: prompt=100 tokens=128 ttft=635ms decodeTPS=45.7 finish=length
    req 1: prompt=1500 tokens=128 ttft=1718ms decodeTPS=73.9 finish=length
  v2 B=4:
    req 0: prompt=100 tokens=128 ttft=1720ms decodeTPS=33.8 finish=length
    req 1: prompt=500 tokens=128 ttft=1720ms decodeTPS=33.8 finish=length
    req 2: prompt=1500 tokens=128 ttft=2839ms decodeTPS=47.7 finish=length
    req 3: prompt=500 tokens=128 ttft=1720ms decodeTPS=33.8 finish=length
  v2 B=8:
    req 0: prompt=500 tokens=128 ttft=2321ms decodeTPS=17.9 finish=length
    req 1: prompt=500 tokens=128 ttft=2321ms decodeTPS=17.9 finish=length
    req 2: prompt=500 tokens=128 ttft=2321ms decodeTPS=17.9 finish=length
    req 3: prompt=500 tokens=128 ttft=2321ms decodeTPS=17.9 finish=length
    req 4: prompt=500 tokens=128 ttft=4543ms decodeTPS=25.9 finish=length
    req 5: prompt=500 tokens=128 ttft=4543ms decodeTPS=25.9 finish=length
    req 6: prompt=500 tokens=128 ttft=4543ms decodeTPS=25.9 finish=length
    req 7: prompt=500 tokens=128 ttft=4543ms decodeTPS=25.9 finish=length
    [v2-compiled B=1] warmup 0.09s (b1=0.03s)
    [v2-compiled B=1] compiledSteps=130 rebinds=3 scratchResets=0 fallbacks=[:]
  v2-compiled B=1:
    req 0: prompt=500 tokens=128 ttft=565ms decodeTPS=96.7 finish=length
    [v2-compiled B=2] warmup 0.12s (b1=0.02s b2=0.03s)
    [v2-compiled B=2] compiledSteps=132 rebinds=8 scratchResets=0 fallbacks=[:]
  v2-compiled B=2:
    req 0: prompt=100 tokens=128 ttft=714ms decodeTPS=42.9 finish=length
    req 1: prompt=1500 tokens=128 ttft=1860ms decodeTPS=69.0 finish=length
    [v2-compiled B=4] warmup 0.16s (b1=0.02s b2=0.03s b4=0.04s)
    [v2-compiled B=4] compiledSteps=132 rebinds=16 scratchResets=0 fallbacks=[:]
  v2-compiled B=4:
    req 0: prompt=100 tokens=128 ttft=1799ms decodeTPS=32.1 finish=length
    req 1: prompt=500 tokens=128 ttft=1799ms decodeTPS=32.1 finish=length
    req 2: prompt=1500 tokens=128 ttft=2946ms decodeTPS=44.8 finish=length
    req 3: prompt=500 tokens=128 ttft=1799ms decodeTPS=32.1 finish=length
    [v2-compiled B=8] warmup 0.16s (b1=0.02s b2=0.03s b4=0.04s)
    [v2-compiled B=8] compiledSteps=3 rebinds=12 scratchResets=0 fallbacks=["batch_8_exceeds_ladder": 127]
  v2-compiled B=8:
    req 0: prompt=500 tokens=128 ttft=2249ms decodeTPS=17.8 finish=length
    req 1: prompt=500 tokens=128 ttft=2249ms decodeTPS=17.8 finish=length
    req 2: prompt=500 tokens=128 ttft=4400ms decodeTPS=25.3 finish=length
    req 3: prompt=500 tokens=128 ttft=2249ms decodeTPS=17.8 finish=length
    req 4: prompt=500 tokens=128 ttft=4400ms decodeTPS=25.3 finish=length
    req 5: prompt=500 tokens=128 ttft=2249ms decodeTPS=17.8 finish=length
    req 6: prompt=500 tokens=128 ttft=4400ms decodeTPS=25.4 finish=length
    req 7: prompt=500 tokens=128 ttft=4400ms decodeTPS=25.3 finish=length
  v2-paged B=1:
    req 0: prompt=500 tokens=128 ttft=570ms decodeTPS=88.5 finish=length
  v2-paged B=2:
    req 0: prompt=100 tokens=128 ttft=722ms decodeTPS=45.4 finish=length
    req 1: prompt=1500 tokens=128 ttft=1809ms decodeTPS=73.3 finish=length
  v2-paged B=4:
    req 0: prompt=100 tokens=128 ttft=1789ms decodeTPS=34.8 finish=length
    req 1: prompt=500 tokens=128 ttft=1789ms decodeTPS=34.8 finish=length
    req 2: prompt=1500 tokens=128 ttft=2896ms decodeTPS=49.4 finish=length
    req 3: prompt=500 tokens=128 ttft=1789ms decodeTPS=34.8 finish=length
  v2-paged B=8:
    req 0: prompt=500 tokens=128 ttft=4385ms decodeTPS=26.2 finish=length
    req 1: prompt=500 tokens=128 ttft=4385ms decodeTPS=26.2 finish=length
    req 2: prompt=500 tokens=128 ttft=2275ms decodeTPS=18.3 finish=length
    req 3: prompt=500 tokens=128 ttft=2275ms decodeTPS=18.3 finish=length
    req 4: prompt=500 tokens=128 ttft=4385ms decodeTPS=26.2 finish=length
    req 5: prompt=500 tokens=128 ttft=2275ms decodeTPS=18.3 finish=length
    req 6: prompt=500 tokens=128 ttft=2275ms decodeTPS=18.3 finish=length
    req 7: prompt=500 tokens=128 ttft=4385ms decodeTPS=26.2 finish=length
