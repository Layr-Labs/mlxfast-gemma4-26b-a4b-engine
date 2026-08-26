# BenchCBv2RealModel report

| | |
|---|---|
| Model | ~/.cache/huggingface/hub/models--mlx-community--gemma-4-26B-A4B-it-qat-4bit/snapshots/0e3cbab38ce568cf6e23543010d08d03b731910c/ |
| Chip | Apple M4 Max |
| RAM | 128 GB |
| OS | Version 26.5 (Build 25F71) |
| Host at start | load avg (1m) 7.2 / 16 cores; no darkbloom process |
| Date | 2026-07-10T05:36:17Z |

model class: Gemma4Model; layers: 30
vocabSize=262144

## Correctness (v2 contiguous, greedy)

prompt tokens: target=24 short=19 long=1543 eos=[1, 50, 106]
[b1-sanity] finish=stop tokens=46 loop=false
[b1-sanity] text: The sky appears blue because sunlight is scattered by the gases and particles in Earth's atmosphere. Blue light travels in shorter, smaller waves and is scattered more strongly than other colors, making the sky look blue from the ground.
[invariance] solo tokens=46 finish=stop
[invariance] solo-repeat divergence=nil
[invariance] solo text: The sky appears blue because sunlight is scattered by the gases and particles in Earth's atmosphere. Blue light travels in shorter, smaller waves and is scattered more strongly than other colors, making the sky look blue from the ground.
[invariance] burst target tokens=46 divergence=nil
[invariance] mid-join: neighbors had (8, 5) tokens at join; target tokens=46 divergence=nil
[invariance] neighbor-invariance (burst vs mid-join at B=3): divergence=nil
[chunked-prefill] chunked=32 tok unchunked=32 tok divergence=nil
[chunked-prefill] chunked text: A quick brown fox jumps over a lazy dog while a seasoned cartographer meticulously annotates ancient maps with details regarding tides, trade winds, and continental drift.
[compiled] solo divergence=nil compiledSteps=46 fallbacks=[:] warmup: b1=0.04s b2=0.03s b4=0.04s
[compiled] text: The sky appears blue because sunlight is scattered by the gases and particles in Earth's atmosphere. Blue light travels in shorter, smaller waves and is scattered more strongly than other colors, making the sky look blue from the ground.
[compiled] burst-vs-solo divergence=nil compiledSteps=46 fallbacks=[:]

- **b1-sanity**: PASS — finish=stop completion=46 repetitionLoop=false
- **invariance**: PASS — solo=46 tok; solo-repeat deterministic; neighbor-invariant at B=3 (burst == mid-join); burst identical to solo; mid-join identical to solo
- **chunked-prefill**: PASS — token-exact over 32 tokens (prompt=1543)
- **compiled-parity**: PASS — compiledSteps=46; token-exact vs eager over 46 tokens
- **compiled-invariance**: PASS — compiled burst identical to compiled solo (compiledSteps=46)

## Performance (maxTokens 128)

| engine | B | prompts (tok) | decode TPS/req | agg TPS | TTFT p50 (ms) | ITL p50 (ms) |
|---|---|---|---|---|---|---|
| v2 | 1 | 500 | 101.8 | 76.0 | 437 | 9.7 |
    [mem after v2 B=1] gpuActive=13.48 GiB gpuPeak=14.79 GiB
| v2 | 2 | 100/1500 | 59.6 | 77.9 | 1045 | 13.9 |
    [mem after v2 B=2] gpuActive=13.48 GiB gpuPeak=15.10 GiB
| v2 | 4 | 100/500/1500/500 | 38.0 | 97.7 | 1569 | 21.1 |
    [mem after v2 B=4] gpuActive=13.48 GiB gpuPeak=16.88 GiB
| v2 | 8 | 500/500/500/500/500/500/500/500 | 24.7 | 122.0 | 3079 | 34.7 |
    [mem after v2 B=8] gpuActive=13.48 GiB gpuPeak=18.53 GiB
    [v2-compiled B=1] warmup 0.09s (b1=0.02s)
    [v2-compiled B=1] compiledSteps=10 rebinds=3 scratchResets=0 fallbacks=[:]
| v2-compiled | 1 | 500 | 100.9 | 73.5 | 483 | 9.9 |
    [mem after v2-compiled B=1] gpuActive=13.48 GiB gpuPeak=14.79 GiB
| v2-compiled | 2 | 100/1500 | 61.5 | 77.7 | 1103 | 13.2 |
    [mem after v2-compiled B=2] gpuActive=13.48 GiB gpuPeak=15.35 GiB
| v2-compiled | 4 | 100/500/1500/500 | 36.1 | 92.6 | 1645 | 22.2 |
    [mem after v2-compiled B=4] gpuActive=13.48 GiB gpuPeak=16.88 GiB
| v2-compiled | 8 | 500/500/500/500/500/500/500/500 | 23.2 | 116.3 | 3153 | 37.0 |
    [mem after v2-compiled B=8] gpuActive=13.48 GiB gpuPeak=18.82 GiB
| v2-paged | 1 | 500 | 99.5 | 72.1 | 500 | 10.0 |
    [mem after v2-paged B=1] gpuActive=13.48 GiB gpuPeak=30.51 GiB
| v2-paged | 2 | 100/1500 | 62.2 | 80.9 | 893 | 12.3 |
    [mem after v2-paged B=2] gpuActive=13.48 GiB gpuPeak=30.51 GiB
| v2-paged | 4 | 100/500/1500/500 | 39.0 | 97.8 | 1632 | 20.2 |
    [mem after v2-paged B=4] gpuActive=13.48 GiB gpuPeak=31.15 GiB
| v2-paged | 8 | 500/500/500/500/500/500/500/500 | 23.8 | 115.6 | 3340 | 35.9 |
    [mem after v2-paged B=8] gpuActive=13.48 GiB gpuPeak=32.49 GiB

Per-request detail:
  v2 B=1:
    req 0: prompt=500 tokens=128 ttft=437ms decodeTPS=101.8 finish=length
  v2 B=2:
    req 0: prompt=100 tokens=128 ttft=567ms decodeTPS=47.1 finish=length
    req 1: prompt=1500 tokens=128 ttft=1523ms decodeTPS=72.0 finish=length
  v2 B=4:
    req 0: prompt=100 tokens=128 ttft=1569ms decodeTPS=34.8 finish=length
    req 1: prompt=500 tokens=128 ttft=1569ms decodeTPS=34.8 finish=length
    req 2: prompt=1500 tokens=128 ttft=2561ms decodeTPS=47.4 finish=length
    req 3: prompt=500 tokens=128 ttft=1569ms decodeTPS=34.8 finish=length
  v2 B=8:
    req 0: prompt=500 tokens=128 ttft=2133ms decodeTPS=20.3 finish=length
    req 1: prompt=500 tokens=128 ttft=2133ms decodeTPS=20.3 finish=length
    req 2: prompt=500 tokens=128 ttft=2133ms decodeTPS=20.3 finish=length
    req 3: prompt=500 tokens=128 ttft=2133ms decodeTPS=20.3 finish=length
    req 4: prompt=500 tokens=128 ttft=4026ms decodeTPS=29.1 finish=length
    req 5: prompt=500 tokens=128 ttft=4026ms decodeTPS=29.1 finish=length
    req 6: prompt=500 tokens=128 ttft=4026ms decodeTPS=29.1 finish=length
    req 7: prompt=500 tokens=128 ttft=4026ms decodeTPS=29.1 finish=length
    [v2-compiled B=1] warmup 0.09s (b1=0.02s)
    [v2-compiled B=1] compiledSteps=130 rebinds=3 scratchResets=0 fallbacks=[:]
  v2-compiled B=1:
    req 0: prompt=500 tokens=128 ttft=483ms decodeTPS=100.9 finish=length
    [v2-compiled B=2] warmup 0.12s (b1=0.02s b2=0.03s)
    [v2-compiled B=2] compiledSteps=132 rebinds=8 scratchResets=0 fallbacks=[:]
  v2-compiled B=2:
    req 0: prompt=100 tokens=128 ttft=586ms decodeTPS=47.2 finish=length
    req 1: prompt=1500 tokens=128 ttft=1619ms decodeTPS=75.7 finish=length
    [v2-compiled B=4] warmup 0.16s (b1=0.02s b2=0.03s b4=0.04s)
    [v2-compiled B=4] compiledSteps=132 rebinds=16 scratchResets=0 fallbacks=[:]
  v2-compiled B=4:
    req 0: prompt=100 tokens=128 ttft=1645ms decodeTPS=33.0 finish=length
    req 1: prompt=500 tokens=128 ttft=1645ms decodeTPS=33.0 finish=length
    req 2: prompt=1500 tokens=128 ttft=2725ms decodeTPS=45.3 finish=length
    req 3: prompt=500 tokens=128 ttft=1645ms decodeTPS=33.0 finish=length
    [v2-compiled B=8] warmup 0.16s (b1=0.02s b2=0.03s b4=0.04s)
    [v2-compiled B=8] compiledSteps=3 rebinds=12 scratchResets=0 fallbacks=["batch_8_exceeds_ladder": 127]
  v2-compiled B=8:
    req 0: prompt=500 tokens=128 ttft=2157ms decodeTPS=19.1 finish=length
    req 1: prompt=500 tokens=128 ttft=2157ms decodeTPS=19.1 finish=length
    req 2: prompt=500 tokens=128 ttft=4148ms decodeTPS=27.3 finish=length
    req 3: prompt=500 tokens=128 ttft=4148ms decodeTPS=27.3 finish=length
    req 4: prompt=500 tokens=128 ttft=2157ms decodeTPS=19.1 finish=length
    req 5: prompt=500 tokens=128 ttft=2157ms decodeTPS=19.1 finish=length
    req 6: prompt=500 tokens=128 ttft=4148ms decodeTPS=27.3 finish=length
    req 7: prompt=500 tokens=128 ttft=4148ms decodeTPS=27.3 finish=length
  v2-paged B=1:
    req 0: prompt=500 tokens=128 ttft=500ms decodeTPS=99.5 finish=length
  v2-paged B=2:
    req 0: prompt=100 tokens=128 ttft=184ms decodeTPS=43.1 finish=length
    req 1: prompt=1500 tokens=128 ttft=1603ms decodeTPS=81.4 finish=length
  v2-paged B=4:
    req 0: prompt=100 tokens=128 ttft=1632ms decodeTPS=35.5 finish=length
    req 1: prompt=500 tokens=128 ttft=1632ms decodeTPS=35.5 finish=length
    req 2: prompt=1500 tokens=128 ttft=2673ms decodeTPS=49.5 finish=length
    req 3: prompt=500 tokens=128 ttft=1632ms decodeTPS=35.5 finish=length
  v2-paged B=8:
    req 0: prompt=500 tokens=128 ttft=2348ms decodeTPS=19.5 finish=length
    req 1: prompt=500 tokens=128 ttft=2348ms decodeTPS=19.5 finish=length
    req 2: prompt=500 tokens=128 ttft=2348ms decodeTPS=19.5 finish=length
    req 3: prompt=500 tokens=128 ttft=2348ms decodeTPS=19.5 finish=length
    req 4: prompt=500 tokens=128 ttft=4333ms decodeTPS=28.1 finish=length
    req 5: prompt=500 tokens=128 ttft=4333ms decodeTPS=28.1 finish=length
    req 6: prompt=500 tokens=128 ttft=4333ms decodeTPS=28.1 finish=length
    req 7: prompt=500 tokens=128 ttft=4333ms decodeTPS=28.1 finish=length
