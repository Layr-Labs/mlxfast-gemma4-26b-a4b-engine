# Gemma 4 DFlash Depth Sweep

Date: 2026-08-30

## Workload

- Apple M5 Max.
- Exact transformed Gemma 4 26B A4B target, 30 layers, hidden size 2,816.
- Organizer-pinned `z-lab/gemma-4-26B-A4B-it-DFlash` revision
  `77d4202772dfe50b2396ec7bac9cfffc7b9e7057`.
- DFlash tree receipt: 859,385,423 bytes, 2 files,
  SHA-256 `9b3a065e1b12a1bb87d064e2fd3ed7d7078a6d0fbb6e000506eb9a804d635b9d`.
- One 1,024-token public prompt and 128 post-seed decode tokens.
- Serial and DFlash phases ran in one resident worker and were interleaved.
- Prefill and decode were timed separately from the parent request boundary.
- Production service transitions used the canonical exclusive GPU guard.

## Ranked depth sweep

The initial sweep used three samples for serial and depths 1, 2, 4, 8, and
15. Missing depths used one screening sample. A depth is fidelity-admissible
only when the 128-token chain has at most 12 positional mismatches versus the
public serial chain (the track's 10% budget).

| Depth | Samples | Decode tok/s | Mismatches | Rounds | Drafted | Accepted | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| serial | 3 | 115.15 | 0 | 128 | 0 | 0 | winner |
| 1 | 3 | 70.36 | 115 | 74 | 74 | 54 | reject |
| 2 | 3 | 83.17 | 115 | 58 | 116 | 70 | reject |
| 3 | 1 | 93.73 | 115 | 45 | 134 | 83 | reject |
| 4 | 3 | 109.76 | 115 | 36 | 141 | 92 | reject |
| 5 | 1 | 100.12 | 115 | 33 | 164 | 95 | reject |
| 6 | 1 | 96.50 | 115 | 31 | 179 | 97 | reject |
| 7 | 1 | 101.19 | 115 | 32 | 223 | 96 | reject |
| 8 | 3 | 84.86 | 115 | 30 | 238 | 98 | reject |
| 9 | 1 | 59.06 | 115 | 29 | 248 | 99 | reject |
| 10 | 1 | 91.35 | 115 | 27 | 270 | 101 | reject |
| 11 | 1 | 82.20 | 0 | 30 | 324 | 98 | admit |
| 12 | 1 | 78.60 | 0 | 30 | 352 | 98 | admit |
| 13 | 1 | 77.56 | 0 | 30 | 379 | 98 | admit |
| 14 | 1 | 73.95 | 0 | 30 | 406 | 98 | admit |
| 15 | 3 | 71.54 | 0 | 30 | 433 | 98 | admit |

Depth 11 was then confirmed in a fresh interleaved mean-of-three bracket:

| Route | Prefill mean s | Prefill tok/s | Decode mean s | Decode tok/s | Mismatches |
|---|---:|---:|---:|---:|---:|
| serial | 0.198283 | 5,167.15 | 1.113315 | 114.97 | 0 |
| DFlash depth 11 | 0.251363 | 4,073.80 | 1.551507 | 82.50 | 0 |

Serial remained 1.394x faster in decode wall time. Depth 11 is retained only
because DFlash was explicitly selected; it is the fastest tested
fidelity-admissible DFlash depth, not a performance win over serial.

## Final default-route receipt

`spec-decoder-head.manifest.json` selects `dflash`. With the benchmarker's
empty DFlash block, the final rebuilt worker resolved effective depth 11 and
completed the same 1K/128 prompt with zero mismatches:

```text
prefill_tps=3955.18
decode_tps=82.56
rounds=30 drafted=324 accepted=98
cache_memory=0
```

