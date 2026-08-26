# CI coverage: what the hosted pipeline gates, and what it cannot

`.github/workflows/ci.yml` is the only automated gate in this repository. This
document says what that pipeline proves. More importantly, it says what the
pipeline does not prove, because this package's most interesting tests need
hardware that GitHub does not rent.

This document exists to enforce one rule: **a check that CI cannot run is
named here, never silently skipped.** `tools/ci-box-only-inventory.sh --check`
runs in CI. It fails the build if the inventory below drifts from the test
sources.

## Why the split exists

The engine targets Apple Silicon through MLX. Two resources decide whether a
test can run on a hosted runner:

- **A Metal GPU with the kernels built for it.** GitHub-hosted macOS runners
  are virtualised. `./setup.sh` never runs there, so nothing ever builds
  `mlx.metallib` (`tools/build-mlx-metallib.sh` needs the Metal toolchain
  component). Nothing in this pipeline relies on MLX executing on a GPU, and
  no timing claim may ever be made from it.
- **The reference checkpoint.** The Qwen 3.8 27B target is ~21.6 GB and the MTP
  head another ~849 MB. Neither is in the repository. This pipeline
  deliberately does not perform that large network fetch.

Everything that needs neither resource compiles and runs on hosted hardware,
and that is what CI gates. This covers protocol encoding and decoding, config
and artifact contracts, safetensors header parsing, scoring arithmetic, chat
templating, wire fixtures, and request validation.

## CI-covered

| Job | Runner | Check |
| --- | --- | --- |
| `lint` | `ubuntu-latest` | `tools/ci-workflow-egress-scan.sh` — nothing in `.github/` matches a credential-or-egress tripwire pattern (see [Deliberate non-goals](#deliberate-non-goals) for the exact set) |
| `lint` | `ubuntu-latest` | `actionlint` over `.github/workflows/`, with the ranked runner's label declared in `.github/actionlint.yaml` |
| `lint` | `ubuntu-latest` | shell syntax (`bash -n` / `sh -n`) over every tracked `*.sh` |
| `lint` | `ubuntu-latest` | `tools/lint-benchmark-manifest.py --gitlink-targets report` — the Yukon track manifest at rest |
| `lint` | `ubuntu-latest` | `tools/ci-box-only-inventory.sh --check` — the inventory below matches the sources |
| `swift` | `macos-26` | toolchain preflight: a full Xcode is selected, Swift >= 6.3 |
| `swift` | `macos-26` | `tools/test-submission-security.sh` — the submission-restriction HARD GATE, all 184 hostile-archive assertions. It sits in the `swift` job because it compiles the real byte-budget enforcer with `swiftc`. Nothing in it is box-dependent, so no assertion is gated off. See [What "hard gate" does and does not mean](#what-hard-gate-does-and-does-not-mean) |
| `swift` | `macos-26` | the suite's **non-vacuity floor**, restated workflow-side against its printed trailer — the run reds if the suite stopped short, reported any failure, or ran fewer assertions than `CI_MIN_ASSERTIONS`, and reds again if that number has drifted from the suite's own `EXPECTED_MIN_ASSERTIONS` |
| `swift` | `macos-26` | `swift build --build-tests --force-resolved-versions` — `MLXFastCore`/`Transform`/`Model`/`Harness`, both executables, and the test bundle |
| `swift` | `macos-26` | `tools/ci-swift-warning-gate.sh` — a warning in `Sources/` or `Tests/` fails the run |
| `swift` | `macos-26` | `git diff --exit-code -- Package.swift Package.resolved` — the frozen dependency graph was not re-resolved |
| `swift` | `macos-26` | `swift test --force-resolved-versions` with `MLXFAST_RUN_MLX_RUNTIME_TESTS=0` |

### What "hard gate" does and does not mean

**This pipeline is ADVISORY, and that is the accepted posture** (ruled by David
2026-08-20: the two-session red-team is the de facto merge gate). CI runs
`tools/test-submission-security.sh` on every pull request and every push to
`main`, and reds the run on a regression. That is the whole of it. **No status
check is required anywhere in this repository, and a red run blocks nothing
mechanically** — not a merge, not a dispatch:

- **there are no required status checks.** Branch protection is unavailable on
  this repository's plan — the REST endpoint answers `403 Upgrade to GitHub Pro
  or make this repository public` — so no required-status-check rule exists, or
  can be configured, for any job in this file;
- there is no `CODEOWNERS` in this tree, so nothing mechanically requires a
  review either;
- the ranked workflow (`.github/workflows/qwen38-mtp-ranked-benchmark.yml`) is a
  fail-closed stub and never invokes the suite;
- a **ranked** `workflow_dispatch` does not consult `ci.yml`, so dispatching a
  benchmark does not consult the gate. (This file carries its own
  `workflow_dispatch` trigger, for re-running CI by hand — a different thing.)

None of that is a defect to be worked around. What holds the line is a person
who reads the run, plus the independent red-team passes that gate a merge.
Dispatch-time gating is separate work that lands with the ranked runner: see
`docs/submission-restriction-spec.md` §9 (not ported) and §10 (the hard gate).

Do not describe any check in this file as "required". Do not add prose implying
that it blocks a merge or a dispatch. If anyone ever configures required
checks, change this section first.

### On the non-vacuity floor

The suite's exit status is not the whole gate. A section can stop running: it
can be deleted, renamed out of the flow, or skipped by an early `continue`.
Such a section leaves every remaining assertion green. The suite also runs
`set -uo pipefail` *without* `-e`. The suite carries `EXPECTED_MIN_ASSERTIONS` for exactly this. That check lives
in the suite's own trailer, so truncating the file removes the floor along with
the assertions it guarded.

So the workflow also states the floor from outside, against the trailer the
suite prints. `tools/ci-swift-warning-gate.sh` uses the same separation to
guard the build from a different file than the build.
`tools/ci-workflow-egress-scan.sh` uses it to keep its patterns outside the
directory it scans. The workflow's `CI_MIN_ASSERTIONS` and the suite's
`EXPECTED_MIN_ASSERTIONS` are cross-checked against each other, so they cannot
drift apart in silence. Raise both in the commit that adds assertions.

### On warnings-as-errors

The gate is a log scan (`tools/ci-swift-warning-gate.sh`), not
`-Xswiftc -warnings-as-errors`, for two reasons. That flag is global, and this
package's graph is mostly vendored third-party code whose warnings belong to
upstream. The per-target alternative, `swiftSettings` in `Package.swift`, is
unavailable because `Package.swift` sits inside the frozen trusted-harness
source scope. So the gate binds where we own the code — `Sources/` and
`Tests/` — and reports vendored warnings as a count.

Widening the gate's exclusion list to make a first-party warning pass is not an
accepted fix. Fix the cause.

## Box-only: real gates that CI does NOT run

### Test files behind an opt-in environment gate

These run on an Apple Silicon box with the checkpoint staged
(`./setup.sh && ./setup-qwen-mtp.sh`). Set the named variable to `1`:

    MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test --force-resolved-versions

The inventory below is generated. Regenerate it with
`tools/ci-box-only-inventory.sh`. Its shape is `<gate> <file> <references>`:

<!-- BEGIN box-only-inventory (generated by tools/ci-box-only-inventory.sh) -->
```
MLXFAST_RUN_GEMMA4_UPSTREAM_EQUIVALENCE Tests/MLXFastTests/Model/Gemma4A4BUpstreamEquivalenceTests.swift 1
MLXFAST_RUN_MLX_RUNTIME_TESTS Tests/MLXFastTests/CohortReferenceReplayTests.swift 4
MLXFAST_RUN_MLX_RUNTIME_TESTS Tests/MLXFastTests/MTPAcceptanceRuleAuditTests.swift 4
MLXFAST_RUN_MLX_RUNTIME_TESTS Tests/MLXFastTests/MTPTargetForwardIdentityTests.swift 2
MLXFAST_RUN_MLX_RUNTIME_TESTS Tests/MLXFastTests/MTPVerificationStrategySealTests.swift 3
MLXFAST_RUN_MLX_RUNTIME_TESTS Tests/MLXFastTests/Model/Gemma4A4BUpstreamEquivalenceTests.swift 5
MLXFAST_RUN_MLX_RUNTIME_TESTS Tests/MLXFastTests/Model/LagunaNVFP4KernelTests.swift 5
MLXFAST_RUN_MLX_RUNTIME_TESTS Tests/MLXFastTests/Model/MLXTensorBridgeTests.swift 1
MLXFAST_RUN_MLX_RUNTIME_TESTS Tests/MLXFastTests/Model/NAXSplitKGEMMTests.swift 1
MLXFAST_RUN_MLX_RUNTIME_TESTS Tests/MLXFastTests/Model/NVFP4QuantizedMMTests.swift 1
MLXFAST_RUN_MLX_RUNTIME_TESTS Tests/MLXFastTests/RuntimeWorkerFreeRunLegIdentityTests.swift 2
MLXFAST_RUN_MLX_RUNTIME_TESTS Tests/MLXFastTests/RuntimeWorkerMTPRoundExecutionTests.swift 4
MLXFAST_RUN_MLX_RUNTIME_TESTS Tests/MLXFastTests/RuntimeWorkerRecordingExecutorTests.swift 2
MLXFAST_RUN_MLX_RUNTIME_TESTS Tests/MLXFastTests/RuntimeWorkerSpecConfigTests.swift 2
MLXFAST_RUN_MLX_RUNTIME_TESTS Tests/MLXFastTests/RuntimeWorkerSupportTests.swift 2
MLXFAST_RUN_MLX_RUNTIME_TESTS Tests/MLXFastTests/TargetQuantizationBindTests.swift 1
MLXFAST_RUN_MLX_RUNTIME_TESTS Tests/MLXFastTests/WidthProbeMachineryTests.swift 1
```
<!-- END box-only-inventory -->

`MLXFAST_RUN_QWEN_REFERENCE_PARITY` additionally needs
`MLXFAST_QWEN_REFERENCE_WEIGHTS_PATH` pointed at a transformed weights tree. It
is checkpoint-resident, not just GPU-resident. It currently gates no test in
this tree. The `GATES` array keeps the name so that a returning parity test is
inventoried rather than running unseen.

`MLXFAST_RUN_LAGUNA_UPSTREAM_EQUIVALENCE` and
`MLXFAST_LAGUNA_EQUIVALENCE_WEIGHTS_PATH` were dropped with the Laguna runtime
tower. The Gemma equivalent is `MLXFAST_RUN_GEMMA4_UPSTREAM_EQUIVALENCE`
(`Tests/MLXFastTests/Model/Gemma4A4BUpstreamEquivalenceTests.swift`). It is now
listed in the `GATES` array and inventoried above. It needs the real
transformed target weights, so it runs on the box and not in CI.

### Other gates CI does not attempt

| Not run in CI | Why | Where it runs |
| --- | --- | --- |
| ~~The two `benchd/scripts/benchmark.sh` command targets in `benchmark.json`~~ | **No longer applicable.** benchd stopped being a source submodule; every `benchmark.json` command target now lives in this repository (`./setup.sh`, `./tools/fetch-benchd.sh`, `./tools/gemma4-measure-and-score.sh`), so CI verifies all three and the linter downgrades nothing | CI, fully verified |
| `tools/build-mlx-metallib.sh` (the AOT `mlx.metallib`) | needs the Metal toolchain component and produces a GPU artifact CI cannot exercise | box, via `./setup.sh` |
| `swift build -c release` into the two production scratch roots (`.build`, `.build-worker`) | the scored binary layout; CI proves the code compiles, not that the release layout is staged | box, before a benchd pass |
| Correctness gates, token fidelity, the timed paired measurement, scoring | measurement authority is benchd's, on the ranked M5 box, behind a thermal gate | the ranked pipeline; nothing in this repository times or scores anything |
| Hidden-golden fidelity | hidden material, R2-provisioned, sha256+bytes pinned | ranked box only |

## Deliberate non-goals

- **No secrets.** Not a token, not a deploy key, not an R2 credential.
- **No artifact upload from CI.** This repository can receive organizer-material
  fixtures, and the runner must not be a way out for them. The one exception is
  the ranked pipeline's score file, which is the product Yukon reads back:
  `.github/workflows/benchmark.yml` uploads `score.json` as
  `benchmark-results-<run_id>`. The scan checks that upload rather than
  exempting it. **Exactly one** upload reference may exist in that file. That
  one step's own block must carry a 40-hex-pinned action, `path: score.json`,
  and `if-no-files-found: error`. A second upload step fails the scan however
  well-formed it is. The three property checks are evaluated inside the checked
  step, so another step cannot vouch for it.
- **Both are tripwired, not proven.** `tools/ci-workflow-egress-scan.sh` scans
  all of `.github/` — workflows and any composite action beside them — and
  fails the run on a named set of patterns: a `secrets` reference
  (`secrets.NAME`, `secrets['NAME']`, `toJSON(secrets)`), `secrets: inherit`,
  any `write` permission (blanket or single-scope, `id-token: write`
  included), and an artifact upload by action or by expression anywhere other
  than the checked ranked-score upload above. That set is
  the whole of what it detects. It cannot see a `run:` step that curls, a
  third-party action that uploads on its own, or an `actions/cache` entry used
  as a side channel. Those remain review's job. The scan is the floor that
  stops the common accidental regressions landing silently. It is not a proof
  that nothing leaves the runner.
- **No ranked measurement.** `.github/workflows/qwen38-mtp-ranked-benchmark.yml`
  is a fail-closed stub and stays one. CI does not dispatch it, and a
  `workflow_dispatch` of it exits 1 by design.
- **No submodule checkout.** See the table above.
