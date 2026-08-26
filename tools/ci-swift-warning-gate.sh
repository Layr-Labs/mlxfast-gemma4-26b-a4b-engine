#!/usr/bin/env bash
#
# Fail a build log that carries a compiler warning in FIRST-PARTY source.
#
# Why this exists instead of `-Xswiftc -warnings-as-errors`: that flag is
# global. SwiftPM applies it to every target in the graph, and this package's
# graph is mostly vendored third-party code (Vendor/mlx-swift,
# Vendor/mlx-swift-lm, the swift-transformers checkout) whose warnings are
# upstream's, not ours. Turning them all into errors would either wedge CI on
# code we do not own or push someone toward blanket suppressions. Package.swift
# is also not an option: it is inside the frozen trusted-harness source scope
# (see the target comments in Package.swift), so per-target swiftSettings are
# off the table too.
#
# So the gate is applied at the log instead: warnings from Sources/ and Tests/
# are errors; warnings from anywhere else are reported as counts and ignored.
#
# One category is deliberately outside the gate and is counted separately so it
# is not invisible: a warning whose primary location is a MACRO EXPANSION
# buffer ("macro expansion #someMacro:8:9: warning: ..."). The offending code is
# the vendored macro's, even though the expansion happens at a first-party call
# site, so first-party code cannot fix it without dropping the macro.
#
# A clean result from this gate only means something if the log it read is the
# log of a build that actually compiled. A fully-cached or no-op build emits no
# diagnostics at all, and "zero first-party warnings" over that log is a
# vacuous pass -- green for the wrong reason. So the gate asserts a floor of
# compile activity first and fails the run if the log has none.
#
# Usage:  tools/ci-swift-warning-gate.sh <build-log> [<build-log> ...]
# Exit:   0 no first-party warnings, 1 otherwise (each one printed),
#         1 if no log shows any compile activity at all.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <build-log> [<build-log> ...]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Recover the source path from a "<path>:<line>:<col>: warning: ..." line and
# reduce it to a repo-relative one, so the Sources/ and Tests/ tests below hold
# whichever form the compiler emitted. Three forms have to collapse to one:
#   ./Sources/x.swift          relative, dot-prefixed
#   <repo_root>/Sources/x.swift            absolute
#   /private<repo_root>/Sources/x.swift    macOS firmlink form of the same file
# Anything that is not under the repo root is returned unchanged and buckets as
# vendored/dependency, which is the correct default for a path we cannot place.
normalize_path() {
  local p="$1"
  p="${p#./}"
  if [[ "${p}" == "${repo_root}/"* ]]; then
    p="${p#"${repo_root}"/}"
  elif [[ "${p}" == "/private${repo_root}/"* ]]; then
    p="${p#"/private${repo_root}"/}"
  elif [[ "${repo_root}" == /private/* && "${p}" == "${repo_root#/private}/"* ]]; then
    p="${p#"${repo_root#/private}"/}"
  fi
  printf '%s' "${p#./}"
}

first_party=0
third_party=0
macro_expansion=0
compile_activity=0
failed=0

for log in "$@"; do
  if [[ ! -f "${log}" ]]; then
    echo "ci-swift-warning-gate: no such log: ${log}" >&2
    exit 2
  fi

  # A diagnostic line is "<path>:<line>:<col>: warning: <message>". Anything
  # else that happens to contain the word "warning" is not a diagnostic and is
  # not this gate's business.
  #
  # The path is NOT a whitespace-delimited field: a checkout directory can
  # contain a space (GitHub's macOS images have historically had them), and
  # matching the path as one would drop those diagnostics from the gate
  # entirely. So the match is anchored on the ":<line>:<col>: warning: " tail
  # and the path is whatever precedes it.
  while IFS= read -r line; do
    # "macro expansion #someMacro:8:9: warning: ..." has that same tail but no
    # path in front of it; it is counted separately below, not bucketed here.
    case "${line}" in
      'macro expansion '*) continue ;;
    esac

    path="${line%:*:*: warning: *}"
    rel="$(normalize_path "${path}")"

    case "${rel}" in
      Sources/*|Tests/*)
        first_party=$((first_party + 1))
        failed=1
        echo "FAIL  first-party warning: ${line}"
        # GitHub renders this inline on the PR diff.
        echo "::warning file=${rel}::${line#*: warning: }"
        ;;
      *)
        third_party=$((third_party + 1))
        ;;
    esac
  done < <(grep -E '^[^[:space:]].*:[0-9]+:[0-9]+: warning: ' "${log}" || true)

  macro_expansion=$((
    macro_expansion + $(grep -cE '^macro expansion #[^:]+:[0-9]+:[0-9]+: warning: ' "${log}" || true)
  ))

  # SwiftPM's progress lines ("[412/1146] Compiling ...") plus the bare
  # "Compiling"/"Emitting" forms. Either one proves the log came from a build
  # that did work, which is what makes a zero-warning result meaningful.
  compile_activity=$((
    compile_activity + $(grep -cE '^(\[[0-9]+/[0-9]+\] )?(Compiling|Emitting) ' "${log}" || true)
  ))
done

echo
echo "ci-swift-warning-gate: ${first_party} first-party (Sources/, Tests/), ${third_party} vendored/dependency, ${macro_expansion} inside vendored macro expansions, ${compile_activity} compile steps"

if [[ ${compile_activity} -eq 0 ]]; then
  cat >&2 <<'EOF'

FAIL  no compile activity in the build log(s) read.

This gate reads a build log. A log with nothing compiled in it yields zero
warnings no matter what the source says, so passing on one would be green for
the wrong reason. Either the build was a no-op against an up-to-date .build
(rebuild from clean, or point the gate at the log of a build that compiled), or
the build failed before compiling and the real error is above this line.
EOF
  exit 1
fi

if [[ ${failed} -ne 0 ]]; then
  cat >&2 <<'EOF'

First-party Swift warnings are errors here. Fix the cause -- do not silence it.
Adding an #[allow]-equivalent suppression, or widening this gate's exclusion
list, is not an accepted fix.
EOF
  exit 1
fi

echo "ci-swift-warning-gate: clean"
