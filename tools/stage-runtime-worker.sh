#!/usr/bin/env bash
# Stage the scored runtime worker and its mlx.metallib into the location the
# benchmarker (benchd) resolves.
#
# WHY THIS EXISTS. benchd resolves the scored worker at a FIXED workspace-relative
# path -- <workspace>/.build/release/mlxfast-runtime-worker -- and Metal loads
# mlx.metallib from the SAME directory as the running binary (Cmlx searches next
# to the executable). But the worker is BUILT under its own isolated SwiftPM
# scratch root (.build-worker) so a participant-code compile can never write into
# the trusted CLI's .build tree, and mlx.metallib is emitted next to that scratch
# binary (.build-worker/release/mlx.metallib). The build output and benchd's
# resolver therefore land in different directories, and the metallib is not beside
# the binary benchd launches -- so a run either fails to find the worker or finds
# one with no metallib and dies at its first MLXArray.
#
# This step copies the FINISHED pair -- the worker binary and its mlx.metallib --
# from the scratch root into .build/release, as a sibling pair, so benchd finds
# both with no manual copy. Only this trusted step performs the copy, and it runs
# AFTER the build, so the scratch-root build isolation is preserved: the
# participant compile still writes only into .build-worker.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
cd "${ROOT_DIR}"

repository_path() {
  local candidate="$1"
  if [[ "${candidate}" == /* ]]; then
    printf '%s\n' "${candidate}"
  else
    printf '%s/%s\n' "${ROOT_DIR}" "${candidate#./}"
  fi
}

BUILD_CONFIGURATION="${MLXFAST_SWIFT_CONFIGURATION:-release}"

# SOURCE pair -- the participant worker's isolated scratch root. Resolved exactly
# as setup.sh and tools/build-mlx-metallib.sh resolve them, so an operator override
# is honoured identically across all three.
RUNTIME_WORKER_BIN="$(repository_path \
  "${MLXFAST_RUNTIME_WORKER_EXECUTABLE:-.build-worker/${BUILD_CONFIGURATION}/mlxfast-runtime-worker}")"
MLX_METALLIB="$(repository_path \
  "${MLXFAST_MLX_METALLIB:-$(dirname "${RUNTIME_WORKER_BIN}")/mlx.metallib}")"

# DESTINATION pair -- the FIXED location benchd resolves. Not env-overridable: it
# is a contract with the benchmarker, not a preference.
STAGED_WORKER_BIN="$(repository_path ".build/${BUILD_CONFIGURATION}/mlxfast-runtime-worker")"
STAGED_METALLIB="$(dirname "${STAGED_WORKER_BIN}")/mlx.metallib"

if [[ ! -x "${RUNTIME_WORKER_BIN}" ]]; then
  echo "stage-runtime-worker.sh: scored runtime worker missing or not executable: ${RUNTIME_WORKER_BIN}" >&2
  echo "stage-runtime-worker.sh: build it first (setup.sh, or swift build -c ${BUILD_CONFIGURATION} --scratch-path .build-worker --product mlxfast-runtime-worker)" >&2
  exit 1
fi

mkdir -p "$(dirname "${STAGED_WORKER_BIN}")"

# `-ef` is true only when both paths already resolve to the same file, which
# happens when an operator override points the source straight at the benchd
# path -- then the copy is a no-op (and cp would error copying a file onto itself).
if [[ ! "${RUNTIME_WORKER_BIN}" -ef "${STAGED_WORKER_BIN}" ]]; then
  cp -f "${RUNTIME_WORKER_BIN}" "${STAGED_WORKER_BIN}"
fi
# benchd checks the execute bit on the resolved binary; guarantee it survives the
# copy regardless of the caller's umask.
chmod +x "${STAGED_WORKER_BIN}"

if [[ "${MLXFAST_SKIP_MLX_METALLIB:-0}" == "1" ]]; then
  # The metallib build was explicitly skipped; there is nothing to stage and the
  # operator has accepted that a real GPU run cannot succeed without it.
  metallib_staged=0
else
  if [[ ! -f "${MLX_METALLIB}" ]]; then
    echo "stage-runtime-worker.sh: mlx.metallib missing next to the scored worker: ${MLX_METALLIB}" >&2
    echo "stage-runtime-worker.sh: build it first (tools/build-mlx-metallib.sh)" >&2
    exit 1
  fi
  if [[ ! "${MLX_METALLIB}" -ef "${STAGED_METALLIB}" ]]; then
    cp -f "${MLX_METALLIB}" "${STAGED_METALLIB}"
  fi
  metallib_staged=1
fi

# ONE closing line, and it reports what this run actually did to the pair. The
# skip branch used to print "staged the worker only, no mlx.metallib" and then
# fall into an unconditional "staged ... with sibling mlx.metallib for benchd",
# so a MLXFAST_SKIP_MLX_METALLIB=1 run -- the normal case under
# ./setup-gemma4-assistant.sh, which skips the metallib because ./setup.sh owns
# it -- ended on two adjacent lines that contradicted each other about whether
# the metallib was there. The skip does not imply the sibling is absent: a
# previous run usually left one in place, and that is the case worth
# distinguishing from the one where a GPU run will die at its first MLXArray.
if [[ "${metallib_staged}" == "1" ]]; then
  echo "stage-runtime-worker.sh: staged ${STAGED_WORKER_BIN} with sibling mlx.metallib for benchd"
elif [[ -f "${STAGED_METALLIB}" ]]; then
  echo "stage-runtime-worker.sh: staged ${STAGED_WORKER_BIN}; MLXFAST_SKIP_MLX_METALLIB=1, kept the mlx.metallib already beside it"
else
  echo "stage-runtime-worker.sh: staged ${STAGED_WORKER_BIN}; MLXFAST_SKIP_MLX_METALLIB=1 and no mlx.metallib is beside it, so a GPU run cannot succeed yet"
fi
