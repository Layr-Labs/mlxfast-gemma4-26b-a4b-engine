#!/usr/bin/env bash
# MTP vs baseline throughput benchmark using llama-benchy.
#
# Starts mlx-server for each model config (standard + MTP), runs llama-benchy
# against the OpenAI-compatible endpoint, then kills the server.
#
# Usage:
#   ./scripts/benchmark-mtp.sh [--pp 512 2048] [--tg 128] [--runs 3]
#
# Requirements:
#   - llama-benchy  (pip install llama-benchy or brew install llama.cpp)
#   - huggingface_hub Python package  (pip install huggingface_hub)
#   Models are downloaded automatically on first run via huggingface_hub.

set -euo pipefail

# ── defaults ─────────────────────────────────────────────────────────────────
BENCH_PORT=18765
PP_SIZES="512 2048"
TG_SIZES="128"
RUNS=3
CREATE_ISSUE=true
TIMESTAMP=$(date +"%Y-%m-%d-%H%M%S")
OUTDIR="benchmarks"
OUTFILE="${OUTDIR}/mtp-${TIMESTAMP}.md"

# ── parse CLI overrides ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --pp)        shift; PP_SIZES="$*"; break ;;
        --tg)        shift; TG_SIZES="$1"; shift ;;
        --runs)      shift; RUNS="$1";     shift ;;
        --no-issue)  CREATE_ISSUE=false;   shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── helpers ───────────────────────────────────────────────────────────────────
SERVER_PID=""

hf_path() {
    python3 - "$1" <<'EOF'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1]))
EOF
}

# ── download models ───────────────────────────────────────────────────────────
echo "=== Downloading / verifying models ==="

GEMMA4_26B=$(hf_path "mlx-community/gemma-4-26b-a4b-it-4bit")
GEMMA4_26B_DRAFTER=$(hf_path "google/gemma-4-26B-A4B-it-assistant")
GEMMA4_31B=$(hf_path "mlx-community/gemma-4-31b-it-4bit")
GEMMA4_31B_DRAFTER=$(hf_path "google/gemma-4-31B-it-assistant")
QWEN_27B=$(hf_path "mlx-community/Qwen3.6-27B-4bit")
QWEN_35B=$(hf_path "mlx-community/Qwen3.6-35B-A3B-4bit")

echo "  gemma-4-26b:         $GEMMA4_26B"
echo "  gemma-4-26b drafter: $GEMMA4_26B_DRAFTER"
echo "  gemma-4-31b:         $GEMMA4_31B"
echo "  gemma-4-31b drafter: $GEMMA4_31B_DRAFTER"
echo "  Qwen3.6-27B:         $QWEN_27B"
echo "  Qwen3.6-35B:         $QWEN_35B"

# ── output file setup ────────────────────────────────────────────────────────
mkdir -p "$OUTDIR"
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")
MACOS_VERSION=$(sw_vers -productVersion 2>/dev/null || echo "unknown")
MACOS_BUILD=$(sw_vers -buildVersion 2>/dev/null || echo "unknown")
SWIFT_VERSION=$(swift --version 2>/dev/null | head -1 || echo "unknown")
TOTAL_RAM_GB=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 / 1024 / 1024 ))
GPU_CORES=$(system_profiler SPDisplaysDataType 2>/dev/null | awk '/Total Number of Cores/{print $NF; exit}' || echo "unknown")
{
    echo "# MTP vs Baseline Benchmark"
    echo ""
    echo "## Environment"
    echo ""
    echo "| | |"
    echo "|---|---|"
    echo "| **Date** | $(date -u '+%Y-%m-%d %H:%M:%S UTC') |"
    echo "| **Commit** | \`${COMMIT}\` (${BRANCH}) |"
    echo "| **Chip** | ${CHIP} |"
    echo "| **GPU cores** | ${GPU_CORES} |"
    echo "| **RAM** | ${TOTAL_RAM_GB} GB |"
    echo "| **macOS** | ${MACOS_VERSION} (${MACOS_BUILD}) |"
    echo "| **Swift** | ${SWIFT_VERSION} |"
    echo "| **pp sizes** | ${PP_SIZES} |"
    echo "| **tg sizes** | ${TG_SIZES} |"
    echo "| **runs** | ${RUNS} |"
    echo ""
    echo "## Results"
    echo ""
} > "$OUTFILE"
echo "Results will be saved to: $OUTFILE"

# ── build ─────────────────────────────────────────────────────────────────────
echo ""
echo "=== Building (release) ==="
swift build -c release 2>&1 | tail -5
SERVER_BIN=".build/release/mlx-server"

start_server() {
    local model_path="$1"; shift
    "$SERVER_BIN" --model "$model_path" --port "$BENCH_PORT" \
        --no-prefix-cache "$@" >/tmp/mlx-server-$$.log 2>&1 &
    SERVER_PID=$!
    local tries=0
    until curl -sf "http://localhost:${BENCH_PORT}/health" >/dev/null 2>&1; do
        # Bail early if the server process has already exited.
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: server process exited before becoming ready" >&2
            cat /tmp/mlx-server-$$.log >&2 || true
            rm -f /tmp/mlx-server-$$.log
            SERVER_PID=""
            return 1
        fi
        tries=$((tries + 1))
        if [[ $tries -ge 300 ]]; then
            echo "ERROR: server did not start within 300 s" >&2
            kill "$SERVER_PID" 2>/dev/null || true
            SERVER_PID=""
            return 1
        fi
        sleep 1
    done
    rm -f /tmp/mlx-server-$$.log
    echo "  server ready (${tries}s)"
}

stop_server() {
    [[ -n "$SERVER_PID" ]] || return
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
}

trap stop_server EXIT

run_bench() {
    local label="$1"; shift
    local model_path="$1"; shift
    # remaining args forwarded to mlx-server (e.g. --mtp, --draft-model <path>)

    echo ""
    echo "━━━ ${label} ━━━"
    if ! start_server "$model_path" "$@"; then
        echo "  SKIPPED: server failed to start for '${label}'" >&2
        printf "### %s\n\n⚠️ Server failed to start (no MTP weights or other startup error).\n\n" \
            "$label" >> "$OUTFILE"
        return 0
    fi

    local result bench_exit
    result=$(KMP_DUPLICATE_LIB_OK=TRUE llama-benchy \
        --base-url "http://localhost:${BENCH_PORT}/v1" \
        --tokenizer "$model_path" \
        --pp $PP_SIZES \
        --tg $TG_SIZES \
        --runs "$RUNS" \
        --no-cache \
        --no-warmup \
        --skip-coherence \
        --latency-mode generation \
        --format md 2>&1) && bench_exit=0 || bench_exit=$?

    echo "$result"
    if [[ $bench_exit -ne 0 ]]; then
        echo "WARNING: llama-benchy exited $bench_exit for '${label}'" >&2
        printf "### %s\n\n⚠️ llama-benchy failed (exit %s):\n\`\`\`\n%s\n\`\`\`\n\n" \
            "$label" "$bench_exit" "$result" >> "$OUTFILE"
    else
        printf "### %s\n\n%s\n\n" "$label" "$result" >> "$OUTFILE"
    fi

    stop_server
}

# ── benchmarks ────────────────────────────────────────────────────────────────
echo ""
echo "=== Gemma 4 26B ==="
echo "## Gemma 4 26B" >> "$OUTFILE"
run_bench "gemma-4-26b-a4b  [baseline]" "$GEMMA4_26B"
run_bench "gemma-4-26b-a4b  [MTP]"      "$GEMMA4_26B" --draft-model "$GEMMA4_26B_DRAFTER"

echo ""
echo "=== Gemma 4 31B ==="
echo "## Gemma 4 31B" >> "$OUTFILE"
run_bench "gemma-4-31b  [baseline]" "$GEMMA4_31B"
run_bench "gemma-4-31b  [MTP]"      "$GEMMA4_31B" --draft-model "$GEMMA4_31B_DRAFTER"

echo ""
echo "=== Qwen3.6 27B ==="
echo "## Qwen3.6 27B" >> "$OUTFILE"
run_bench "Qwen3.6-27B  [baseline]" "$QWEN_27B"
run_bench "Qwen3.6-27B  [MTP]"      "$QWEN_27B" --mtp

echo ""
echo "=== Qwen3.6 35B ==="
echo "## Qwen3.6 35B" >> "$OUTFILE"
run_bench "Qwen3.6-35B  [baseline]" "$QWEN_35B"
run_bench "Qwen3.6-35B  [MTP]"      "$QWEN_35B" --mtp

# ── post results ──────────────────────────────────────────────────────────────
echo ""
echo "=== Results saved to: $OUTFILE ==="

if [[ "$CREATE_ISSUE" == true ]]; then
    echo ""
    echo "=== Creating GitHub issue ==="
    gh issue create \
        --title "MTP benchmark ${TIMESTAMP} (${COMMIT})" \
        --label "benchmark" \
        --body-file "$OUTFILE"
else
    echo ""
    echo "=== Skipping GitHub issue (--no-issue) ==="
fi

echo ""
echo "=== Done ==="
