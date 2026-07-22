#!/usr/bin/env bash
# Portable fused RMSNorm — builds AND runs on gfx950 (CDNA4).
# Pure HIP (FMA math, LDS, __shfl_down) with wave-size-agnostic reduction.
set -euo pipefail
cd "$(dirname "$0")"

ARCH="${1:-gfx950}"

echo "== Building rmsnorm for ${ARCH} =="
hipcc --offload-arch="${ARCH}" -O3 -std=c++17 rmsnorm.hip.cpp -o rmsnorm

echo "== Running =="
./rmsnorm
