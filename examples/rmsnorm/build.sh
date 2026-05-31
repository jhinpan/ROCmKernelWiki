#!/usr/bin/env bash
# Portable fused RMSNorm — builds AND runs on gfx1201 (RDNA4).
# Pure HIP (FMA math, LDS, __shfl_down). Also runs unchanged on CDNA (wave64).
set -euo pipefail
cd "$(dirname "$0")"

ARCH="${1:-gfx1201}"

echo "== Building rmsnorm for ${ARCH} =="
hipcc --offload-arch="${ARCH}" -O3 -std=c++17 rmsnorm.hip.cpp -o rmsnorm

echo "== Running =="
./rmsnorm
