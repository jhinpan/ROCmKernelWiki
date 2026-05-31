#!/usr/bin/env bash
# Portable rocWMMA grouped GEMM — builds AND runs on gfx1201 (RDNA4 WMMA).
# rocWMMA also abstracts MFMA on CDNA, so the same source is portable there.
set -euo pipefail
cd "$(dirname "$0")"

hipcc --offload-arch=gfx1201 -O3 -std=c++17 -I/opt/rocm/include \
      grouped_gemm_wmma.cpp -o grouped_gemm_wmma

echo "Build OK — running:"
./grouped_gemm_wmma
