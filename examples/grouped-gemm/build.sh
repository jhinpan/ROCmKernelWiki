#!/usr/bin/env bash
# Portable rocWMMA API grouped GEMM — builds and runs on gfx950.
# The gfx950 device code uses MFMA instructions.
set -euo pipefail
cd "$(dirname "$0")"

hipcc --offload-arch=gfx950 -O3 -std=c++17 -I/opt/rocm/include \
      grouped_gemm_wmma.cpp -o grouped_gemm_wmma

echo "Build OK — running:"
./grouped_gemm_wmma
