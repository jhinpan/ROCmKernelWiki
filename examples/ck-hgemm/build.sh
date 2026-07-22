#!/usr/bin/env bash
# Portable rocWMMA FP16 GEMM — builds AND runs on gfx950 (CDNA4).
# rocWMMA is the API; the gfx950 device code uses MFMA instructions.
set -euo pipefail
cd "$(dirname "$0")"

hipcc --offload-arch=gfx950 -O3 -std=c++17 -I/opt/rocm/include \
      hgemm_wmma.cpp -o hgemm_wmma

echo "=== run (gfx950) ==="
./hgemm_wmma 256 256 256
