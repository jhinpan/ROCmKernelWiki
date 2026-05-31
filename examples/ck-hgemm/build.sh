#!/usr/bin/env bash
# Portable rocWMMA FP16 GEMM — builds AND runs on gfx1201 (RDNA4).
# rocWMMA lowers mma_sync to WMMA here and to MFMA on CDNA from the same source.
set -euo pipefail
cd "$(dirname "$0")"

hipcc --offload-arch=gfx1201 -O3 -std=c++17 -I/opt/rocm/include \
      hgemm_wmma.cpp -o hgemm_wmma

echo "=== run (gfx1201) ==="
./hgemm_wmma 256 256 256
