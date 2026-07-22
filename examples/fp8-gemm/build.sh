#!/usr/bin/env bash
# Builds both examples in this directory:
#   1) PORTABLE rocWMMA FP16 GEMM  -> builds AND runs on gfx950 (this box)
#   2) CDNA-MFMA FP8 f8f6f4 GEMM   -> cross-compile-verify for gfx950 + gfx942
set -euo pipefail
cd "$(dirname "$0")"
source ../_common.sh

ROCM_INC=/opt/rocm/include

echo "==> [1/2] PORTABLE rocWMMA FP16 GEMM (gfx950, build + run)"
hipcc --offload-arch=gfx950 -I"${ROCM_INC}" wmma_hgemm.cpp -o wmma_hgemm
rocm_wiki_run ./wmma_hgemm

echo
echo "==> [2/2] CDNA-MFMA FP8 GEMM (cross-compile-verify only, NOT run here)"
echo "    -- gfx950 (CDNA4, OCP E4M3, f8f6f4 scaled MMA) object + exe"
hipcc --offload-arch=gfx950 -c fp8_gemm_cdna.cpp -o fp8_gemm_gfx950.o
hipcc --offload-arch=gfx950    fp8_gemm_cdna.cpp -o fp8_gemm_gfx950
echo "    -- gfx942 (CDNA3, FNUZ E4M3, K=32 fp8_fp8 MMA) object"
hipcc --offload-arch=gfx942 -c fp8_gemm_cdna.cpp -o fp8_gemm_gfx942.o

echo
echo "==> Confirm the matrix-core instructions were actually emitted:"
hipcc --offload-arch=gfx950 -S fp8_gemm_cdna.cpp -o /tmp/fp8_g950.s
grep -m1 -o "v_mfma_scale_f32_16x16x128_f8f6f4" /tmp/fp8_g950.s \
    && echo "    gfx950: f8f6f4 scaled MMA present"
hipcc --offload-arch=gfx942 -S fp8_gemm_cdna.cpp -o /tmp/fp8_g942.s
grep -m1 -o "v_mfma_f32_16x16x32_fp8_fp8" /tmp/fp8_g942.s \
    && echo "    gfx942: fp8_fp8 (FNUZ) MMA present"

echo
echo "build.sh: all targets built OK"
