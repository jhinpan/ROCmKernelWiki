#!/usr/bin/env bash
# Build + run the PORTABLE rocWMMA demonstration of the preshuffle-GEMM idea.
# Runs on gfx950, where the rocWMMA API emits MFMA, and self-checks numerics.
#
# The FlyDSL file (04_preshuffle_gemm_flydsl.py) is a reference snippet only:
# FlyDSL is not installed here and targets CDNA MFMA, so it is NOT built/run.
set -euo pipefail
cd "$(dirname "$0")"
source ../_common.sh

ARCH="${1:-gfx950}"

echo "== Building rocWMMA preshuffle GEMM demo for ${ARCH} =="
hipcc --offload-arch="${ARCH}" -O3 -I/opt/rocm/include \
      preshuffle_gemm_rocwmma.cpp -o demo

echo "== Running =="
rocm_wiki_run ./demo
