#!/usr/bin/env bash
# Build + run the portable HIP vector add (Part 1), then cross-compile the
# GCN inline-asm CDNA variant (Part 2, build-only).
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Part 1: portable HIP vadd (build + RUN on gfx1201) ==="
hipcc --offload-arch=gfx1201 -O3 vadd_hip.cpp -o vadd_hip
./vadd_hip

echo
echo "=== Part 2: GCN inline-asm vadd (CROSS-COMPILE-ONLY for gfx942) ==="
# Object-only build proves the GCN VMEM asm encodings assemble for CDNA3.
hipcc --offload-arch=gfx942 -O3 -c vadd_asm_gfx942.cpp -o vadd_asm_gfx942.o
echo "OK: vadd_asm_gfx942.o produced (not executed on gfx1201)"
ls -l vadd_asm_gfx942.o
