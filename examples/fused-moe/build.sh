#!/usr/bin/env bash
# Portable HIP fused-MoE reference. Builds AND runs on gfx950 (CDNA4).
set -euo pipefail
cd "$(dirname "$0")"

hipcc --offload-arch=gfx950 -O3 fused_moe.cpp -o fused_moe

echo "Build OK. Running..."
./fused_moe
