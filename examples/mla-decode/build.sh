#!/usr/bin/env bash
# Portable HIP MLA (Multi-Latent Attention) decode reference.
# Builds AND runs on gfx1201 (RDNA4); self-checks numerics vs a CPU reference.
set -euo pipefail
cd "$(dirname "$0")"

hipcc --offload-arch=gfx1201 -O3 mla_decode.cpp -o mla_decode
echo "build: OK"

./mla_decode
