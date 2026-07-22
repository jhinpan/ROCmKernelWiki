#!/usr/bin/env bash
# Portable HIP MLA (Multi-Latent Attention) decode reference.
# Builds AND runs on gfx950 (CDNA4); self-checks numerics vs a CPU reference.
set -euo pipefail
cd "$(dirname "$0")"
source ../_common.sh

hipcc --offload-arch=gfx950 -O3 mla_decode.cpp -o mla_decode
echo "build: OK"

rocm_wiki_run ./mla_decode
