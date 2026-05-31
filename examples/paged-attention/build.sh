#!/usr/bin/env bash
# Portable HIP paged-attention decode reference — builds AND runs on gfx1201.
set -euo pipefail
cd "$(dirname "$0")"

hipcc --offload-arch=gfx1201 -O3 paged_attention.cpp -o paged_attention

echo "build: OK"
./paged_attention
