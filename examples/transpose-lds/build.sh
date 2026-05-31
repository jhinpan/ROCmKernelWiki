#!/usr/bin/env bash
# Portable HIP LDS-staged transpose: build AND run on gfx1201 (RDNA4).
set -euo pipefail
cd "$(dirname "$0")"

hipcc --offload-arch=gfx1201 -O3 transpose_lds.cpp -o transpose_lds
echo "build: OK"

./transpose_lds
