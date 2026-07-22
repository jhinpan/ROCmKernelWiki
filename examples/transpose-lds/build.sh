#!/usr/bin/env bash
# Portable HIP LDS-staged transpose: build AND run on gfx950 (CDNA4).
set -euo pipefail
cd "$(dirname "$0")"
source ../_common.sh

hipcc --offload-arch=gfx950 -O3 transpose_lds.cpp -o transpose_lds
echo "build: OK"

rocm_wiki_run ./transpose_lds
