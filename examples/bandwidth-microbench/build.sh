#!/usr/bin/env bash
# Build AND run the portable float4 HBM read-bandwidth microbenchmark.
# Runs natively on this gfx1201 (RDNA4) box.
set -euo pipefail
cd "$(dirname "$0")"

hipcc -O3 --offload-arch=gfx1201 bandwidth_memread.hip -o bwread

echo "=== verifying wide non-temporal loads in ISA ==="
hipcc -O3 --offload-arch=gfx1201 -S bandwidth_memread.hip -o bwread.s
grep -E -m4 'global_load_b128|global_load_dwordx4' bwread.s || \
    echo "(no dwordx4 found — check unroll)"

echo "=== running ==="
./bwread
