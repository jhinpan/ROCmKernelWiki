#!/usr/bin/env bash
# Portable HIP FlashAttention-2 forward reference.
# Builds AND runs on gfx1201 (RDNA4); self-checks numerics vs a CPU reference.
set -euo pipefail
cd "$(dirname "$0")"

hipcc --offload-arch=gfx1201 -O3 flash_attention_fwd.hip -o flash_attention_fwd

echo "=== build OK, running ==="
./flash_attention_fwd
