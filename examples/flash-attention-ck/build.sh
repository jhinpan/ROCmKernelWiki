#!/usr/bin/env bash
# Portable HIP FlashAttention-2 forward reference.
# Builds AND runs on gfx950 (CDNA4); self-checks numerics vs a CPU reference.
set -euo pipefail
cd "$(dirname "$0")"
source ../_common.sh

hipcc --offload-arch=gfx950 -O3 flash_attention_fwd.hip -o flash_attention_fwd

echo "=== build OK, running ==="
rocm_wiki_run ./flash_attention_fwd
