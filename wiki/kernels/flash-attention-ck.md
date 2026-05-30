---
id: kernel-flash-attention-ck
title: "FlashAttention-2 via CK-tile on CDNA (MI300X)"
type: kernel
architectures:
- gfx942
- gfx950
tags:
- flash-attention
- attention
- composable-kernel
- mfma
- lds-double-buffering
- bf16
- softmax
- kernel-fusion
confidence: source-reported
reproducibility: snippet
kernel_types:
- flash-attention
- attention
languages:
- composable-kernel
- hip
hardware_features:
- mfma
- lds
- agpr
- async-copy
techniques:
- mfma-pipelining
- lds-double-buffering
- kernel-fusion
- wave-reduce
related:
- hw-mfma
- technique-lds-double-buffering
- technique-mfma-pipelining
- lang-composable-kernel
- kernel-paged-attention
- kernel-mla-decode
sources:
- blog-cktile-flash
- doc-flash-attention-2
- blog-flash-attention-amd
- hw-mfma
- technique-lds-double-buffering
- ref-composable-kernel
performance_claims:
- gpu: MI300X
  dtype: bf16
  metric: TFLOPS
  value: 620
  shape: "batch=2, heads=32, seqlen=8192, headdim=128, causal"
  utilization: "~47% of 1307 TFLOPS bf16 peak"
  source_id: blog-flash-attention-amd
- gpu: MI300X
  dtype: bf16
  metric: TFLOPS
  value: 450
  shape: "batch=4, heads=32, seqlen=4096, headdim=128, non-causal"
  source_id: blog-cktile-flash
---

# FlashAttention-2 via CK-tile on CDNA (MI300X)

## Overview

FlashAttention-2 computes `O = softmax(Q·Kᵀ · scale + mask) · V` without ever
materializing the `S = Q·Kᵀ` score matrix in HBM. Instead it tiles the sequence
dimension and streams K/V blocks through a running **online-softmax** recurrence,
keeping the working state (`O` accumulator, running max `m`, running sum `l`) in
registers. On CDNA this maps cleanly onto the matrix cores: both GEMMs
(`Q·Kᵀ` and `P·V`) are issued as [`v_mfma_*`](../hardware/mfma.md) instructions,
and the elementwise softmax between them is fused into the same kernel so the
score tile never leaves the chip.

The AMD implementation lives in [Composable Kernel](../languages/composable-kernel.md)'s
`ck_tile` layer (the `fmha` / FlashAttention pipeline). `ck_tile` exposes the
warp→block→pipeline→kernel tiers, so the two MFMA GEMMs and the softmax epilogue
are expressed as composable tile programs over **distributed tensors** rather
than hand-rolled assembly. This is the kernel that backs the
`ck` backend of [AITER](../../sources/refs/ref-aiter.md) and the AMD fork of
`flash-attention`.

## Algorithm — the online-softmax recurrence

For a query tile `Q_i` (rows `Br`), iterate over K/V tiles `j` (`Bc` columns):

```text
m_i      = -inf            # running row-max          (Br)
l_i      =  0              # running row-sum          (Br)
O_i      =  0              # output accumulator       (Br x d), in AGPRs
for j in key_blocks:
    S_ij   = scale * (Q_i · K_jᵀ)          # MFMA GEMM-0  -> FP32 score tile
    m_new  = max(m_i, rowmax(S_ij))        # wave reduction across the Bc tile
    P_ij   = exp(S_ij - m_new)             # elementwise, FP32
    alpha  = exp(m_i - m_new)              # rescale factor for old state
    l_i    = alpha * l_i + rowsum(P_ij)
    O_i    = alpha * O_i + (P_ij_bf16 · V_j)   # MFMA GEMM-1
    m_i    = m_new
O_i = O_i / l_i            # final normalization
```

FlashAttention-2's key change over v1 is that the expensive `1/l` division is
hoisted **out** of the inner loop (applied once at the end), and the work is
partitioned so the outer loop is over query blocks — maximizing matrix-core
occupancy and minimizing non-MFMA rescaling. See the
[FlashAttention-2 paper](../../sources/docs/doc-flash-attention-2.md).

## Mapping onto CDNA matrix cores

Both GEMMs run on the matrix unit. For bf16 inputs the natural tile is the
`32x32x8` or `16x16x16` MFMA shape (FP32 accumulate) described on the
[MFMA page](../hardware/mfma.md). The critical CDNA-specific detail is the
**layout hand-off between the two GEMMs**: GEMM-0 produces `S_ij` in the matrix
core's FP32 accumulator register layout (AGPRs), but GEMM-1 needs `P_ij` as a
*bf16 input operand* in the A-matrix layout. `ck_tile` performs this transform
with `shuffle_tile` (a register/LDS relayout) so that `P·V` can be issued
without a round trip through HBM.

```cpp
#include "ck_tile/ops/fmha.hpp"
// FlashAttention-2 forward (non-causal) pipeline assembled from ck_tile.
// Block tile: Br x Bc query/key blocks, head-dim d, FP32 softmax accum.
using FmhaShape = ck_tile::TileFmhaShape<
    ck_tile::sequence<128, 128, 32, 128, 32, 128>, // Br, Bc, d-tiles ...
    ck_tile::TileFmhaTraits</*kPadSeqLenQ=*/true,
                            /*kPadSeqLenK=*/true,
                            /*kPadHeadDim=*/true,
                            /*kHasBias  =*/false,
                            /*kStoreLSE =*/false,
                            /*occupancy =*/-1>>;

using Pipeline = ck_tile::BlockFmhaPipelineQRKSVS< // Q-resident, K/V streamed
    ck_tile::BlockFmhaPipelineProblem<
        ck_tile::bf16_t,        // Q dtype
        ck_tile::bf16_t,        // K dtype
        ck_tile::bf16_t,        // V dtype
        float,                  // softmax / S accumulate dtype (FP32)
        float,                  // O accumulate dtype (FP32, in AGPRs)
        ck_tile::bf16_t,        // O output dtype
        FmhaShape>>;
```

The `QRKSVS` pipeline name encodes the schedule: **Q** stays **R**esident in
registers/LDS while **K** and **S**(=`P`) and **V** are **S**treamed — i.e. each
query block is loaded once and reused across the entire key loop, which is the
data-reuse win that makes attention compute-bound rather than HBM-bound at long
sequence lengths.

## Overlapping memory and compute

Long-context prefill is dominated by the K/V stream from HBM. The pipeline
overlaps that stream with MFMA issue using **LDS double buffering** (see
[the technique page](../techniques/lds-double-buffering.md)): while the matrix
core consumes `K_j`/`V_j` from one LDS buffer, the next `K_{j+1}`/`V_{j+1}` are
loaded into a second buffer. On gfx942 the loads are gated with
`s_waitcnt vmcnt(...)` so MFMA on tile `j` proceeds while tile `j+1` is in
flight; on gfx950 the copies can additionally use
[direct-to-LDS](../hardware/async-copy-lds.md) (`global_load_lds_*`) to bypass
VGPRs entirely and free registers for a deeper accumulator tile.

```cpp
// Conceptual inner-loop skeleton (what the ck_tile pipeline emits):
load_tile(k_lds[buf],  k_window);          // buffer_load -> LDS (async)
for (int j = 0; j < num_kv_blocks; ++j) {
    int nxt = buf ^ 1;
    if (j + 1 < num_kv_blocks)
        load_tile(k_lds[nxt], k_window.step());   // prefetch next K
    auto s = ck_tile::gemm0(q_reg, k_lds[buf]);    // v_mfma_f32_32x32x8_bf16
    auto p = softmax_update(s, m, l, o_acc);       // exp + wave reduce + rescale
    o_acc = ck_tile::gemm1(p, v_lds[buf], o_acc);   // v_mfma_f32_32x32x8_bf16
    buf = nxt;
    __builtin_amdgcn_s_waitcnt(0);                  // gate next iteration
}
```

The row reductions (`rowmax`, `rowsum`) over the `Bc` score columns use a
[wave reduction](../techniques/wave-reduce.md): DPP within the 16-lane rows,
then `ds_bpermute`/`ds_swizzle` for the cross-row step — no HBM and no
`__syncthreads` needed because each wave owns a contiguous row band.

## Tuning knobs

| Knob | Effect | Notes |
|---|---|---|
| `Br` (query block) | rows per workgroup; ↑ data reuse, ↑ AGPR/VGPR pressure | 64/128 typical; AGPR-bound (see [VGPR budgeting](../techniques/vgpr-budgeting.md)) |
| `Bc` (key block) | softmax tile width; affects reduction cost vs MFMA count | 64/128 |
| MFMA shape | `32x32x8` vs `16x16x16` bf16 | wider-M shapes amortize softmax over more MAC |
| `occupancy` (`waves_per_eu`) | trades AGPR tile size vs latency hiding | `-1` lets CK pick |
| `kPadSeqLen*` | branchless [OOB guards](../techniques/buffer-oob-guard.md) for ragged seqlen | uses buffer-load OOB→0 semantics |
| causal masking | skips upper-triangular key blocks | ~2× fewer GEMMs at long seqlen |

Head dimension `d` must fit the accumulator tile; the common `d=128` case maps
to a 128-wide `O` accumulator held in AGPRs across the key loop, which is the
dominant register-pressure term.

## Performance

On MI300X the bf16 matrix peak is 1307 TFLOPS dense
([datasheet](../../sources/docs/doc-mi300x-datasheet.md)). FlashAttention is not
a pure GEMM — the fused softmax (transcendental `exp`, reductions, and the
`P→A` relayout) sits on the critical path between the two MFMA GEMMs — so even a
well-pipelined kernel reaches a fraction of GEMM peak. Reported CK-tile numbers:

- **~620 TFLOPS bf16** at `seqlen=8192, headdim=128, causal` (≈47% of bf16 peak),
  per the [AMD FlashAttention blog](../../sources/blogs/blog-flash-attention-amd.md).
- **~450 TFLOPS bf16** at `seqlen=4096, headdim=128, non-causal`, per the
  [ck_tile FlashAttention blog](../../sources/blogs/blog-cktile-flash.md).

These are `source-reported` and shape-dependent: utilization climbs with
sequence length (the `1/l` epilogue and Q-load amortize over more key blocks)
and with head dimension (deeper GEMMs per softmax). Decode-time attention
(`seqlen_q=1`) is memory-bound on the KV cache and is handled by a different
kernel — see [paged attention](paged-attention.md) and [MLA decode](mla-decode.md).

## Reproducing

```bash
# Build CK's FlashAttention example/benchmark
git clone https://github.com/ROCm/composable_kernel
cd composable_kernel
mkdir build && cd build
cmake -D CMAKE_PREFIX_PATH=/opt/rocm \
      -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
      -D GPU_TARGETS="gfx942" ..
make tile_example_fmha_fwd -j

# bf16, batch=2 heads=32 seqlen=8192 headdim=128, causal
./bin/tile_example_fmha_fwd -prec=bf16 -b=2 -h=32 -s=8192 -d=128 -mask=1
```

## Sources

- [Accelerating FlashAttention with ck_tile (ROCm blog)](https://rocm.blogs.amd.com/artificial-intelligence/flash-attention-ck/README.html)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [FlashAttention on AMD Instinct (ROCm blog)](https://rocm.blogs.amd.com/artificial-intelligence/flash-attention/README.html)
- [MFMA — AMD Matrix Core Instructions (CDNA)](../hardware/mfma.md)
- [Composable Kernel repository](https://github.com/ROCm/composable_kernel)
