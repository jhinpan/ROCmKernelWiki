---
id: kernel-flydsl-flash-attention
title: FlyDSL Flash Attention — generic + gfx950 dual-wave fast path
type: kernel
architectures:
- gfx942
- gfx950
tags:
- flash-attention
- attention
- flydsl
- mfma
- software-pipelining
- lds-double-buffering
- bf16
- softmax
- register-pressure
- kernel-profiling
confidence: source-reported
reproducibility: snippet
kernel_types:
- flash-attention
- attention
languages:
- flydsl
- mlir
- python
hardware_features:
- mfma
- lds
- async-copy
- ds-instructions
- agpr
- vgpr
- wave64
techniques:
- mfma-pipelining
- software-pipelining
- lds-double-buffering
- wave-reduce
- direct-to-lds
- vgpr-budgeting
- occupancy-tuning
related:
- lang-flydsl
- kernel-flash-attention-ck
- hw-mfma
- hw-async-copy-lds
- technique-mfma-pipelining
- technique-vgpr-budgeting
- technique-occupancy-tuning
- kernel-flydsl-preshuffle-gemm
sources:
- ref-flydsl
- ref-flydsl-kernel-profiling
- doc-flash-attention-2
performance_claims:
- gpu: MI350X
  dtype: bf16
  metric: FlyDSL fwd kernel-time vs CK-tile FlashAttention (throughput ratio, >1 =
    FlyDSL faster)
  value: 0.92
  bucket: HEADROOM
  baseline: CK-tile FlashAttention
  shape: D=128, causal, seq % 256 == 0
  utilization: register-pressure-capped — 1-2 waves/SIMD resident, VGPR 175-251 live
  source_id: ref-flydsl-kernel-profiling
  unreproduced: true
implemented_by:
- pr-FlyDSL-225
- pr-aiter-2701
- pr-aiter-3072
- pr-aiter-2945
- pr-composable_kernel-1224
- pr-flash-attention-179
- pr-composable_kernel-1789
- pr-aiter-1383
---
# FlyDSL Flash Attention — generic + gfx950 dual-wave fast path

## Overview

FlyDSL ships a forward FlashAttention-2 kernel written in its
[Python + MLIR layout DSL](../languages/flydsl.md) rather than in C++/CK. The same
online-softmax math as the [CK-tile implementation](flash-attention-ck.md) — see
that page for the recurrence — but the kernel is **built** by tracing Python into
the `fly` dialect and lowering to ROCDL/MFMA. It exists as **two kernels behind one
dispatcher**:

1. **`flash_attn_generic.py`** — the portable, compiler-scheduled kernel. The public
   builder `build_flash_attn_func_module_primary(...)` auto-selects `BLOCK_M=128`
   (4 waves / 256 threads) or `BLOCK_M=256` (8 waves / 512 threads) by `B·S`, runs
   **GEMM-1 `K·Qᵀ`** so scores land in MFMA-32 register layout, keeps **P resident in
   registers** (no LDS round-trip), and feeds **GEMM-2 `Vᵀ·P`**. Online softmax (running
   max/sum, exp2, causal mask) is in registers.
2. **`flash_attn_gfx950.py`** — `build_flash_attn_dualwave_swp_module(...)`, a
   **hand-scheduled dual-wave software-pipelined** fast path. Same math as the
   `BLOCK_M=256` generic path, dispatched only when `gpu_arch ≥ gfx950`,
   `head_dim == 128`, `dtype ∈ {bf16, f16}`, and at runtime `seq_len % 256 == 0`,
   `seq_len ≥ 384`.

The generic builder imports and routes to the gfx950 module when those gates hold,
falling back gracefully otherwise.

## The gfx950 dual-wave fast path

What the hand-written schedule buys over the compiler-scheduled generic kernel:

- **Dual-wave time-multiplexing.** Two wave-groups co-resident on one EU are
  phase-shifted by an extra `s_barrier` in the prologue so group B's compute hides
  group A's KV-load latency; they realign at the epilogue.
- **Explicit software pipeline.** Prologue loads K0; an 8-cluster main loop runs with
  a **2-tile lookahead** (K/V prefetched two iterations ahead); a 14-cluster epilogue
  drains the tail.
- **gfx950 hardware intrinsics.** `ds_read_tr16_b64` (HW-transpose LDS read for V),
  `buffer_load…lds` ([direct-to-LDS](../hardware/async-copy-lds.md) DMA, gfx950 widens
  to 16 B), `permlane32_swap` wave reductions, and manual `sched_group_barrier` /
  `s_setprio` to pin MFMA↔VALU↔EXP issue windows.
- **Lazy rescale.** The O accumulator is rescaled only when `m_new − m_old > 8`
  (ballot-gated), skipping the correction on the common path.
- **log2-space softmax.** Q is pre-scaled by `1/√d · log2(e)` so the inner loop uses
  `exp2` and the operand stays in the gfx950 `exp2` accuracy window.

## How it is built (illustrative)

FlyDSL kernels are *built*, not just called — the builder emits a specialized MLIR
module per config. Conceptually:

```python
from kernels.flash_attn_generic import build_flash_attn_func_module_primary

# One entry point; it picks BLOCK_M and, on gfx950 D=128 bf16, the dual-wave
# software-pipelined fast path (flash_attn_gfx950.build_flash_attn_dualwave_swp_module).
launch = build_flash_attn_func_module_primary(
    num_heads=32, num_kv_heads=8,      # GQA: num_heads % num_kv_heads == 0
    head_dim=128, dtype_str="bf16",
    causal=True,                        # masks upper-triangular KV tiles
    gpu_arch="gfx950",                  # gates the dual-wave path
)
# launch(Q, K, V, O, seq_len) with grid = (B * num_q_tiles * H, 1, 1)
```

Most of the kernel body is **compile-time metaprogramming**: `const_expr(...)` branches
and `range_constexpr(...)` loops resolve when the module is built (the IR for one config
is ~40k lines), so feature toggles (`USE_HW_TR`, `USE_K16`, `ENABLE_DMA`, `CAUSAL`)
generate code rather than branch at runtime.

## Evolution (upstream PRs)

| PR | What it added |
|---|---|
| [#225](../../sources/prs/FlyDSL/PR-225.md) | Original FMHA kernel (MFMA-32, online softmax) |
| [#334](../../sources/prs/FlyDSL/PR-334.md) | Tile-M tuning + `BLOCK_M=128`/`256` runtime dispatch |
| [#462](../../sources/prs/FlyDSL/PR-462.md) | Clean-up: low-level MLIR → modern `fly.*`/`Vec`/Pythonic control flow |
| #629 | gfx950 dual-wave SWP kernel (`flash_attn_gfx950.py`); rename `flash_attn_func.py` → `flash_attn_generic.py` |
| #661 | Route MFMA through the **layout MMA-atom API** (`make_mma_atom` / `mma_atom_call_ssa`) — same perf, ~1k fewer lines |

(#629 / #661 post-date this wiki's PR-harvest cutoff and have no source page yet.)

## Performance (measured on MI350X)

The canonical measurement record is the
[MI350X rocprofv3 ATT sweep](../../sources/refs/ref-flydsl-kernel-profiling.md);
this page only carries the kernel-level interpretation. For `D=128` causal, the
frontmatter records a 0.92 FlyDSL/CK-tile throughput ratio in the **HEADROOM**
bucket. The actionable gap is [VGPR pressure](../techniques/vgpr-budgeting.md):
the trace is capped at 1–2 waves/SIMD, so the dual-wave SWP schedule is the
structural attempt to trade hand-scheduled register reuse for more latency hiding.

## See also

- [FlyDSL language guide](../languages/flydsl.md) — layout algebra, `@flyc.kernel`, the MMA-atom API
- [FlashAttention-2 via CK-tile](flash-attention-ck.md) — the C++ baseline + the online-softmax math
- [VGPR / AGPR budgeting](../techniques/vgpr-budgeting.md) · [MFMA pipelining](../techniques/mfma-pipelining.md)
- [MI350X profiling sweep & dashboard](../../sources/refs/ref-flydsl-kernel-profiling.md)

## Sources

- [FlyDSL reference repository](https://github.com/ROCm/FlyDSL)
- [FlyDSL kernel profiling dashboard (MI350X)](https://jhinpan.github.io/flydsl-kernel-profiling/)
- [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691)
- [MFMA — AMD Matrix Core Instructions](../hardware/mfma.md)
