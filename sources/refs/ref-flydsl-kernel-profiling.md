---
id: ref-flydsl-kernel-profiling
title: FlyDSL Kernel Profiling — MI350X rocprofv3 ATT Sweep & Dashboard
repo: jhinpan/flydsl-kernel-profiling
url: https://jhinpan.github.io/flydsl-kernel-profiling/
author: Jin Pan
source_category: reference-repo
architectures:
- gfx950
tags:
- flydsl
- profiling
- rocprofv3
- kernel-profiling
- register-pressure
- mfma
languages:
- flydsl
- python
- mlir
retrieved_at: '2026-06-08'
---

# FlyDSL Kernel Profiling — MI350X rocprofv3 ATT Sweep & Dashboard

A first-party profiling study of **every major FlyDSL gfx950 kernel** captured with
**rocprofv3 ATT (Advanced Thread Trace) + hardware counters on real AMD Instinct
MI350X silicon**, with matched-shape baselines from AITER / Composable Kernel /
hipBLASLt. Each kernel ships a reproducible bundle (REPORT.md + ATT trace +
counters + source) and the results are browsable as an interactive GitHub Pages
dashboard.

- **Dashboard:** <https://jhinpan.github.io/flydsl-kernel-profiling/>
- **Repo:** <https://github.com/jhinpan/flydsl-kernel-profiling>

## Method

- **GPU:** 8× AMD Instinct MI350X (`gfx950`, CDNA4). **Stack:** ROCm 7.2.0.
- **FlyDSL:** 0.1.9.dev594 @ `18c5a7ed`.
- **Capture:** rocprofv3 ATT (95–100% source-mapped per kernel) + counter pass;
  `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` for source attribution.
- **17 kernels** ATT-profiled; **15** compared against matched-shape AITER / CK /
  hipBLASLt baselines. Ratios below are **FlyDSL throughput ÷ baseline throughput**
  (`>1` = FlyDSL faster).

## Verdicts (MI350X, gfx950)

| Bucket | Kernel | FlyDSL vs baseline | Baseline |
|---|---|---|---|
| **WIN** | softmax | **2.05×** | Triton |
| **WIN** | hgemm_splitk | **1.66×** | CK / hipBLASLt |
| **WIN** | moe_gemm | **1.11×** (stage2-atomic 1.30×) | AITER |
| PARITY | layernorm, quant, moe_reduce | ~1.0× | AITER |
| HEADROOM | moe_blockscale | 0.82× | tuned-CK |
| HEADROOM | rmsnorm | 0.89× | AITER |
| HEADROOM | mla (decode) | 0.90× | AITER |
| HEADROOM | flash_attn | 0.92× | CK-tile |
| HEADROOM | paged-attention | 0.48× | AITER |
| **HEADROOM** | topk_gating | **0.22×** | AITER |
| **HEADROOM** | rope | **0.17×** | AITER |

GEMM re-measured at compute-bound shapes: preshuffle **0.77×**, blockscale **0.66×**
vs tuned-CK; an internal v2 path is **1.20×** over v1.

## Key findings

- **Register-pressure-capped occupancy** is the dominant headroom on the attention /
  GEMM losers (mla, pa, flash_attn, moe_blockscale): only **1–2 waves/SIMD** resident,
  **VGPR 175–251** live. Cutting the live VGPR set to admit a 2nd wave is the lever.
  (See [register/AGPR budgeting](../../wiki/techniques/vgpr-budgeting.md).)
- **rope / topk_gating** are slow because cross-lane reductions serialize on
  `shuffle_xor` / `ds_bpermute` against `LGKMCNT`; the fix is a DPP / `v_permlane16`
  [wave reduction](../../wiki/techniques/wave-reduce.md).
- **softmax fast path (`BufferCopy128b`), filed as FlyDSL #627 / fixed in #650:** was
  dead-coded behind `False and`. Re-enabling it is **not** a 2× win — measured A/B
  (fast vs scalar, MI350X) is **on-par to +7% on large bf16, neutral/slightly-negative
  on f32**; both paths already saturate HBM (~5 TB/s) and register-buffer the whole row,
  so vectorization only trims instruction count. (The 2.05× headline is FlyDSL-vs-Triton,
  not fast-vs-scalar.)
- **`fp8_gemm_4wave` (rowscale) fails to compile** — `flyc.compile(): missing
  _reusable_slot_spec` on the fast-dispatch path. A real, config-independent regression.

## Reference

- Dashboard: <https://jhinpan.github.io/flydsl-kernel-profiling/>
- Source / bundles: <https://github.com/jhinpan/flydsl-kernel-profiling>
- Silicon facts cross-check: [`VERIFICATION.md`](../../VERIFICATION.md)
