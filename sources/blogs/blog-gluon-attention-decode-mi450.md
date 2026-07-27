---
id: blog-gluon-attention-decode-mi450
title: 'Attention Decode on AMD MI450 GPUs: A Gluon Kernel Optimization Guide'
author: AMD ROCm
url: https://rocm.blogs.amd.com/software-tools-optimization/gluon-attention-decode-mi450/README.html
source_category: benchmark-blog
architectures:
- gfx1250
tags:
- attention
- decode
- flash-attention
- kv-cache
- fp8
- mxfp
- block-scale
- wmma
- tdm
- workgroup-cluster
- wgp
- lds
- split-k
- software-pipelining
- lds-triple-buffering
- tdm-tile-widening
- layout-conversion-avoidance
- gluon
- triton
- memory-bound
- high-hbm-bw
retrieved_at: '2026-07-27'
---

# Attention Decode on AMD MI450 GPUs: A Gluon Kernel Optimization Guide

Published 2026-07-27 by Pengzhan Zhao, Lixun Zhang, and Lei Zhang (antiagainst).

An MQA/GQA **attention decode** case study on the AMD Instinct **MI450** series,
written in **Gluon** (a Triton-based DSL in which every tensor carries an
explicit layout). The post walks four optimizations — WMMA tensor layout, TDM
data loading, software pipelining, and split-k with workgroup clusters — and
reports an early result of **85% of peak HBM bandwidth**.

> Scope note: MI450 / `gfx1250` is outside this wiki's active gfx942/gfx950
> publication scope. This page is retained as forward-looking provenance.

## Reported MI450 vs MI350 specifications

| Specification | MI350 series | MI450 series |
|---|---|---|
| VGPRs per SIMD | 512 | 1024 |
| Max LDS per WGP | 160 KB | 320 KB |
| HBM capacity | 288 GB | 432 GB |
| HBM bandwidth | 8.1 TB/s | 19.6 TB/s |

| Feature | MI350 series | MI450 series |
|---|---|---|
| Memory instruction | Global/Buffer load to LDS | TDM load to LDS |
| Load granularity | 32/96/128-bit vector loads | Descriptor-based tensor tiles |
| Memory units per WGP | 1 | 2 |

The post defines a **WGP** (Workgroup Processor) as the unit that runs a
workgroup, "named CU in earlier MI-series", containing **four SIMD32 units, each
with 32 lanes**; each SIMD32 owns its VGPRs (used by VALU and WMMA) and all
SIMD32 units in a WGP share LDS. It states MI450 has **256 workgroup
processors**. It does not state the wavefront width.

Two further MI450 features are described:

- **TDM** (a specialized data-movement unit): the kernel describes a tensor by
  address, shape, strides, and layout, then issues a bulk asynchronous transfer.
  A per-WGP cache sits at the same level as LDS; TDM can either write directly
  into LDS or route through that cache path. Direct-to-LDS requires an
  **innermost dimension of at least 128 bytes, with 256 bytes recommended**.
- **Workgroup clusters**: several workgroups, each on its own WGP, coordinate
  through hardware cluster barriers and can share data via multicast loads. L2
  is shared across a cluster.

## Gluon layout notes

- `AMDWMMALayout` describes the WMMA output layout; `DotOperandLayout` describes
  the operand layout. An `AMDWMMALayout` with instruction shape `[16, 16, 128]`
  for `wmma_scaled` generates `v_wmma_scale_f32_16x16x128_f8f6f4`, consuming FP8
  operands of `(16, 128)` and `(128, 16)`, reducing over K = 128, and producing
  an FP32 `(16, 16)` tile per wave.
- Setting `transpose=True` in `AMDWMMALayout` emits a transposed output **with no
  extra instructions**, which is how the QK output is made layout-compatible with
  the PV input.
- "K Width" is how many contiguous K-dimension elements land in one lane. The
  standard `16x16x128` WMMA assumes K Width 16, but the transposed QK output has
  K Width **8**, so the PV operand layout must be set to K Width 8 to match.
- Waves may be any power-of-two count, distributed over two dimensions of the
  WMMA output tile. Because online softmax reduces along `T_kv`, the post
  distributes waves along the grouped-Q dimension; for `H_q / H_kv = 32` it uses
  **2 waves**.
- Multi-CTA mapping: Triton `warp` = wavefront, `CTA` = workgroup, `CGA` =
  workgroup cluster. Layouts take a `cga_layout` field. Under a multi-CTA
  layout, "block size" means the size of the **whole cluster**.

## Reported instruction mix

For FP8 decode with `BLOCK_M = 32`, `BLOCK_N = 128`, `D = 128`, and two waves:

| Operation | Instruction | Count | Cycles |
|---|---|---|---|
| TDM K | `tensor_load_to_lds` | 1 | |
| LDS K | `ds_load_b128` | 32 | |
| QK | `v_wmma_scale_f32_16x16x128_f8f6f4` | 8 | 8 |
| MAX | `v_maximum3_f32` | 32 | 1 |
| FMA | `v_pk_fma_f32` | 32 | 1 |
| EXP | `v_exp_f32` | 64 | 2 |
| SUM | `v_pk_add_f32` | 32 | 1 |
| MUL | `v_pk_mul_f32` | 32 | 1 |
| CVT | `v_cvt_scalef32_pk8_fp8_f32` | 8 | 4 |
| TDM V | `tensor_load_to_lds` | 1 | |
| LDS V | `ds_load_tr8_b64` | 64 | |
| PV | `v_wmma_scale_f32_16x16x128_f8f6f4` | 8 | 8 |

Cycles are a "simplified per-instruction cost estimate based on hardware
specifications"; memory instructions are blank because their latency is modeled
separately.

## Reported pipeline cycle model

Assumes TDM latency of ~1000 cycles (an explicitly "empirical number to help us
reason about the pipeline") and fully hidden LDS latency, with two interleave
rules: one WMMA takes 8 cycles and hides 2 cycles of VALU; one EXP takes 2
cycles and hides 1 cycle of non-EXP VALU.

| Group | Total Cycles |
|---|---|
| QK | 64 |
| PV | 64 |
| VEC0 = MAX + FMA + EXP | 192 |
| VEC1 = SUM + MUL + CVT | 96 |
| QK + VEC0, interleaved | 176 |
| PV + VEC1, interleaved | 144 |
| Total, interleaved | 320 |

Double buffering therefore exposes ~680 cycles of TDM wait per load (1000 − 320);
triple buffering looks ahead two iterations for ~640 cycles of overlap, cutting
exposed wait to ~360 cycles.

## Reported performance

Evaluation settings: batch 64, 64 Q heads, 1 or 2 KV heads, KV sequence length
4096–65536, QKV in FP8 with a global scale. The metric is **effective
bandwidth** — total bytes read from and written to global memory divided by
kernel execution time. Peak read-only bandwidth was measured with
[BabelStream](https://github.com/UoB-HPC/BabelStream) on the same system and
reported as **20 TB/s**.

| Case | Effective bandwidth | Fraction of measured peak |
|---|---|---|
| GQA, `H_kv = 2` | 17.10 TB/s | 85% |
| MQA, `H_kv = 1` | 16.65 TB/s | 83% |

Throughput rises with sequence length; at short sequences the prologue and
split-k reduction overheads are a larger share of runtime. The post calls this
"an early result".

Captured stack: **ROCm 7.14.0**, **PyTorch 2.11.0**, Triton commit
[`ecfc626`](https://github.com/triton-lang/triton/commit/ecfc62692aaa6c36a48350fd0e09faa6e2eb304e),
driving
[`mxfp_fa_gfx1250.py`](https://github.com/triton-lang/triton/blob/main/third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py).
Reproduction invocations sweep `--seqlen_k` with
`--q_type e4m3 --kv_type e4m3 --batch 64 --seqlen_q 1 --num_q_heads 64
--head_sz 128 --pipelined --scale_type global --profile`.

Stated future work: memory prefetching, better instruction-level scheduling,
improved split-k reduction, paged KV cache, and MLA/DSA variants.

## Reference

- Upstream: <https://rocm.blogs.amd.com/software-tools-optimization/gluon-attention-decode-mi450/README.html>
- Kernel source: <https://github.com/triton-lang/triton/blob/main/third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py>
- Gluon examples: <https://github.com/triton-lang/triton/tree/main/third_party/amd/python/examples/gluon>
- Gluon documentation: <https://triton-lang.org/main/gluon/index.html>
