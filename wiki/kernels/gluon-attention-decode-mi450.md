---
id: kernel-gluon-attention-decode-mi450
title: Gluon MQA/GQA Attention Decode on MI450 — 85% of peak HBM bandwidth
type: kernel
architectures:
- gfx1250
tags:
- attention
- decode
- flash-attention
- kv-cache
- softmax
- fp8
- mxfp
- block-scale
- wmma
- tdm
- workgroup-cluster
- split-k
- software-pipelining
- lds-triple-buffering
- tdm-tile-widening
- layout-conversion-avoidance
- gluon
- triton
- memory-bound
- high-hbm-bw
confidence: source-reported
reproducibility: snippet
kernel_types:
- attention
- decode
- flash-attention
- kv-cache
- softmax
languages:
- gluon
- python
- triton
hardware_features:
- tdm
- wmma
- workgroup-cluster
- lds
- fp8
- mxfp
techniques:
- tdm-tile-widening
- software-pipelining
- lds-triple-buffering
- split-k
- layout-conversion-avoidance
evidence_basis:
- source_id: blog-gluon-attention-decode-mi450
  evidence_type: benchmark
related:
- hw-mi450-tdm
- hw-mi450-workgroup-cluster
- technique-tdm-tile-widening
- technique-attention-decode-pipelining
- technique-split-k
- kernel-mla-decode
- kernel-paged-attention
- lang-triton-amd
sources:
- blog-gluon-attention-decode-mi450
- hw-mi450-tdm
- hw-mi450-workgroup-cluster
- blog-gluon-gemm
performance_claims:
- gpu: MI450
  dtype: fp8
  metric: effective-hbm-bandwidth
  value: 17.10 TB/s
  shape: 'GQA H_kv=2, batch 64, H_q 64, D 128, T_q 1, T_kv 4096-32768, e4m3 with global scale'
  utilization: 85% of the 20 TB/s BabelStream read-only peak measured on the same system
  baseline: BabelStream read-only bandwidth
  source_id: blog-gluon-attention-decode-mi450
  confidence: source-reported
  unreproduced: true
- gpu: MI450
  dtype: fp8
  metric: effective-hbm-bandwidth
  value: 16.65 TB/s
  shape: 'MQA H_kv=1, batch 64, H_q 64, D 128, T_q 1, T_kv 8192-65536, e4m3 with global scale'
  utilization: 83% of the 20 TB/s BabelStream read-only peak measured on the same system
  baseline: BabelStream read-only bandwidth
  source_id: blog-gluon-attention-decode-mi450
  confidence: source-reported
  unreproduced: true
---
# Gluon MQA/GQA Attention Decode on MI450 — 85% of peak HBM bandwidth

> **Scope.** MI450 / `gfx1250` is **outside** this wiki's active gfx942/gfx950
> publication scope. This page is retained as forward-looking research. Every
> number is `source-reported` from a single vendor blog describing pre-release
> hardware, and **none of it was verified on silicon by this project** — unlike
> the gfx950 claims in [`VERIFICATION.md`](../../VERIFICATION.md). Do not use it
> to justify a gfx942/gfx950 design decision.

## Why decode is the interesting case

In long-context generation each new token must attend to every previous token in
the KV cache, so the kernel re-reads the entire history from HBM and does very
little arithmetic per byte. As context grows toward a million tokens, the
bottleneck moves decisively from the matrix units to the memory system — the
[memory-bound](../patterns/memory-bound.md) regime.

The workload shape makes this concrete. With `T_q = 1`:

```text
Q: [B, H_q,  1,    D]          # one token
K: [B, H_kv, T_kv, D]          # entire history
V: [B, H_kv, T_kv, D]
```

Standard MHA would waste nearly all compute on a single-token query. **MQA/GQA
rescues it**: because several Q heads share one KV head, those heads are grouped
into the matrix-multiply's `M` dimension, so one KV read serves `H_q / H_kv`
queries:

```text
Q: [B, H_kv, H_q / H_kv, D]    # grouped Q heads become the M dimension
```

Each `(batch, kv_head)` pair is independent, so the baseline launch is a
`(B, H_kv, 1)` grid where each program owns `Q: [H_q/H_kv, D]`, `K/V: [T_kv, D]`
and slides `(BLOCK_N, D)` KV tiles along `T_kv` in the usual
[FlashAttention](flash-attention-ck.md) online-softmax loop. That baseline is
where the four optimizations below start. The same grouping insight drives
[MLA decode](mla-decode.md) on CDNA, where weight absorption makes 128 heads
share one latent KV.

## 1. Choose WMMA layouts so no conversion is needed

Gluon requires every tensor to carry an explicit layout, and the layouts that
matter most are the matrix-multiply operands. An `AMDWMMALayout` with instruction
shape `[16, 16, 128]` for `wmma_scaled` selects
`v_wmma_scale_f32_16x16x128_f8f6f4`: two FP8 operands of `(16, 128)` and
`(128, 16)`, reduced over `K = 128`, producing an FP32 `(16, 16)` tile per wave.

Two decisions follow.

**Wave distribution.** Waves can be spread over either dimension of the output
tile. Because online softmax reduces along `T_kv`, spreading waves along that
axis would force cross-wave communication in the reduction — so waves go along
the **grouped-Q** dimension instead. But `H_q / H_kv` is small, which caps how
many waves are useful: at `H_q / H_kv = 32` the kernel uses **2 waves**, each
owning its own first operand and sharing the second.

**Avoiding a layout conversion in the hot loop.** When two Gluon tensors disagree
on layout, an explicit conversion is required — sometimes through LDS. In
attention this bites at exactly the worst place: the QK output becomes the PV
input after softmax, once per iteration. Two settings make them compatible for
free:

- `transposed=True` on the QK `AMDWMMALayout` emits a transposed output directly.
  The blog is explicit that this costs **no extra instructions**.
- The transposed QK output has **K Width 8**, not the 16 that a standard
  `16x16x128` WMMA assumes, so the PV operand layout must be declared with
  `k_width=8` to match.

```python
import triton.experimental.gluon.language as ttgl

def wmma_layout(shape, warp_bases, cga_layout):
    return ttgl.amd.AMDWMMALayout(
        version=3,
        transposed=True,                 # free transpose: QK out -> PV in
        warp_bases=warp_bases,           # waves along the grouped-Q axis
        reg_bases=[],
        instr_shape=[16, 16, 128],       # -> v_wmma_scale_f32_16x16x128_f8f6f4
        cga_layout=cga_layout,
        rank=len(shape),
    )

# Operand 0 = "A" side, operand 1 = "B" side.
q_layout = ttgl.DotOperandLayout(0, wmma_layout(shape, wb, cga), k_width=16)
k_layout = ttgl.DotOperandLayout(1, wmma_layout(shape, wb, cga), k_width=16)
# P/V must use K Width 8 to match the *transposed* QK output.
p_layout = ttgl.DotOperandLayout(0, wmma_layout(shape, wb, cga), k_width=8)
v_layout = ttgl.DotOperandLayout(1, wmma_layout(shape, wb, cga), k_width=8)
```

Getting `k_width` wrong does not fail to compile — it silently reintroduces a
per-iteration conversion, which is the expensive failure mode.

## 2. Make the KV loads take TDM's direct path

The hot loop is dominated by streaming K and V into LDS, and on MI450 that goes
through [TDM](../hardware/mi450-tdm.md). Using TDM naively is not enough: a
transfer only bypasses the small per-WGP cache and land directly in LDS if its
**innermost dimension is at least 128 B (256 B recommended)**. For FP8 K/V tiles
of shape `(BLOCK_N, D)` the innermost dimension is `D` — typically 128 — so the
default shape misses the recommended path.

The fix is to reshape K and V so TDM sees 256-byte rows, then restore the logical
tile view when reading from LDS. This is
[TDM tile widening](../techniques/tdm-tile-widening.md), and it is free for the
common contiguous case.

## 3. Pipeline four ways, with triple buffering

A two-stage memory/compute split leaves LDS reads, QK, softmax, and PV
serialized inside one compute block. Because `QK[i+1]` does not depend on
`softmax[i]`, and because WMMA and VALU use different ports, the work is
regrouped into four interleaved stages — with softmax split into
`VEC0 = MAX+FMA+EXP` and `VEC1 = SUM+MUL+CVT` so QK overlaps `VEC0` and PV
overlaps `VEC1`.

Even then, the blog's cycle model puts one interleaved iteration at ~320 cycles
against a TDM latency of ~1000, so double buffering leaves ~680 cycles exposed.
**Triple buffering** pushes the lookahead to two iterations (~640 cycles
overlapped, ~360 exposed). Full derivation:
[four-stage decode pipelining](../techniques/attention-decode-pipelining.md).

## 4. Split-k across a workgroup cluster

The baseline grid is only `B * H_kv` workgroups. Against **256 WGPs**, a small
batch or a single KV head cannot saturate memory bandwidth. Split-k partitions
the KV sequence into `S` slices for a `B * H_kv * S` grid, which also shortens
each workgroup's KV walk.

The partials then have to be merged. Conventionally that is a second kernel
communicating through global memory. MI450's
[workgroup clusters](../hardware/mi450-workgroup-cluster.md) allow it in **one
kernel**: write partials, `cluster.arrive()` / `cluster.wait()`, read back and
reduce — with cluster-shared L2 keeping the partials off the global path.

```python
from triton.experimental.gluon.language.amd.gfx1250 import cluster, tdm

# ... each CTA computes its own KV-slice partial and stores it ...
tdm.async_wait(0)     # drain my partial stores before publishing
cluster.arrive()
cluster.wait()
# ... reduction phase: same grid, CTAs redistributed along the head dimension ...
```

Under the multi-CTA layout the partial attention is 3D, with **Q shared across
all partitions** while the second operand and the output differ per partition:

```text
Q: [1, H_q / H_kv, D]
K: [S, T_kv / S, D]
V: [S, T_kv / S, D]
```

## Reported performance

Effective bandwidth = bytes read from and written to global memory ÷ kernel time.
The reference peak is a **BabelStream read-only measurement of 20 TB/s on the same
system** (the MI450 spec figure quoted in the blog is 19.6 TB/s).

| Case | Effective bandwidth | Fraction of measured peak |
|---|---|---|
| GQA, `H_kv = 2` | **17.10 TB/s** | **85%** |
| MQA, `H_kv = 1` | 16.65 TB/s | 83% |

Settings: batch 64, 64 Q heads, `D = 128`, `T_q = 1`, `T_kv` from 4096 to 65536,
QKV in FP8 `e4m3` with a global scale. Throughput **rises with sequence length**,
because at short `T_kv` the prologue and split-k reduction are a larger share of
runtime — the [tail-effect](../patterns/tail-effect.md) reasoning applied to
fixed per-launch cost.

Both figures carry `unreproduced: true`: there is no reproduction bundle in this
repository, the hardware is pre-release, and the blog itself calls the result
"early". The residual 15% is consistent with the pipeline analysis, which still
leaves ~360 cycles of TDM latency exposed per load.

## Reproducing

The kernel is open source; the blog pins its stack to **ROCm 7.14.0**,
**PyTorch 2.11.0**, and Triton commit
[`ecfc626`](https://github.com/triton-lang/triton/commit/ecfc62692aaa6c36a48350fd0e09faa6e2eb304e).

```bash
# H_kv = 2 (the 17.10 TB/s / 85% case); sweep --seqlen_k 4096..32768
python3 third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py \
  --q_type e4m3 --kv_type e4m3 --batch 64 --seqlen_q 1 --seqlen_k 4096 \
  --num_q_heads 64 --num_k_heads 2 --head_sz 128 \
  --pipelined --scale_type global --profile
```

Use `--num_k_heads 1` and `--seqlen_k` up to 65536 for the MQA case.

## Reading this on CDNA

Only the *reasoning* ports to gfx942/gfx950; none of the mechanisms do. On CDNA
there is no TDM (use [direct-to-LDS](../hardware/async-copy-lds.md)), no
workgroup cluster (split-k reduction stays a second kernel), and `tl.dot` lowers
to [MFMA](../hardware/mfma.md) rather than WMMA — with
[FP8 encoding differing by generation](../migration/gfx942-to-gfx950.md)
(FNUZ on gfx942, OCP on gfx950). For an in-scope, silicon-verified decode kernel
see [MLA decode](mla-decode.md) or
[paged attention](paged-attention.md); for Gluon on gfx950 see the
[Gluon GEMM blog](../../sources/blogs/blog-gluon-gemm.md) and
[Triton on AMD](../languages/triton-amd.md).

## See also

- [MI450 TDM](../hardware/mi450-tdm.md)
- [MI450 workgroup clusters](../hardware/mi450-workgroup-cluster.md)
- [TDM tile widening](../techniques/tdm-tile-widening.md)
- [Four-stage decode pipelining](../techniques/attention-decode-pipelining.md)
- [Split-K / flash-decoding](../techniques/split-k.md)
- [MLA decode (gfx942/gfx950)](mla-decode.md)

## Sources

- [Attention Decode on AMD MI450 GPUs: A Gluon Kernel Optimization Guide](../../sources/blogs/blog-gluon-attention-decode-mi450.md)
- [Upstream Gluon example `mxfp_fa_gfx1250.py`](https://github.com/triton-lang/triton/blob/main/third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py)
- [Gluon documentation](https://triton-lang.org/main/gluon/index.html)
- [From Naive to Near-Peak: GEMM with Gluon](../../sources/blogs/blog-gluon-gemm.md)
