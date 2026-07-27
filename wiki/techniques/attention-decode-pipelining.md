---
id: technique-attention-decode-pipelining
title: Four-Stage Decode Pipelining — interleaving TDM, WMMA, and online softmax
type: technique
architectures:
- gfx1250
tags:
- software-pipelining
- async-pipeline
- lds-double-buffering
- lds-triple-buffering
- tdm
- wmma
- attention
- decode
- latency-bound
- memory-bound
- gluon
confidence: source-reported
reproducibility: snippet
hardware_features:
- tdm
- wmma
- lds
techniques:
- software-pipelining
- lds-double-buffering
- lds-triple-buffering
kernel_types:
- attention
- decode
- flash-attention
- softmax
languages:
- gluon
- python
evidence_basis:
- source_id: blog-gluon-attention-decode-mi450
  evidence_type: benchmark
related:
- hw-mi450-tdm
- technique-tdm-tile-widening
- kernel-gluon-attention-decode-mi450
- technique-lds-double-buffering
- technique-mfma-pipelining
sources:
- blog-gluon-attention-decode-mi450
- hw-mi450-tdm
- technique-lds-double-buffering
---
# Four-Stage Decode Pipelining — interleaving TDM, WMMA, and online softmax

> **Scope.** MI450 / `gfx1250` is **outside** this wiki's active gfx942/gfx950
> publication scope; this page is retained as forward-looking research and is
> `source-reported` from one vendor blog plus the upstream Gluon example. The
> cycle model below is the **blog's own simplified estimate**, not a measurement,
> and nothing here was checked on silicon by this project.

## Why two stages is not enough

The obvious pipeline for a flash-attention decode loop separates memory from
compute: one stage issues [TDM](../hardware/mi450-tdm.md) loads for future
iterations, the other consumes tiles already in LDS. With double buffering, one
LDS slot is being read while the other is being filled, and the two swap each
step. This is the standard
[LDS double-buffering](lds-double-buffering.md) shape.

It overlaps TDM with compute, but it treats **all** compute as one monolithic
block. Inside that block, LDS reads, the QK matrix multiply, the softmax VALU
work, and the PV matrix multiply end up effectively serialized — even though they
use *different hardware units*. The block inflates, and a kernel that should be
limited by memory bandwidth can become limited by serialized compute instead.

## The dependency structure that permits more overlap

Writing one loop iteration `i` of tiled attention and asking what actually
depends on what:

```text
TDM K[i] ──> LDS K[i] ──> QK[i] ──> softmax[i] ──> PV[i] ──> O[i]
                                        ↑              ↑
TDM V[i] ─────────────> LDS V[i] ───────┘──────────────┘
```

Two observations unlock the schedule:

1. **`QK[i+1]` does not depend on `softmax[i]`.** The next KV tile's score
   matrix can be prefetched and computed *early* — it only needs Q (loop
   invariant) and `K[i+1]`. Only the running max/sum update is serial.
2. **QK and PV use WMMA; online softmax uses only VALU.** Different issue ports,
   so they can interleave rather than serialize. `v_exp_f32` additionally goes to
   the transcendental unit and can overlap with other VALU work.

So instead of "memory stage, compute stage", a stage may hold `QK` from iteration
`i+1`, softmax work from iteration `i`, an LDS read for `i`, and a TDM request for
a *later* iteration — all at once.

## The four-stage schedule

The softmax work is split into two groups of roughly equal cost so the hardware
can interleave WMMA against VALU evenly:

| Group | Contents |
|---|---|
| `VEC0` | `MAX` + `FMA` + `EXP` |
| `VEC1` | `SUM` + `MUL` + `CVT` |

then paired so that **`QK` interleaves with `VEC0`** and **`PV` interleaves with
`VEC1`**. Gluon expresses the grouping with an explicit stage scope, alternating
compute and memory stages; the upstream kernel's mainloop reduces to:

```python
from triton.experimental import gluon
from triton.experimental.gluon.language.amd import warp_pipeline_stage

@gluon.jit
def mainloop_body(self, i, a, b, pred, q, q_scale, k0, k1, k_scale, ...):
    # a / b are the double-buffered LDS slot indices (b = 1 - a)
    with warp_pipeline_stage("compute0"):
        qk0 = self.compute_qk(q, q_scale, k0, k_scale, zero)   # iter i+1
        p1 = ttgl.exp2(qk1_shifted)                            # iter i   (VEC0)
        acc0 = acc0 * expand_dims(alpha, -1)

    self.async_wait(4)                       # keep 4 TDM loads outstanding
    with warp_pipeline_stage("memory0"):
        k1 = self.shared_load_k(sub_idx=1, buf=b)              # iter i+1
        self.issue_global_load_v(i + 1, sub_idx=0, buf=b)      # iter i+1

    with warp_pipeline_stage("compute1"):
        qk1 = self.compute_qk(q, q_scale, k1, k_scale, zero)   # iter i+1
        l_ij = ttgl.sum(self.concat_subtile(p0, p1), -1)       # iter i   (VEC1)
        p, p_scale = self.downcast_p(p)                        #          (CVT)

    self.async_wait(4)
    with warp_pipeline_stage("memory1"):
        v0 = self.shared_load_v(sub_idx=0, buf=a)              # iter i
        self.issue_global_load_v(i + 1, sub_idx=1, buf=b)      # iter i+1

    with warp_pipeline_stage("compute2"):
        acc0 = self.compute_pv(p, p_scale, v0, v_scale, acc0)  # iter i
        m_ij = maximum(m_i, max(self.concat_subtile(qk0, qk1), -1))   # VEC0

    self.async_wait(4)
    with warp_pipeline_stage("memory2"):
        v1 = self.shared_load_v(sub_idx=1, buf=a)              # iter i
        self.issue_global_load_k(i + 3, sub_idx=0, buf=b, pred=pred)  # iter i+3
```

Note the lookahead distances in the comments: LDS reads serve `i`/`i+1`/`i+2`
while the TDM requests are for **`i + 3`**. The `async_wait(4)` calls are partial
drains — "let at most 4 transfers remain outstanding" — not full barriers.

## The cycle model, and why double buffering is not enough

The blog's estimate for one FP8 iteration at `BLOCK_M = 32`, `BLOCK_N = 128`,
`D = 128`, two waves, under two interleave rules (one WMMA = 8 cycles and hides
2 cycles of VALU; one `EXP` = 2 cycles and hides 1 cycle of non-`EXP` VALU):

| Group | Cycles |
|---|---|
| QK | 64 |
| PV | 64 |
| `VEC0` = MAX + FMA + EXP | 192 |
| `VEC1` = SUM + MUL + CVT | 96 |
| QK + `VEC0`, interleaved | 176 |
| PV + `VEC1`, interleaved | 144 |
| **Total, interleaved** | **320** |

Now compare against a TDM latency of roughly **1000 cycles** (the blog's
explicitly empirical placeholder):

| Buffering | LDS slots | Lookahead | Compute overlapped | Exposed wait |
|---|---|---|---|---|
| Double | 2 | ~1 iteration | ~320 cycles | ~680 cycles |
| Triple | 3 | ~2 iterations | ~640 cycles | ~360 cycles |

With only two slots, a TDM request gets one iteration — about 320 cycles — of work
to hide behind, leaving roughly 680 cycles exposed. Interleaving the compute
block was necessary but insufficient: **the loop is still latency-bound, and no
amount of intra-iteration reordering fixes it**, because the total compute per
iteration is simply smaller than the memory latency.

The fix is a longer lookahead. **Triple buffering** adds a third LDS slot, so at
any moment one buffer is being consumed, one is ready or nearly ready, and one is
being filled. Requests can be issued two iterations ahead instead of waiting for
the next swap, roughly doubling the overlapped work to ~640 cycles and halving
the exposed wait to ~360.

Triple buffering still does not fully cover the latency — it just gives the
hardware more outstanding memory work. That residual gap is why the reported
result is 85% of peak rather than closer to 100%, and why the blog lists memory
prefetching and better instruction scheduling as remaining work.

## Choosing the buffer count

More slots cost LDS, and LDS caps occupancy. The upstream example selects
`NUM_BUFFERS ∈ {1, 2, 3}` from the tile footprint — roughly
`2 * BLOCK_N * HEAD_SZ * NUM_BUFFERS / KV_PACK_DIV` bytes for K and V together —
falling back to 2 or 1 when three slots will not fit. The 320 KB LDS per WGP is
what makes three slots practical at decode tile sizes at all; the same schedule
on a 160 KB budget would force a shallower pipeline.

The general rule this instantiates: **pick the lookahead distance from the ratio
of memory latency to per-iteration compute, then check it fits in LDS** — not the
other way around.

## Portability

The *reasoning* transfers to gfx942/gfx950 — see
[MFMA pipelining](mfma-pipelining.md) and
[LDS double buffering](lds-double-buffering.md), where the analogous knob is
`num_stages` over [direct-to-LDS](../hardware/async-copy-lds.md) copies gated by
`s_waitcnt vmcnt(N)`. The *mechanism* does not: `warp_pipeline_stage`,
`tdm.async_wait`, and the 320 KB LDS budget are MI450-specific, and the CDNA
cycle costs differ (MFMA rather than WMMA shapes). Re-derive the cycle table per
target instead of copying these numbers.

## Caveats

- **The cycle table is an estimate**, built from per-instruction costs and two
  hand-applied interleave rules — not from a profile. Treat it as a way to reason
  about lookahead distance, not as a performance prediction.
- **1000 cycles is a placeholder.** Real TDM latency depends on tile shape and
  access pattern.
- **Register pressure is unaddressed.** Holding `qk0/qk1`, `p0/p1`, and two
  accumulators across four stages is what the doubled 1024 VGPRs/SIMD budget
  buys; the blog does not report the achieved occupancy.

## See also

- [MI450 TDM](../hardware/mi450-tdm.md)
- [TDM tile widening](tdm-tile-widening.md)
- [Gluon attention decode kernel](../kernels/gluon-attention-decode-mi450.md)
- [LDS double buffering (gfx942/gfx950)](lds-double-buffering.md)
- [MFMA pipelining (gfx942/gfx950)](mfma-pipelining.md)

## Sources

- [Attention Decode on AMD MI450 GPUs: A Gluon Kernel Optimization Guide](../../sources/blogs/blog-gluon-attention-decode-mi450.md)
- [Upstream Gluon example `mxfp_fa_gfx1250.py`](https://github.com/triton-lang/triton/blob/main/third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py)
- [LDS double buffering](lds-double-buffering.md)
