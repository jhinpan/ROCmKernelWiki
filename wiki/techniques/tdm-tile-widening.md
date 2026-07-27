---
id: technique-tdm-tile-widening
title: TDM Tile Widening — reshaping K/V so the innermost dimension hits 256 B
type: technique
architectures:
- gfx1250
tags:
- tdm-tile-widening
- tdm
- direct-to-lds
- lds-bypass
- swizzled-layout
- preshuffle-layout
- data-reuse
- kv-cache
- fp8
- memory-bound
- gluon
confidence: source-reported
reproducibility: snippet
hardware_features:
- tdm
- lds
- fp8
kernel_types:
- attention
- decode
- kv-cache
languages:
- gluon
- python
evidence_basis:
- source_id: blog-gluon-attention-decode-mi450
  evidence_type: benchmark
related:
- hw-mi450-tdm
- technique-attention-decode-pipelining
- kernel-gluon-attention-decode-mi450
- technique-preshuffle-layout
- technique-vectorized-loads
sources:
- blog-gluon-attention-decode-mi450
- hw-mi450-tdm
---
# TDM Tile Widening — reshaping K/V so the innermost dimension hits 256 B

> **Scope.** MI450 / `gfx1250` is **outside** this wiki's active gfx942/gfx950
> publication scope; this page is retained as forward-looking research and is
> `source-reported` from one vendor blog plus the upstream Gluon example. It has
> **not** been checked on silicon by this project.

## The problem

On MI450, whether a [TDM](../hardware/mi450-tdm.md) transfer lands **directly in
LDS** or detours through the small per-WGP cache is decided by the *shape* of the
request, not by a flag:

- innermost dimension **< 128 B** → cache path,
- **≥ 128 B** → direct path,
- **256 B** → recommended, because it keeps more memory traffic in flight.

For attention decode the K and V tiles are `(BLOCK_N, D)`, so the innermost
dimension is the head dimension `D`. In FP8 (one byte per element) that means
**`D` would have to be 256 bytes = 256 elements** to hit the recommended shape.
But `D` is fixed by the model architecture — typically 128 — and a kernel cannot
change it.

A memory-bound decode kernel therefore starts out on the *worse* path for the
exact tensors that dominate its bandwidth.

## The technique

`D` is fixed, but the *tile geometry TDM sees* is not. Because K and V are
contiguous in memory, a `[…, dim_outer, dim_inner]` tensor can be **reinterpreted**
as `[…, dim_outer * dim_inner / 256, 256]` without moving any bytes in the common
case: same elements, same order, wider innermost dimension. TDM then transfers
256-byte rows and takes the direct path; the kernel restores the logical
`(BLOCK_N, D)` view with a `reshape` when it reads the tile back out of LDS.

So the data flow becomes:

```text
global memory        TDM              LDS                 registers
[.., N*D/256, 256] ------> [.., 256] wide tile -----> reshape -> (BLOCK_N, D)
   (widened view)          direct path, 256 B/row      (logical view restored)
```

The widening happens **on the host** (a view/reshape of the KV tensor before
launch) and is undone **inside the kernel** (a reshape of the LDS buffer). The
kernel's math never sees the widened shape.

## Implementation

The upstream Gluon example calls the host side `preshuffle` and the in-kernel
side `unshuffle`, with the docstring stating the goal directly: *"To get better
performance from TDM, we need to make sure the inner-most dim of the target block
is 256B."*

```python
import math
import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl

ELEMS = 256   # 256 x 8-bit = 256 B innermost dimension

def preshuffle(x: torch.Tensor) -> torch.Tensor:
    """Host side: widen the innermost dim to 256 B. FP8/8-bit elements only."""
    assert x.element_size() == 1, "widening assumes 8-bit elements"
    *prefix, dim_outer, dim_inner = x.shape
    return x.contiguous().reshape(*prefix, dim_outer * dim_inner // ELEMS, ELEMS)

@gluon.constexpr_function
def widened_shape(shape):
    """Compile-time counterpart, for building the tensor descriptor."""
    *prefix, dim_inner = shape
    return [*prefix[:-1], prefix[-1] * dim_inner // ELEMS, ELEMS]

@gluon.jit
def unshuffle(buffer, block_shape: ttgl.constexpr):
    """In-kernel: restore the logical (BLOCK_N, D) tile view from LDS."""
    return buffer.reshape(block_shape)
```

The descriptor is then built from `widened_shape(...)` rather than the model
shape, so `block_shape[-1] == 256` and every transfer qualifies for the direct
path.

## When subtiling forces a real permute

The cheap case above is a pure view. It stops being free when the kernel also
**subtiles the inner dimension** — splitting each tile in half along `D` so the
pipeline can consume half-tiles (this is what makes `compute0…compute3` in the
four-stage schedule possible; see
[attention-decode pipelining](attention-decode-pipelining.md)).

Reinterpreting rows would then interleave elements from the two subtiles, so the
subtiles must be **permuted into place first**, which is a genuine data movement:

```python
def preshuffle_subtiled_inner(x: torch.Tensor, block_dim_outer: int) -> torch.Tensor:
    """Inner-dim subtiling: reorder subtiles before widening (real copy)."""
    *prefix, dim_outer, dim_inner = x.shape
    batch = math.prod(prefix)
    x = x.reshape(batch, dim_outer, dim_inner)
    # split inner dim into 2 subtiles, hoist the subtile axis above the rows
    x = x.view(batch, dim_outer // block_dim_outer, block_dim_outer, 2, dim_inner // 2)
    x = x.permute(0, 1, 3, 2, 4).contiguous()       # <-- materializes a copy
    return x.reshape(*prefix, dim_outer * dim_inner // ELEMS, ELEMS)
```

Subtiling the **outer** dimension needs no permute — it behaves like the plain
case. So the ordering rule is: *outer-dim subtiling is free; inner-dim subtiling
costs one host-side permute.* Because a decode KV cache is written once per token
and read on every step, paying that permute once at cache-population time is
usually the right trade — but it is a real cost, and on a paged KV cache that is
appended to continuously it may not be affordable at all.

## Relationship to CDNA practice

The mechanism differs from, but the instinct is the same as,
[preshuffled weight layouts](preshuffle-layout.md) on gfx942/gfx950: rearrange
the operand offline so the hot loop's memory instructions hit their widest,
most-aligned form. The difference is *what* you are widening for —

- on gfx942/gfx950 you widen toward the **per-lane vector width**, e.g. reaching
  `global_load_dwordx4` / 16-byte [direct-to-LDS](../hardware/async-copy-lds.md)
  (see [vectorized loads](vectorized-loads.md));
- on MI450 you widen toward a **descriptor's innermost byte count**, because
  that is what selects TDM's direct path.

There is no equivalent 128 B/256 B threshold on CDNA, and this reshape buys
nothing there. Do not port it as a general AMD optimization.

## Caveats

- **8-bit elements assumed.** The upstream helper asserts a 1-byte element type.
  For 16-bit data 256 B is 128 elements, so the divisor changes and the arithmetic
  must be redone; the example does not cover it.
- **Requires contiguity.** The free path is a `reshape` on a contiguous tensor.
  A strided or non-contiguous KV view forces a copy.
- **Divisibility.** `dim_outer * dim_inner` must be a multiple of 256.
- **The 256 B figure is a recommendation, not a threshold.** 128 B already
  reaches the direct path; 256 B is reported to help keep more traffic in flight.
  The blog does not publish the delta between them.

## See also

- [MI450 TDM](../hardware/mi450-tdm.md)
- [Attention-decode pipelining](attention-decode-pipelining.md)
- [Gluon attention decode kernel](../kernels/gluon-attention-decode-mi450.md)
- [Preshuffled layouts (gfx942/gfx950)](preshuffle-layout.md)
- [Vectorized loads](vectorized-loads.md)

## Sources

- [Attention Decode on AMD MI450 GPUs: A Gluon Kernel Optimization Guide](../../sources/blogs/blog-gluon-attention-decode-mi450.md)
- [Upstream Gluon example `mxfp_fa_gfx1250.py`](https://github.com/triton-lang/triton/blob/main/third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py)
- [MI450 TDM](../hardware/mi450-tdm.md)
