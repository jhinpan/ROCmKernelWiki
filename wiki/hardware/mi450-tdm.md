---
id: hw-mi450-tdm
title: TDM — MI450 Tensor Data Mover (descriptor-based global→LDS DMA)
type: hardware
architectures:
- gfx1250
tags:
- tdm
- wgp
- lds
- async-copy
- direct-to-lds
- lds-bypass
- global-instructions
- memory-bound
confidence: source-reported
evidence_basis:
- source_id: blog-gluon-attention-decode-mi450
  evidence_type: benchmark
related:
- hw-mi450-workgroup-cluster
- hw-async-copy-lds
- hw-lds
- hw-s-waitcnt
- technique-tdm-tile-widening
- technique-attention-decode-pipelining
- kernel-gluon-attention-decode-mi450
sources:
- blog-gluon-attention-decode-mi450
- hw-async-copy-lds
aliases:
- TDM
- Tensor Data Mover
- tensor_load_to_lds
- tdm.async_load
- descriptor-based tensor load
---
# TDM — MI450 Tensor Data Mover (descriptor-based global→LDS DMA)

> **Scope.** MI450 / `gfx1250` is **outside** this wiki's active gfx942/gfx950
> publication scope, so this page is retained as forward-looking research only.
> Everything here is `source-reported` from a single vendor blog plus the
> upstream Gluon example it links; **none of it has been checked on silicon by
> this project**, unlike the gfx950 claims in [`VERIFICATION.md`](../../VERIFICATION.md).

## Overview

**TDM** is a dedicated data-movement unit on the MI450 series that copies
*structured tensor tiles* between global memory and LDS. Instead of the kernel
computing per-lane addresses and issuing many small vector loads, it builds a
**descriptor** — base address, shape, strides, and target layout — and issues one
asynchronous bulk transfer.

This is a different abstraction level from the CDNA3/CDNA4 answer to the same
problem. On gfx942/gfx950 the async global→LDS path is
[direct-to-LDS](async-copy-lds.md): still a *per-lane vector load*
(`buffer_load…lds` / `global_load_lds`), just one that skips the register file
and is capped at 4 B (gfx942) or 16 B (gfx950) per lane. TDM instead moves a
**tile described once**, which is why the reported MI450 load granularity is
"descriptor-based tensor tiles" rather than "32/96/128-bit vector loads".

| | MI350 series (gfx950) | MI450 series |
|---|---|---|
| Memory instruction | `Global`/`Buffer` load to LDS | TDM load to LDS |
| Load granularity | 32/96/128-bit vector loads | Descriptor-based tensor tiles |
| Memory units per WGP | 1 | 2 |
| Max LDS per WGP | 160 KB | 320 KB |
| VGPRs per SIMD | 512 | 1024 |

A **WGP** (Workgroup Processor) is the unit that runs a workgroup — the source
calls it the part "named CU in earlier MI-series" — and contains four SIMD32
units of 32 lanes each. Each SIMD32 owns its VGPRs (used by the VALU and WMMA);
all four share the WGP's LDS.

## The descriptor model

In Gluon the TDM surface lives in a target-specific module. A descriptor binds a
global tensor's geometry to the shared-memory layout that will receive it, and
the shared allocation carries an extra leading dimension for pipeline slots:

```python
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl
from triton.experimental.gluon.language.amd.gfx1250 import tdm

@gluon.jit
def stage_kv_tile(k_base, shape, block_shape, smem_layout, num_slots: ttgl.constexpr):
    # Describe the global tensor once: base, logical shape, strides, tile shape.
    desc = tdm.make_tensor_descriptor(
        base=k_base,
        shape=[shape[0], shape[1]],
        strides=[shape[1], 1],            # row-major
        block_shape=[block_shape[0], block_shape[1]],
        layout=smem_layout,
    )
    # One LDS allocation, `num_slots` pipeline buffers deep.
    smem = ttgl.allocate_shared_memory(
        k_base.dtype.element_ty, [num_slots] + block_shape, smem_layout
    )

    # Issue a bulk async tile copy into slot 0 at tile offset (off_m, off_n).
    tdm.async_load(desc, [0, 0], smem.index(0), 1)   # last arg is a predicate
    tdm.async_wait(0)                                 # drain outstanding TDM loads
    return smem.index(0).load(ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0], []))
```

Three properties matter for scheduling:

- **Asynchronous.** `tdm.async_load` returns immediately;
  `tdm.async_wait(count)` blocks until at most `count` transfers remain
  outstanding. This is a *count-of-remaining* wait, the same shape of contract as
  [`s_waitcnt vmcnt(N)`](s-waitcnt.md) on CDNA, so partial drains let the kernel
  keep several tiles in flight.
- **Descriptor-shaped, not lane-shaped.** The transfer geometry comes from
  `block_shape`, so the *innermost dimension in bytes* — not a per-lane vector
  width — is the knob that decides how efficiently the copy runs.
- **Store direction too.** `tdm.async_store` writes an LDS tile back out, which
  is how split-k partials reach memory before a cluster barrier (see
  [workgroup clusters](mi450-workgroup-cluster.md)).

## Direct path vs cache path

MI450 has a **per-WGP cache at the same level as LDS** that is not directly
programmer-visible. TDM can either write **directly into LDS** or route the data
**through that cache** first. Because the cache is small, the source's guidance
for memory-bound work is to bypass it and land straight in LDS.

Which path a transfer takes is not a flag — it follows from the request shape:

> The innermost dimension needs to be **at least 128 bytes** to use the direct
> path, and **256 bytes is recommended** to keep more in-flight memory traffic.

That single rule is the origin of the whole KV-reshape trick: for an FP8 tensor
whose innermost dimension is the head size `D`, 256 bytes means `D = 256`, but
`D` is fixed by the model. The kernel therefore reshapes to manufacture a wider
innermost dimension — see
[TDM tile widening](../techniques/tdm-tile-widening.md).

## Consequences for LDS layout

The upstream example pads the shared layout to spread a tile across banks,
noting that with 8-bit elements **256 elements cover 64 banks**, so the padding
interval is chosen to be at least 256 elements (and at least the inner
dimension):

```python
ttgl.PaddedSharedLayout.with_identity_for(
    interval_padding_pairs=[[256, 16]],   # pad 16 elements every 256
    shape=block_shape,
    order=[1, 0],
    cga_layout=[],
)
```

The consumer side still uses ordinary LDS reads — `ds_load_b128` for K, and the
transposing `ds_load_tr8_b64` for V — so **TDM removes the global→LDS addressing
cost, not LDS bank arbitration**. The same discipline as
[LDS swizzling](../techniques/lds-swizzling.md) on CDNA applies.

## What is *not* established here

Being explicit, because this page is out of scope and unverified:

- **No ISA reference.** The corpus has no MI450 ISA document. `tensor_load_to_lds`
  is named in the source's instruction table; its encoding, cache-policy bits,
  and alignment rules are not covered.
- **TDM latency is an estimate.** The ~1000-cycle figure used in the pipeline
  analysis is called out by the source as "an empirical number to help us reason
  about the pipeline", varying with tile shape and access pattern.
- **Wavefront width is not stated** by the blog. The upstream kernel's softmax
  layout uses `threads_per_warp=[1, 16, 2]`, i.e. 32 lanes per warp, which
  implies wave32 — but that is an inference from one layout helper, not a
  documented hardware fact. Do **not** carry the CDNA
  [wave64](wavefront.md) assumption onto this target.
- **No portability claim.** TDM does not exist on gfx942/gfx950; there is no
  source-compatible fallback. Kernels using it are target-specific.

## See also

- [Direct-to-LDS async copy (gfx942/gfx950 analog)](async-copy-lds.md)
- [MI450 workgroup clusters](mi450-workgroup-cluster.md)
- [TDM tile widening](../techniques/tdm-tile-widening.md)
- [Attention-decode pipelining](../techniques/attention-decode-pipelining.md)
- [Gluon attention decode kernel](../kernels/gluon-attention-decode-mi450.md)
- [Local Data Share (LDS)](lds.md)
- [s_waitcnt counter semantics](s-waitcnt.md)

## Sources

- [Attention Decode on AMD MI450 GPUs: A Gluon Kernel Optimization Guide](../../sources/blogs/blog-gluon-attention-decode-mi450.md)
- [Upstream Gluon example `mxfp_fa_gfx1250.py`](https://github.com/triton-lang/triton/blob/main/third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py)
- [Direct-to-LDS async copy](async-copy-lds.md)
