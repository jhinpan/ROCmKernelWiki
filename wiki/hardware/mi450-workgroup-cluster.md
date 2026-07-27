---
id: hw-mi450-workgroup-cluster
title: MI450 Workgroup Clusters — hardware-synchronized WGP groups (Triton CGA)
type: hardware
architectures:
- gfx1250
tags:
- workgroup-cluster
- wgp
- l2-cache
- lds
- split-k
- reduction
- gluon
confidence: source-reported
evidence_basis:
- source_id: blog-gluon-attention-decode-mi450
  evidence_type: benchmark
related:
- hw-mi450-tdm
- hw-chiplet-xcd
- technique-attention-decode-pipelining
- kernel-gluon-attention-decode-mi450
- technique-split-k
sources:
- blog-gluon-attention-decode-mi450
- technique-split-k
aliases:
- workgroup cluster
- CGA
- cooperative grid array
- cluster barrier
- multi-CTA
- cluster.arrive
---
# MI450 Workgroup Clusters — hardware-synchronized WGP groups (Triton CGA)

> **Scope.** MI450 / `gfx1250` is **outside** this wiki's active gfx942/gfx950
> publication scope; this page is retained as forward-looking research. All
> statements are `source-reported` from one vendor blog plus the upstream Gluon
> example it links, and **none has been checked on silicon by this project**.

## Overview

On CDNA, a workgroup is the largest unit with hardware-backed cooperation: it
runs on one CU, and its waves synchronize through `s_barrier` and share LDS.
Anything wider — cross-workgroup reduction, split-k merges — has to fall back to
**global memory plus a second kernel launch**, because there is no hardware
barrier spanning workgroups.

MI450 adds a middle tier. A **workgroup cluster** is a set of workgroups, each
still on its own WGP, that can:

- synchronize through **hardware cluster barriers**, and
- share data across WGPs through **multicast loads**,

with **L2 shared across the cluster**. That combination is what lets a
[split-k](../techniques/split-k.md) kernel fuse its reduction into the *same*
launch instead of round-tripping through global memory.

## Terminology mapping

Gluon inherits Triton's names, which do not match AMD's:

| Triton concept | MI450 concept |
|---|---|
| Warp | Wavefront |
| CTA | Workgroup |
| CGA | Workgroup cluster |

Two traps follow from this table:

1. **"Block size" changes meaning.** In ordinary Triton, a block size is
   per-workgroup. Under a multi-CTA layout it describes the **whole cluster**.
   Mis-reading this over-allocates by a factor of `NUM_CTAS`.
2. **CTAs are distributed through the layout system**, exactly like warps.
   Layouts such as `AMDWMMALayout` take a `cga_layout` field; you do not write a
   separate cluster-mapping pass.

## Expressing a cluster in Gluon

`cga_layout` is a list of basis vectors saying which tensor axis successive CTAs
advance along — the same linear-layout encoding used for `warp_bases`. The
upstream example derives it by doubling a stride until it covers `num_ctas`:

```python
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl
from triton.experimental.gluon.language.amd.gfx1250 import cluster

@gluon.constexpr_function
def make_cga_layout(rank, num_ctas, cta_axis=0):
    """Distribute `num_ctas` CTAs along one axis as linear-layout bases."""
    cga_layout, ctas = [], 1
    while ctas < num_ctas:
        base = [0] * rank
        base[cta_axis] = ctas
        cga_layout.append(base)
        ctas <<= 1
    return cga_layout

@gluon.jit
def split_k_reduce(partials_ptr, out_ptr):
    # ... each CTA writes its own partial tile to memory ...
    tdm.async_wait(0)      # my stores are visible
    cluster.arrive()       # signal this workgroup reached the barrier
    cluster.wait()         # block until every workgroup in the cluster arrived
    # ... now read back all partials and reduce, under a different layout ...
```

`cluster.arrive()` / `cluster.wait()` is a split barrier: arrive publishes, wait
blocks. The pairing matters — the partial stores must be drained
(`tdm.async_wait(0)`) *before* `arrive()`, or a peer may read stale data.

## Why this changes split-k attention

In baseline MQA/GQA decode, one workgroup owns one `(batch, kv_head)` pair and
walks that pair's KV sequence serially, so the grid is only `B * H_kv`
workgroups. With **256 WGPs** on MI450, a small batch or a single KV head leaves
most of the machine idle — the classic
[low-occupancy tail](../patterns/low-occupancy.md) that split-k exists to fix.

Split-k partitions the KV sequence into `S` pieces, raising the grid to
`B * H_kv * S`. Each program then computes a *partial* attention over its slice,
which in 3D form is:

```text
Q: [1, H_q / H_kv, D]     # shared by every partition
K: [S, T_kv / S, D]
V: [S, T_kv / S, D]
```

The cost is that the partials must be merged. Conventionally that is a second
reduction kernel communicating through global memory, which also supplies the
synchronization point for free. With a cluster, the merge becomes: write
partials, `cluster.arrive()` / `cluster.wait()`, read partials back and reduce —
all in one kernel. Because **L2 is shared across the cluster**, the partials do
not have to travel to global memory, cutting reduction overhead.

The reduction reuses the *same launch grid* but a **different layout**: for the
attention phase CTAs are distributed over KV partitions, and for the reduction
phase they are redistributed along the head dimension, looping over partitions.
This layout switch inside one kernel is the part that has no CDNA equivalent.

## Multi-CTA WMMA operands

Adding a CTA dimension to a WMMA layout does not replicate every operand. In the
split-k decode mapping, all partitions share the same Q, so under the multi-CTA
layout the **first operand is shared across partitions** while the **second
operand and the output differ per partition**. Wave distribution inside each
workgroup is unchanged. Sharing the Q operand is precisely what makes multicast
loads worthwhile.

## What is *not* established here

- **No ISA or programming-guide reference** for cluster size limits, barrier
  cost, multicast semantics, or how clusters interact with the memory partition
  modes. The corpus contains no MI450 hardware document.
- **The blog does not quantify** the reduction saving from L2-shared partials
  versus a global-memory round trip; it reports only that it "helps to cut the
  overhead".
- **Forward progress / deadlock rules are unstated.** Cluster barriers imply all
  member workgroups are co-resident, but the source gives no guarantee to rely on.
- **Not a CDNA feature.** gfx942/gfx950 have no cluster tier; on those targets
  split-k reduction remains a separate kernel. Do not present cluster-fused
  reduction as portable AMD advice.

## See also

- [MI450 TDM](mi450-tdm.md)
- [Split-K / flash-decoding](../techniques/split-k.md)
- [Attention-decode pipelining](../techniques/attention-decode-pipelining.md)
- [Gluon attention decode kernel](../kernels/gluon-attention-decode-mi450.md)
- [Chiplet/XCD locality on MI300 (the CDNA scaling story)](chiplet-xcd.md)

## Sources

- [Attention Decode on AMD MI450 GPUs: A Gluon Kernel Optimization Guide](../../sources/blogs/blog-gluon-attention-decode-mi450.md)
- [Upstream Gluon example `mxfp_fa_gfx1250.py`](https://github.com/triton-lang/triton/blob/main/third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py)
- [Split-K technique](../techniques/split-k.md)
