# FlashAttention-2 forward — portable HIP reference

A pure-HIP, fp32 implementation of the **FlashAttention-2 forward** online-softmax
recurrence. It is the *demonstrable* companion to the
[CK-tile FlashAttention page](../../wiki/kernels/flash-attention-ck.md): the
production AMD path issues the two GEMMs (`Q·Kᵀ` and `P·V`) as `v_mfma_*`
matrix-core instructions through `ck_tile`'s `BlockFmhaPipelineQRKSVS`, which
**does not execute on RDNA4 (gfx1201)**. This example reproduces the same
algorithm with scalar FMA math so it actually runs on this box and self-checks.

## What it shows

- The FlashAttention-2 recurrence: tile over KV blocks, maintain running
  `(m, l, O)` state in registers, rescale the accumulator by
  `alpha = exp(m_old - m_new)` each step, and **hoist the `1/l` division out of
  the inner loop** to the very end (the key FA-2 change over v1).
- Work partitioning: **one thread block per `(head, query-tile)`**, one query row
  per thread; K/V tiles streamed through LDS (`__shared__`) — mirroring the
  Q-resident / KV-streamed `QRKSVS` schedule.
- `O = softmax(scale · Q·Kᵀ) · V`, non-causal, verified against a naive CPU
  `softmax(QK^T)V`.

This is **not** an MFMA kernel — the scores and `P·V` are computed with scalar
FMAs, so it is correct but not fast. It exists to make the algorithm runnable and
checkable; for performance see the CK-tile numbers on the wiki page.

## Build & run

```bash
./build.sh
# or:
hipcc --offload-arch=gfx1201 -O3 flash_attention_fwd.hip -o flash_attention_fwd
./flash_attention_fwd            # defaults: H=4 N=256 D=64
./flash_attention_fwd 8 512 128  # H N D  (D <= 128)
```

Runs natively on **gfx1201 (RDNA4)**. Pure HIP — no rocWMMA, no MFMA — so it is
portable to any ROCm GPU (and the CDNA matrix path is the production CK-tile
kernel, not this reference).

## Expected output

Real output captured on this gfx1201 (RX 9070 XT), ROCm 7.2.3:

```
=== build OK, running ===
FlashAttention-2 fwd (portable HIP, fp32) on gfx1201
  H=4 N=256 D=64  BR=64 BC=64  scale=0.12500
  avg kernel time: 2.1091 ms   (31.8 GFLOP/s)
  max abs error: 1.490e-07   max rel error: 4.404e-03
PASS
```

`max abs error ~1.5e-7` confirms the streaming online-softmax matches the
full-matrix CPU softmax to fp32 precision. (The GFLOP/s figure is for this
scalar reference and is not a performance claim; the wiki page reports the
matrix-core CK-tile throughput.)
