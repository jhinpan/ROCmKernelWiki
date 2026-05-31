# Paged Attention Decode — portable HIP reference (fp32)

A minimal, runnable **paged-attention decode** kernel: a single query step
(`q_len = 1`) attends over a **paged KV cache** addressed through a per-sequence
**block table** (page-table indirection), exactly the structure used by
vLLM / AITER decode. Online-softmax (FlashAttention) recurrence streams the
gathered KV blocks. GQA is included (`GROUP` query heads share one KV head).

This is **PORTABLE pure-HIP** — FMA math, LDS tree reduction, `__syncthreads()`.
It **builds and runs on gfx1201 (RDNA4)** and self-checks every output element
against a CPU reference. No MFMA/WMMA needed (decode is bandwidth-bound, not a
big GEMM).

## What it demonstrates

- **Block-table indirection**: logical KV blocks are scattered to non-contiguous
  physical blocks (`block_table` is shuffled), so the kernel must gather pages —
  the defining feature of paged attention.
- **Ragged sequences / partial tail blocks**: `seq_lens = {40, 17, 64}` exercise
  partial last blocks (40 → 2 full + 1 half block, etc.).
- **GQA**: `num_q_heads = num_kv_heads * GROUP` (8 = 2 × 4).
- **Online softmax**: numerically-stable running max / sumexp / accumulator.

Layout mirrors the wiki page:

```
q           : [num_seqs, num_q_heads, HEAD_DIM]
k_cache     : [num_blocks, num_kv_heads, BLOCK_SIZE, HEAD_DIM]
v_cache     : [num_blocks, num_kv_heads, BLOCK_SIZE, HEAD_DIM]
block_table : [num_seqs, max_num_blocks]   (logical block -> physical block)
seq_lens    : [num_seqs]
out         : [num_seqs, num_q_heads, HEAD_DIM]
```

Kernel mapping: one workgroup per `(sequence, query-head)`, `HEAD_DIM` threads
per block (one thread per head-dim element).

## Build & run

```bash
./build.sh
# or directly:
hipcc --offload-arch=gfx1201 -O3 paged_attention.cpp -o paged_attention && ./paged_attention
```

## Expected output (captured on this gfx1201 box, ROCm 7.2.3)

```
build: OK
paged-attention decode (fp32, portable HIP, gfx1201)
  num_seqs=3  num_q_heads=8  num_kv_heads=2  GROUP=4
  HEAD_DIM=64  BLOCK_SIZE=16  seq_lens={40,17,64}
  kernel time: 0.0471 ms/iter (avg of 200)
  max abs error vs CPU: 8.941e-08
PASS
```

`max abs error ~9e-8` (fp32 rounding) and `PASS`. Timing is for the tiny demo
problem on a Radeon RX 9070 XT; it is illustrative, not a tuned benchmark.

## Arch

- **Runs on:** gfx1201 (RDNA4, this box). Portable — also runs on any other ROCm
  GPU (gfx942/gfx950) since it uses only generic HIP.
- This is a *reference* for clarity. Production AITER/vLLM kernels add: FP8 KV
  cache, 128-bit vectorized KV loads, flash-decoding chunk-parallel split with a
  reduction pass, and MFMA/WMMA tiling for large GQA groups.
