---
id: kernel-paged-attention
title: Paged Attention Decode (vLLM / AITER) on MI300
type: kernel
architectures:
- gfx942
- gfx950
tags:
- paged-attention
- attention
- decode
- kv-cache
- memory-bound
- mfma
- triton
- hip
confidence: source-reported
reproducibility: runnable
artifact_dir: examples/paged-attention
kernel_types:
- paged-attention
- attention
- decode
languages:
- hip
- triton
related:
- kernel-flash-attention-ck
- kernel-mla-decode
- hw-mfma
- hw-chiplet-xcd
- lang-triton-amd
- technique-kernel-fusion
sources:
- ref-aiter
- blog-flash-attention-amd
- doc-flash-attention-2
- doc-mi300x-datasheet
- blog-cktile-flash
performance_claims:
- gpu: MI300X
  dtype: fp16
  metric: hbm-bandwidth-utilization
  value: ~80% of 5.3 TB/s peak at large batch decode
  source_id: doc-mi300x-datasheet
  confidence: inferred
- gpu: MI300X
  dtype: fp8
  metric: kv-cache-read-volume
  value: ~2x lower than fp16 KV cache (1 byte/elem vs 2)
  source_id: ref-aiter
  confidence: source-reported
- gpu: MI300X
  dtype: bf16
  metric: decode-throughput
  value: AITER paged-attn outperforms the reference vLLM HIP kernel on long-context
    decode
  source_id: blog-flash-attention-amd
  confidence: source-reported
implemented_by:
- pr-aiter-1383
- pr-triton-718
- pr-aiter-376
- pr-aiter-159
- pr-sglang-23620
- pr-composable_kernel-1789
- pr-vllm-43817
- pr-vllm-38502
---
# Paged Attention Decode (vLLM / AITER) on MI300

## Overview

**Paged attention** is the attention kernel that backs autoregressive *decode*
in LLM serving engines (vLLM, SGLang). During decode each step appends a single
new token, so the query length is `1` while the KV cache grows to thousands of
tokens. Instead of storing each sequence's KV cache as one contiguous tensor,
the cache is split into fixed-size **blocks** (pages) of `BLOCK_SIZE` tokens
(typically 16 or 32) that are scattered across HBM. A per-sequence **block
table** maps logical token positions to physical block ids — this is what lets
the serving engine pack many variable-length sequences into memory without
fragmentation.

The kernel is therefore a *gather + flash-attention* fused operation:

1. For each (sequence, KV head) read the block table.
2. Stream the K/V blocks it points to from HBM.
3. Run the online-softmax FlashAttention recurrence against the single query row.
4. Write one output row per query head.

Decode paged attention is almost always **HBM-bandwidth bound**, not compute
bound: with `q_len = 1` there is no large GEMM to fill the
[matrix cores](../hardware/mfma.md); the cost is dominated by streaming the KV
cache. On [MI300X](../../sources/docs/doc-mi300x-datasheet.md) that means the
ceiling is the 5.3 TB/s HBM3 bandwidth, and the optimization goal is to keep the
KV read stream saturated.

## Why GQA changes the arithmetic intensity

Modern models use **grouped-query attention (GQA)**: many query heads share one
KV head (e.g. 8 query heads → 1 KV head). For a *single* query head, decode reads
the whole KV cache to produce one output row — arithmetic intensity is hopeless.
But because all query heads in a group read the *same* K/V blocks, the kernel can
load each KV block once and reuse it across the whole group of query rows. That
reuse turns the per-group query into a tall-skinny `[GROUP, head_dim] x
[head_dim, kv_len]` product that actually maps onto an
[MFMA](../hardware/mfma.md) tile (e.g. `16x16x16` for FP16), recovering useful
matrix-core utilization while reading KV from HBM only once.

## AITER / vLLM kernel structure

[AITER](../../sources/refs/ref-aiter.md) is AMD's inference operator library and
the default attention backend for vLLM on ROCm. It ships paged attention in
multiple backends — a **Triton** kernel (portable, `tl.dot`→MFMA) and tuned
**CK / ASM** kernels for gfx942/gfx950. The common decode strategy is
**chunk-parallel** (the vLLM "v2"/"flash-decoding" split): the KV length is split
into chunks processed by separate workgroups, each producing a partial output
plus its running `max` and `sumexp`; a small second pass rescales and reduces the
partials. This exposes parallelism when batch × heads alone cannot fill all
[304 CUs](../hardware/chiplet-xcd.md).

```python
import triton
import triton.language as tl

@triton.jit
def paged_attn_decode_kernel(
    q_ptr,            # [num_seqs, num_q_heads, HEAD_DIM]
    k_cache_ptr,      # [num_blocks, num_kv_heads, BLOCK_SIZE, HEAD_DIM]
    v_cache_ptr,      # [num_blocks, num_kv_heads, BLOCK_SIZE, HEAD_DIM]
    block_tables_ptr, # [num_seqs, max_num_blocks]
    seq_lens_ptr,     # [num_seqs]
    out_ptr,          # [num_seqs, num_q_heads, HEAD_DIM]
    scale,
    stride_bt_seq, num_kv_heads: tl.constexpr,
    GROUP: tl.constexpr,        # q heads per kv head (GQA)
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    seq = tl.program_id(0)
    kv_head = tl.program_id(1)
    seq_len = tl.load(seq_lens_ptr + seq)

    # Load the GQA group of query rows that share this KV head -> [GROUP, HEAD_DIM]
    d = tl.arange(0, HEAD_DIM)
    g = tl.arange(0, GROUP)
    q = tl.load(q_ptr + seq * (num_kv_heads * GROUP * HEAD_DIM)
                + (kv_head * GROUP + g)[:, None] * HEAD_DIM + d[None, :])
    q = (q * scale).to(tl.float16)

    # Online-softmax (FlashAttention) running state, one per query row in the group
    m_i = tl.full([GROUP], -float("inf"), tl.float32)
    l_i = tl.zeros([GROUP], tl.float32)
    acc = tl.zeros([GROUP, HEAD_DIM], tl.float32)

    num_blocks = tl.cdiv(seq_len, BLOCK_SIZE)
    for b in range(0, num_blocks):
        phys = tl.load(block_tables_ptr + seq * stride_bt_seq + b)  # block-table gather
        offs = tl.arange(0, BLOCK_SIZE)
        mask = (b * BLOCK_SIZE + offs) < seq_len                    # tail guard

        base = (phys * num_kv_heads + kv_head) * BLOCK_SIZE * HEAD_DIM
        k = tl.load(k_cache_ptr + base + offs[:, None] * HEAD_DIM + d[None, :],
                    mask=mask[:, None], other=0.0)
        v = tl.load(v_cache_ptr + base + offs[:, None] * HEAD_DIM + d[None, :],
                    mask=mask[:, None], other=0.0)

        # [GROUP, HEAD_DIM] x [HEAD_DIM, BLOCK_SIZE] -> MFMA on the AMD backend
        qk = tl.dot(q, k.trans()).to(tl.float32)
        qk = tl.where(mask[None, :], qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    out_off = seq * (num_kv_heads * GROUP * HEAD_DIM) \
              + (kv_head * GROUP + g)[:, None] * HEAD_DIM + d[None, :]
    tl.store(out_ptr + out_off, acc.to(tl.float16))
```

This is the single-pass form for clarity; production AITER/vLLM splits the
`num_blocks` loop across workgroups (flash-decoding) and adds a reduction pass.

## AMD-specific optimization notes

- **KV layout for coalescing.** The physical block tensor is laid out
  `[block, kv_head, BLOCK_SIZE, HEAD_DIM]` so that the contiguous `HEAD_DIM`
  dimension feeds 128-bit `global_load`/`ds_read_b128` vectorized reads. Some
  AITER kernels further reshape K to `[..., HEAD_DIM/x, BLOCK_SIZE, x]` so the
  QKᵀ MFMA sees a transpose-free K tile.
- **FP8 KV cache.** Storing K/V as FP8 (FNUZ on gfx942, OCP on gfx950) halves
  the KV read volume — the dominant cost — and roughly doubles achievable decode
  throughput at long context. The query is kept in FP16/BF16 and K is
  dequantized (or a mixed-input MFMA is used). See
  [MFMA dtypes](../hardware/mfma.md).
- **XCD / NUMA locality.** With block tables scattering KV across HBM, decode
  traffic crosses [XCD](../hardware/chiplet-xcd.md) boundaries; the per-XCD L2 is
  a NUMA domain, so grouping a sequence's workgroups onto one XCD improves L2
  reuse for the (small) query and partial state.
- **Triton AMD knobs.** `matrix_instr_nonkdim=16` selects the `16x16` MFMA
  (a good fit for `GROUP=16` / `head_dim` tiles); `waves_per_eu` is raised to
  overlap the KV load latency since this kernel is memory bound, not VGPR bound.
  See [Triton AMD backend](../languages/triton-amd.md).
- **Tail handling.** The last KV block is partial; boundary masking uses the
  `other=0.0` predicated load (which lowers to AMD `buffer_load` OOB-returns-0
  semantics), so no divergent branch is needed for the ragged tail.

## Performance characteristics

Decode paged attention is a **bandwidth roofline** problem. The bytes moved per
step are dominated by the KV cache:

```
bytes ≈ batch * num_kv_heads * seq_len * head_dim * 2 (K and V) * sizeof(dtype)
```

Dividing by 5.3 TB/s gives a hard latency floor on
[MI300X](../../sources/docs/doc-mi300x-datasheet.md). A well-tuned kernel keeps
KV streaming at a large fraction of peak HBM bandwidth; below that, the usual
culprits are uncoalesced KV reads, too few workgroups to hide latency at small
batch (use flash-decoding chunking), or block-table gather stalls. FP8 KV cache
attacks the numerator directly. The compute side (the QKᵀ and PV MFMAs) only
becomes relevant for large GQA groups and long head dims — see the
[FlashAttention-2 algorithm](../../sources/docs/doc-flash-attention-2.md) and the
[CK-tile flash kernel](flash-attention-ck.md) for the prefill counterpart.

## Runnable example

A portable, self-checking **pure-HIP** paged-attention decode reference lives in
[`examples/paged-attention/`](../../examples/paged-attention/). It implements the
single-query-step decode described above — block-table page-table indirection,
ragged sequences with partial tail blocks, GQA (`GROUP` query heads per KV head),
and the numerically-stable online-softmax recurrence — in fp32, and verifies
every output element against a CPU reference. Decode is bandwidth-bound (no large
GEMM at `q_len=1`), so the reference uses generic HIP (LDS reduction + FMA) and
runs on **gfx950**. No gfx942 runtime is claimed.

```bash
cd examples/paged-attention && ./build.sh
```

Expected output (captured on MI355X / gfx950):

```
paged-attention decode (fp32, portable HIP, gfx950)
  num_seqs=3  num_q_heads=8  num_kv_heads=2  GROUP=4
  HEAD_DIM=64  BLOCK_SIZE=16  seq_lens={40,17,64}
  kernel time: 0.0416 ms/iter (avg of 200)
  max abs error vs CPU: 8.941e-08
PASS
```

The production AITER/vLLM kernels add FP8 KV cache, 128-bit vectorized KV loads,
flash-decoding chunk-parallel split with a reduction pass, and MFMA tiling
for large GQA groups — see the notes above.

## See also

- [FlashAttention (CK-tile) on CDNA](flash-attention-ck.md) — the prefill kernel
- [MLA decode (DeepSeek)](mla-decode.md) — latent-KV variant of decode attention
- [MFMA matrix cores](../hardware/mfma.md)
- [Triton AMD backend](../languages/triton-amd.md)

## Sources

- [AITER — AMD AI Tensor Engine for Inference](https://github.com/ROCm/aiter)
- [Flash Attention on AMD GPUs (ROCm blog)](https://rocm.blogs.amd.com/artificial-intelligence/flash-attention/README.html)
- [FlashAttention-2 (Dao, 2023)](https://arxiv.org/abs/2307.08691)
- [AMD Instinct MI300X datasheet](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
- [CK-tile FlashAttention (ROCm blog)](https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html)
