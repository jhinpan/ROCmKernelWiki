---
id: kernel-mla-decode
title: MLA Decode (DeepSeek Multi-Latent Attention) on MI300
type: kernel
architectures:
- gfx942
- gfx950
tags:
- mla
- attention
- decode
- kv-cache
- flash-attention
- mfma
- memory-bound
- bf16
- fp8
confidence: source-reported
reproducibility: runnable
artifact_dir: examples/mla-decode
kernel_types:
- mla
- attention
- decode
languages:
- hip
- triton
related:
- kernel-paged-attention
- kernel-flash-attention-ck
- hw-mfma
- hw-chiplet-xcd
- technique-kernel-fusion
- technique-split-k
sources:
- ref-aiter
- hw-mfma
- blog-flash-attention-amd
- doc-flash-attention-2
- doc-mi300x-datasheet
- blog-cktile-flash
performance_claims:
- gpu: MI300X
  dtype: bf16
  metric: kv-cache-bandwidth-utilization
  value: ~80-90% of 5.3 TB/s HBM3 peak at large batch
  source_id: doc-mi300x-datasheet
  confidence: inferred
  unreproduced: true
- gpu: MI300X
  dtype: bf16
  metric: arithmetic-intensity-vs-mha
  value: weight absorption raises AI ~128x (one latent KV shared across 128 heads)
  source_id: doc-flash-attention-2
  confidence: inferred
  unreproduced: true
- gpu: MI300X
  dtype: fp8
  metric: kv-cache-footprint
  value: 576 elems/token/layer (512 latent + 64 rope) vs ~16k for dense MHA
  source_id: ref-aiter
  confidence: source-reported
  unreproduced: true
implemented_by:
- pr-sglang-21511
- pr-vllm-40871
- pr-vllm-43543
- pr-vllm-39616
- pr-sglang-25463
- pr-sglang-26134
- pr-sglang-24933
- pr-aiter-3072
---
# MLA Decode (DeepSeek Multi-Latent Attention) on MI300

## Overview

**Multi-head Latent Attention (MLA)** is the attention variant introduced in
DeepSeek-V2 / V3. Instead of caching a full key and value per head, MLA caches a
single low-rank **latent** vector per token plus a small decoupled RoPE key.
For DeepSeek-V3 the cached state per token per layer is:

- `kv_lora_rank = 512` — the compressed latent (`c_kv`), shared by all heads,
- `qk_rope_head_dim = 64` — the decoupled RoPE key `k_pe`, also shared,

for a total of **576 elements/token/layer** — versus roughly `2 · n_heads ·
head_dim ≈ 16k` for dense multi-head attention. The query side uses
`n_heads = 128`, `qk_nope_head_dim = 128`, and `v_head_dim = 128`.

Decode (one new token at a time, `q_len = 1`) is **strongly memory-bound**: the
kernel streams the entire KV-cache history for the sequence and does very little
math per byte. The whole optimization game is (1) shrinking the bytes moved and
(2) keeping the [matrix cores](../hardware/mfma.md) fed despite the tiny
`q_len`. MLA wins on both axes through *weight absorption*.

## Weight absorption: MLA decode ≈ MQA

Naively, decode would up-project the latent `c_kv` back to per-head K and V
(`n_heads · head_dim`) before attending — that reads/writes huge intermediate
tensors. The standard trick folds the up-projection matrices into the Q and
output projections **offline**:

- `W_q_nope · W_k_nope^T` is pre-multiplied into the query, so the score
  `q · k` is computed directly against the *latent* `c_kv` (dim 512).
- `W_v · W_o` is folded into the output projection, so attention outputs the
  latent and the value up-projection disappears from the hot loop.

After absorption, **all 128 heads attend against the same shared latent KV** —
structurally this is Multi-Query Attention (MQA) with an effective key width of
`512 + 64 = 576` and value width `512`. Two consequences for the kernel:

1. **Arithmetic intensity jumps ~128×** because one KV read serves 128 query
   heads — exactly the property that lets a memory-bound decode kernel still
   reach high MFMA utilization on large batches.
2. The KV-cache is paged (a single latent stream), so MLA decode reuses the same
   block-table machinery as [paged attention](paged-attention.md).

## Kernel structure (flash-decoding)

MLA decode uses the **flash-decoding** schedule: split the KV history into
chunks across the grid's `z`-dimension (a [split-K](../techniques/split-k.md)
over the sequence), run online-softmax flash attention
([FlashAttention-2](https://arxiv.org/abs/2307.08691) style) on each chunk, then
a second small kernel rescales and reduces the partials. This exposes enough
parallelism to fill all 304 CUs even at batch size 1, which a single-block
decode could never do.

A Triton sketch of the per-chunk inner kernel (absorbed MLA, BF16 latent):

```python
import triton
import triton.language as tl

@triton.jit
def mla_decode_chunk(
    Q_nope, Q_pe,                # [H, 512], [H, 64]  (absorbed query)
    KV_c, K_pe,                  # paged latent [N, 512], rope key [N, 64]
    Out_partial, Lse_partial,    # [num_chunks, H, 512], [num_chunks, H]
    block_table, seq_len,
    sm_scale,
    H: tl.constexpr, D_C: tl.constexpr,      # 128, 512
    D_PE: tl.constexpr, BLOCK_N: tl.constexpr  # 64, 64
):
    chunk = tl.program_id(0)
    # Load this head-group's absorbed query once (stays in VGPRs/MFMA tiles)
    h = tl.arange(0, H)
    q_c  = tl.load(Q_nope + h[:, None] * D_C  + tl.arange(0, D_C)[None, :])
    q_pe = tl.load(Q_pe   + h[:, None] * D_PE + tl.arange(0, D_PE)[None, :])

    m_i = tl.full((H,), -float("inf"), tl.float32)   # running max
    l_i = tl.zeros((H,), tl.float32)                 # running denom
    acc = tl.zeros((H, D_C), tl.float32)             # latent accumulator

    start = chunk * BLOCK_N
    for n in range(start, tl.minimum(start + BLOCK_N, seq_len), 16):
        kv = load_paged(KV_c,  block_table, n, D_C)   # [16, 512]
        kp = load_paged(K_pe,  block_table, n, D_PE)  # [16, 64]
        # score = q_nope . c_kv^T  +  q_pe . k_pe^T   -> tl.dot lowers to v_mfma_*
        s = tl.dot(q_c, tl.trans(kv)) + tl.dot(q_pe, tl.trans(kp))
        s *= sm_scale
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)  # attend latent
        m_i = m_new

    tl.store(Out_partial + chunk * H * D_C + h[:, None] * D_C
             + tl.arange(0, D_C)[None, :], acc / l_i[:, None])
    tl.store(Lse_partial + chunk * H + h, m_i + tl.log(l_i))
```

Notes that matter on CDNA:

- `tl.dot` lowers to `v_mfma_f32_16x16x16_f16` / `..._bf16` on gfx942 (see
  [MFMA](../hardware/mfma.md)). Because the absorbed key width is 512/64, the
  contraction `K` dimension is large and MFMA stays busy even though `q_len = 1`
  is packed into the 128-head dimension as the MFMA `M`.
- Set `waves_per_eu` and `matrix_instr_nonkdim` via the Triton AMD backend; for
  decode the kernel is latency/bandwidth bound, so favor **occupancy** over big
  per-wave tiles.
- The paged loads should use **buffer loads with OOB guards**
  (`num_records`-based) so the ragged tail of each sequence costs no branch.

## Production path: AITER

In practice you do not hand-roll this. **AITER** (AMD's inference operator
library and the default vLLM/SGLang attention backend) ships tuned MLA decode in
both Triton and CK/ASM forms and selects per shape. A typical call site:

```python
# AITER MLA decode (paged latent KV). Backend = Triton or CK/ASM, auto-selected.
from aiter.mla import mla_decode_fwd

out = mla_decode_fwd(
    q,                 # [num_tokens, n_heads, qk_nope+qk_rope]  (absorbed)
    kv_cache,          # paged latent [num_blocks, block_size, 512+64], bf16/fp8
    block_table,       # [batch, max_blocks]
    seq_lens,          # [batch]
    sm_scale=head_dim ** -0.5,
    num_kv_splits=8,   # flash-decoding split over sequence
)
```

FP8 KV-cache (FNUZ on [gfx942](../migration/gfx942-to-gfx950.md), OCP on
gfx950) halves the 576-element footprint again and is the common production
configuration for long context, at the cost of a dequant in the score `tl.dot`.

## Performance notes

Because decode is bandwidth-bound, the right roofline ceiling is **HBM3
bandwidth, 5.3 TB/s on MI300X**, not the 1307 TFLOPS BF16 matrix peak. A
well-tuned MLA decode at large batch should approach 80–90% of HBM3 bandwidth;
the FLOP utilization will look low and that is *expected* and correct for a
memory-bound op. Watch for:

- **XCD locality** — the paged latent for a sequence should stay resident in one
  [XCD's L2](../hardware/chiplet-xcd.md); scattering block-table pages across
  XCDs causes cross-XCD L2 traffic and stalls.
- **Split count (`num_kv_splits`)** — too few starves the 304 CUs at low batch
  (tail effect); too many inflates the partial-reduction cost. Tune per
  `(batch, seq_len)`.
- **Latent in LDS** — staging the chunk's `c_kv` through
  [direct-to-LDS](../hardware/async-copy-lds.md) frees VGPRs for the MFMA
  accumulators and overlaps the HBM stream with the matrix math.

## Performance claims

All figures are bandwidth-roofline reasoning for the absorbed MLA decode shape;
treat them as `inferred`/`source-reported` until reproduced on your stack.

- MI300X, BF16: kernel approaches **~80–90% of the 5.3 TB/s HBM3 peak** at large
  batch (memory-bound), per the [MI300X datasheet](../../sources/docs/doc-mi300x-datasheet.md).
- Weight absorption raises arithmetic intensity **~128×** (one latent KV serves
  128 heads), the property that keeps the matrix core fed at `q_len = 1`.
- FP8 KV-cache stores **576 elements/token/layer** (512 latent + 64 rope), per
  [AITER](../../sources/refs/ref-aiter.md).

## Runnable example

A self-checking **portable HIP** reference for the absorbed MLA decode math lives
in [`examples/mla-decode/`](../../examples/mla-decode/). It is *not* the tuned
CDNA/MFMA production kernel — it is a small, readable fp32 implementation of the
exact decode math (low-rank latent KV, online softmax, value == latent) that
builds and runs on **gfx950** and verifies against a CPU reference. The
Triton/CK production path above targets gfx942/gfx950 matrix cores; this example
keeps the math directly inspectable.

```bash
cd examples/mla-decode && ./build.sh
```

Expected output (captured on MI355X / gfx950):

```
MLA decode (absorbed, low-rank latent KV) -- portable HIP, fp32
  H=16 heads, D_C=64 latent, D_PE=16 rope, N=256 KV tokens
  per-decode: 173.68 us   KV-stream BW: 0.5 GB/s
  max_abs_err = 3.353e-08   max_rel_err = 1.301e-04
PASS
```

The dims are tiny but realistically proportioned (latent ≫ rope, value ==
latent). The reported "BW" is launch-overhead-dominated at this size and is *not*
a benchmark — it only marks decode as the memory-bound term.

## See also

- [Paged attention decode](paged-attention.md)
- [FlashAttention-2 CK-tile on CDNA](flash-attention-ck.md)
- [MFMA matrix cores](../hardware/mfma.md)
- [Split-K / flash-decoding](../techniques/split-k.md)

## Sources

- [AITER — AMD AI operator library (MLA/PagedAttn)](https://github.com/ROCm/aiter)
- [MFMA — AMD Matrix Core Instructions (CDNA)](../hardware/mfma.md)
- [Accelerating Flash Attention on AMD GPUs (ROCm blog)](https://rocm.blogs.amd.com/artificial-intelligence/flash-attention/README.html)
- [FlashAttention-2 (Dao, 2023)](https://arxiv.org/abs/2307.08691)
- [AMD Instinct MI300X datasheet](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
- [CK-Tile Flash Attention](https://rocm.blogs.amd.com/)
