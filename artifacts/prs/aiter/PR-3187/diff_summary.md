# Diff summary

- **files changed:** 21
- **lines:** +36 / -68
- **kernel-ish files:** 21

## Files (by churn)

- `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w8_blockscale.py`  (+4/-12)
- `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py`  (+2/-10)
- `aiter/ops/triton/_triton_kernels/gemm/fused/fused_gemm_a8w8_blockscale_split_cat.py`  (+2/-10)
- `aiter/ops/triton/_triton_kernels/gmm.py`  (+5/-5)
- `aiter/ops/triton/_triton_kernels/gemm/fused/fused_gemm_a8w8_blockscale_a16w16.py`  (+2/-6)
- `aiter/ops/triton/_triton_kernels/gemm/fused/fused_gemm_a8w8_blockscale_mul_add.py`  (+1/-5)
- `aiter/ops/triton/_triton_kernels/attention/fp8_mqa_logits.py`  (+2/-2)
- `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16_gated.py`  (+2/-2)
- `aiter/ops/triton/_triton_kernels/gemm/feed_forward/ff_a16w16_fused_gated.py`  (+2/-2)
- `aiter/ops/triton/_triton_kernels/gemm/fused/fused_gemm_afp4wfp4_a16w16.py`  (+2/-2)
- `aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_int8_smoothquant.py`  (+2/-2)
- `aiter/ops/triton/_triton_kernels/fusions/fused_bmm_rope_kv_cache.py`  (+1/-1)
- `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py`  (+1/-1)
- `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16_atomic.py`  (+1/-1)
- `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8.py`  (+1/-1)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/attention/fp8_mqa_logits.py`**
```
scores = tl.dot(q_block, kv_block)
scores = tl.dot(q_block, kv_block)
```

**`aiter/ops/triton/_triton_kernels/fusions/fused_bmm_rope_kv_cache.py`**
```
accumulator += tl.dot(a, b) * a_scale
```

**`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py`**
```
accumulator += tl.dot(a, b)
```

**`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16_atomic.py`**
```
accumulator += tl.dot(a, b)
```

**`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16_gated.py`**
```
acc0 += tl.dot(a, b0)
acc1 += tl.dot(a, b1)
```
