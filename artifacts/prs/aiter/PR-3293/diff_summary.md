# Diff summary

- **files changed:** 16
- **lines:** +49 / -37
- **kernel-ish files:** 16

## Files (by churn)

- `aiter/ops/triton/_gluon_kernels/gfx1250/moe/moe_op_gemm_a8w4.py`  (+20/-20)
- `aiter/ops/triton/moe/moe_routing/routing.py`  (+8/-2)
- `aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a4w4.py`  (+1/-3)
- `aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w4.py`  (+1/-3)
- `aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w8.py`  (+1/-3)
- `aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w8_blockscale.py`  (+1/-3)
- `aiter/ops/triton/_triton_kernels/moe/moe_routing/routing.py`  (+2/-2)
- `aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a16w4.py`  (+1/-1)
- `aiter/ops/triton/moe/moe_op_gemm_a16w4.py`  (+2/-0)
- `aiter/ops/triton/moe/moe_op_gemm_a4w4.py`  (+2/-0)
- `aiter/ops/triton/moe/moe_op_gemm_a8w4.py`  (+2/-0)
- `aiter/ops/triton/moe/moe_op_gemm_a8w8.py`  (+2/-0)
- `aiter/ops/triton/moe/moe_op_gemm_a8w8_blockscale.py`  (+2/-0)
- `aiter/ops/triton/moe/moe_op_gemm_int8_smoothquant.py`  (+2/-0)
- `aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_int8_smoothquant.py`  (+1/-0)

## Key added lines (kernel files)

**`aiter/ops/triton/_gluon_kernels/gfx1250/moe/moe_op_gemm_a8w4.py`**
```
gindx = gindx.to(torch.int32)
for _ in gl.static_range(NUM_BUFFERS):
gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * NUM_TDM_OPS)
for k in range(num_k_iter - NUM_BUFFERS):
```

**`aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a16w4.py`**
```
gindx = gindx.to(torch.int32)
```

**`aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a4w4.py`**
```
gindx = gindx.to(torch.int32)
```

**`aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w4.py`**
```
gindx = gindx.to(torch.int32)
```

**`aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w8.py`**
```
gindx = gindx.to(torch.int32)
```
