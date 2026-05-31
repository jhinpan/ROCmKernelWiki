# Diff summary

- **files changed:** 67
- **lines:** +59 / -71
- **kernel-ish files:** 40

## Files (by churn)

- `aiter/ops/triton/utils/_triton/arch_info.py`  (+0/-12)
- `aiter/ops/triton/_triton_kernels/batched_gemm_a16wfp4.py`  (+2/-2)
- `aiter/ops/triton/_triton_kernels/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py`  (+2/-2)
- `aiter/ops/triton/_triton_kernels/batched_gemm_afp4wfp4.py`  (+2/-2)
- `aiter/ops/triton/_triton_kernels/ff_a16w16_fused_gated.py`  (+2/-2)
- `aiter/ops/triton/_triton_kernels/ff_a16w16_fused_ungated.py`  (+2/-2)
- `aiter/ops/triton/_triton_kernels/fused_gemm_a8w8_blockscale_a16w16.py`  (+2/-2)
- `aiter/ops/triton/_triton_kernels/fused_gemm_afp4wfp4_a16w16.py`  (+2/-2)
- `aiter/ops/triton/_triton_kernels/fused_gemm_afp4wfp4_mul_add.py`  (+2/-2)
- `aiter/ops/triton/_triton_kernels/fused_gemm_afp4wfp4_split_cat.py`  (+2/-2)
- `aiter/ops/triton/_triton_kernels/gemm_a16w16.py`  (+2/-2)
- `aiter/ops/triton/_triton_kernels/gemm_a16w16_atomic.py`  (+2/-2)
- `aiter/ops/triton/_triton_kernels/gemm_a16w16_gated.py`  (+2/-2)
- `aiter/ops/triton/_triton_kernels/gemm_a16w8_blockscale.py`  (+2/-2)
- `aiter/ops/triton/_triton_kernels/gemm_a16wfp4.py`  (+2/-2)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/batched_gemm_a16wfp4.py`**
```
dev = arch_info.get_arch()
dev = arch_info.get_arch()
```

**`aiter/ops/triton/_triton_kernels/batched_gemm_a8w8.py`**
```
dev = arch_info.get_arch()
```

**`aiter/ops/triton/_triton_kernels/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py`**
```
dev = arch_info.get_arch()
dev = arch_info.get_arch()
```

**`aiter/ops/triton/_triton_kernels/batched_gemm_afp4wfp4.py`**
```
dev = arch_info.get_arch()
dev = arch_info.get_arch()
```

**`aiter/ops/triton/_triton_kernels/batched_gemm_bf16.py`**
```
dev = arch_info.get_arch()
```
