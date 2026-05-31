# Diff summary

- **files changed:** 25
- **lines:** +95 / -33
- **kernel-ish files:** 25

## Files (by churn)

- `aiter/ops/triton/utils/_triton/gemm_tune_check.py`  (+62/-0)
- `aiter/ops/triton/utils/gemm_config_utils.py`  (+4/-4)
- `aiter/ops/triton/gemm_afp4wfp4.py`  (+3/-3)
- `aiter/ops/triton/_triton_kernels/batched_gemm_afp4wfp4.py`  (+2/-2)
- `aiter/ops/triton/_triton_kernels/gemm_a16w16.py`  (+2/-2)
- `aiter/ops/triton/_triton_kernels/gemm_a8wfp4.py`  (+2/-2)
- `aiter/ops/triton/fused_gemm_afp4wfp4_mul_add.py`  (+2/-2)
- `aiter/ops/triton/batched_gemm_a16wfp4.py`  (+1/-1)
- `aiter/ops/triton/batched_gemm_a8w8.py`  (+1/-1)
- `aiter/ops/triton/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py`  (+1/-1)
- `aiter/ops/triton/batched_gemm_afp4wfp4.py`  (+1/-1)
- `aiter/ops/triton/batched_gemm_bf16.py`  (+1/-1)
- `aiter/ops/triton/ff_a16w16_fused_gated.py`  (+1/-1)
- `aiter/ops/triton/ff_a16w16_fused_ungated.py`  (+1/-1)
- `aiter/ops/triton/fused_gemm_a8w8_blockscale_a16w16.py`  (+1/-1)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/batched_gemm_afp4wfp4.py`**
```
config, is_tunned = get_gemm_config("BATCHED_GEMM-AFP4WFP4", M, N, 2 * K)
return config, is_tunned
```

**`aiter/ops/triton/_triton_kernels/gemm_a16w16.py`**
```
config, is_tunned = get_gemm_config("GEMM-A16W16", M, N, K)
return compute_splitk_params(config, K), is_tunned
```

**`aiter/ops/triton/_triton_kernels/gemm_a8wfp4.py`**
```
config, is_tunned = get_gemm_config("GEMM-A8WFP4", M, N, K)
return config, is_tunned
```

**`aiter/ops/triton/batched_gemm_a16wfp4.py`**
```
config, _ = _get_config(M, N, K)
```

**`aiter/ops/triton/batched_gemm_a8w8.py`**
```
config, _ = _get_config(M, N, K)
```
