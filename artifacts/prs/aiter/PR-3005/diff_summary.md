# Diff summary

- **files changed:** 7 (diff was byte-capped; summary is partial)
- **lines:** +3201 / -7
- **kernel-ish files:** 2

## Files (by churn)

- `aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_qwen3_next_80b_a3b.csv`  (+1483/-0)
- `aiter/configs/model_configs/a8w8_blockscale_untuned_gemm_qwen3_next_80b_a3b.csv`  (+1483/-0)
- `aiter/ops/triton/_triton_kernels/causal_conv1d_update_single_token.py`  (+213/-0)
- `aiter/ops/gemm_op_a8w8.py`  (+9/-2)
- `.github/scripts/build_aiter_triton.sh`  (+9/-1)
- `.github/workflows/aiter-test.yaml`  (+3/-3)
- `.github/workflows/vllm_benchmark.yaml`  (+1/-1)

## Key added lines (kernel files)

**`aiter/ops/gemm_op_a8w8.py`**
```
splitK: int = 0,
splitK: int = 0,
splitK = int(config.get("splitK", 0))
return gemm_a8w8_blockscale_ck(
```

**`aiter/ops/triton/_triton_kernels/causal_conv1d_update_single_token.py`**
```
import triton
import triton.language as tl
@triton.jit()
def _causal_conv1d_update_single_token_kernel(
```
