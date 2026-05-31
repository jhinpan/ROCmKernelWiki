# Diff summary

- **files changed:** 22
- **lines:** +939 / -134
- **kernel-ish files:** 20

## Files (by churn)

- `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`  (+528/-60)
- `vllm/model_executor/layers/mhc.py`  (+105/-2)
- `vllm/model_executor/layers/deepseek_v4_attention.py`  (+73/-19)
- `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`  (+79/-2)
- `csrc/moe/topk_softplus_sqrt_kernels.cu`  (+32/-21)
- `csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu`  (+36/-2)
- `vllm/model_executor/layers/sparse_attn_indexer.py`  (+22/-8)
- `vllm/model_executor/kernels/linear/scaled_mm/aiter.py`  (+15/-0)
- `CMakeLists.txt`  (+6/-6)
- `vllm/model_executor/layers/quantization/utils/fp8_utils.py`  (+9/-0)
- `vllm/model_executor/models/deepseek_v4_mtp.py`  (+6/-2)
- `vllm/model_executor/models/deepseek_v4.py`  (+6/-1)
- `tests/kernels/moe/test_topk_softplus_sqrt.py`  (+4/-2)
- `vllm/model_executor/layers/deepseek_compressor.py`  (+3/-2)
- `vllm/model_executor/layers/activation.py`  (+3/-1)

## Key added lines (kernel files)

**`csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu`**
```
__device__ __forceinline__ uint8_t rocm_cvt_float_to_fp8_e4m3(float val) {
__hip_fp8_e4m3 fp8_val(val);
__hip_fp8_e4m3_fnuz fp8_val(val);
return reinterpret_cast<uint8_t&>(fp8_val);
```

**`csrc/moe/topk_softplus_sqrt_kernels.cu`**
```
for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
selected_sum +=
VLLM_SHFL_XOR_SYNC_WIDTH(selected_sum, mask, THREADS_PER_ROW);
static constexpr int BYTES_PER_LDG_MULTIPLE_64_NARROW =
```

**`tests/kernels/moe/test_topk_softplus_sqrt.py`**
```
not current_platform.is_cuda_alike(),
reason="This test is skipped on non-CUDA platform.",
not current_platform.is_cuda_alike(),
reason="This test is skipped on non-CUDA platform.",
```

**`vllm/config/kernel.py`**
```
"triton_unfused",
- "triton_unfused": Use Triton unfused MoE kernels
```

**`vllm/model_executor/kernels/linear/scaled_mm/aiter.py`**
```
if As.dtype != Bs.dtype:
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
_upcast_e8m0_to_fp32,
if As.dtype == torch.float8_e8m0fnu:
```
