# Diff summary

- **files changed:** 36 (diff was byte-capped; summary is partial)
- **lines:** +3199 / -2435
- **kernel-ish files:** 36

## Files (by churn)

- `csrc/include/quant_utils.cuh`  (+810/-696)
- `csrc/include/custom_all_reduce.cuh`  (+556/-463)
- `csrc/kernels/cache_kernels.cu`  (+480/-397)
- `csrc/kernels/custom_kernels.cu`  (+502/-316)
- `csrc/include/hip_float8_impl.h`  (+322/-260)
- `csrc/kernels/activation_kernels.cu`  (+131/-108)
- `csrc/kernels/custom_all_reduce.cu`  (+83/-53)
- `csrc/include/dtype_fp8.cuh`  (+56/-36)
- `csrc/include/hip_float8.h`  (+57/-35)
- `csrc/kernels/fused_kernels.cu`  (+43/-17)
- `csrc/include/hip_compat.h`  (+36/-22)
- `csrc/include/pos_encoding.h`  (+15/-1)
- `csrc/include/rmsnorm.h`  (+15/-1)
- `csrc/include/custom_all_reduce.h`  (+15/-0)
- `csrc/include/dispatch_utils.h`  (+15/-0)

## Key added lines (kernel files)

**`csrc/include/attention_asm.h`**
```
std::optional<torch::Tensor> &K_QScale,
std::optional<torch::Tensor> &V_QScale,
std::optional<torch::Tensor> &out_);
```

**`csrc/include/cache.h`**
```
torch::Tensor &key_cache, torch::Tensor &value_cache,
torch::Tensor &k_dequant_scales, torch::Tensor &v_dequant_scales,
torch::Tensor &slot_mapping,
const bool asm_layout);
```

**`csrc/include/custom_all_reduce.cuh`**
```
do                                                                \
{                                                                 \
if (e != cudaSuccess)                                           \
{                                                               \
```

**`csrc/include/dtype_fp8.cuh`**
```
namespace vllm
enum class Fp8KVCacheDataType
kAuto = 0,
kFp8E4M3 = 1,
```

**`csrc/include/hip_compat.h`**
```
__shfl_xor_sync(uint32_t(-1), var, lane_mask)
__shfl_xor_sync(uint32_t(-1), var, lane_mask, width)
__shfl_xor(var, lane_mask, width)
__shfl_down_sync(uint32_t(-1), var, lane_delta)
```
