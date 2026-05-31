# Diff summary

- **files changed:** 14
- **lines:** +2678 / -50
- **kernel-ish files:** 11

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_xdl_fpAintB_b_scale.hpp`  (+1007/-0)
- `example/24_batched_gemm/run_batched_gemm_example_fp16int4_b_scale.inc`  (+578/-0)
- `profiler/include/profiler/profile_batched_gemm_b_scale_impl.hpp`  (+488/-0)
- `profiler/src/profile_batched_gemm_b_scale.cpp`  (+200/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_b_scale/device_batched_gemm_b_scale_xdl_f16_i4_f16/device_batched_gemm_b_scale_xdl_f16_i4_f16_mk_nk_mn.hpp`  (+95/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/batched_gemm_b_scale.hpp`  (+92/-0)
- `example/24_batched_gemm/batched_gemm_xdl_fp16int4_b_scale_v3.cpp`  (+82/-0)
- `library/src/tensor_operation_instance/gpu/gemm_b_scale/device_gemm_b_scale_xdl_f16_i4_f16/device_gemm_b_scale_xdl_f16_i4_f16_mk_nk_mn.hpp`  (+34/-38)
- `include/ck/tensor_operation/gpu/device/device_batched_gemm.hpp`  (+42/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_b_scale/device_batched_gemm_b_scale_xdl_f16_i4_f16/device_batched_gemm_b_scale_xdl_f16_i4_f16_mk_nk_mn_mem_v2_default_instance.cpp`  (+33/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_b_scale.hpp`  (+12/-12)
- `library/src/tensor_operation_instance/gpu/batched_gemm_b_scale/CMakeLists.txt`  (+10/-0)
- `example/24_batched_gemm/CMakeLists.txt`  (+3/-0)
- `profiler/src/CMakeLists.txt`  (+2/-0)

## Key added lines (kernel files)

**`example/24_batched_gemm/batched_gemm_xdl_fp16int4_b_scale_v3.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F16 = ck::half_t;
using F32 = float;
```

**`example/24_batched_gemm/run_batched_gemm_example_fp16int4_b_scale.inc`**
```
struct ProblemSize final
ck::index_t M = 128;
ck::index_t N = 128;
ck::index_t K = 384;
```

**`include/ck/tensor_operation/gpu/device/device_batched_gemm.hpp`**
```
template <typename ALayout,
typename BLayout,
typename CLayout,
typename ADataType,
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_xdl_fpAintB_b_scale.hpp`**
```
namespace ck {
template <typename GridwiseGemm,
typename BatchedGemmArg,
bool HasMainKBlockLoop,
```

**`include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_b_scale.hpp`**
```
auto splitk_batch_offset = typename GridwiseGemm::SplitKBatchOffset(karg, blockIdx.z);
auto splitk_batch_offset = typename GridwiseGemm::SplitKBatchOffset(karg, blockIdx.z);
__device__ SplitKBatchOffset(Argument& karg, index_t k_id)
a_k_split_offset = k_id * karg.KRead / APackedSize;
```
