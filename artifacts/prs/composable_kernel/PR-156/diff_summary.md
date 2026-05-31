# Diff summary

- **files changed:** 27
- **lines:** +2145 / -62
- **kernel-ish files:** 20

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/device_batched_gemm_reduce_xdl_cshuffle.hpp`  (+940/-0)
- `profiler/include/profile_batched_gemm_reduce_impl.hpp`  (+354/-0)
- `example/18_batched_gemm_reduce/batched_gemm_reduce_xdl_fp16.cpp`  (+281/-0)
- `profiler/src/profile_batched_gemm_reduce.cpp`  (+154/-0)
- `include/ck/tensor_operation/gpu/device/device_batched_gemm_xdl.hpp`  (+33/-41)
- `library/src/tensor_operation_instance/gpu/batched_gemm_reduce/device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gkm_gkn_gmn_instance.cpp`  (+70/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_reduce/device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gkm_gnk_gmn_instance.cpp`  (+70/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_reduce/device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gmk_gkn_gmn_instance.cpp`  (+70/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_reduce/device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gmk_gnk_gmn_instance.cpp`  (+67/-0)
- `test/batched_gemm_reduce/batched_gemm_reduce_fp16.cpp`  (+64/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_reduce/CMakeLists.txt`  (+11/-0)
- `library/include/ck/library/host_tensor/host_tensor.hpp`  (+5/-4)
- `test/batched_gemm_reduce/CMakeLists.txt`  (+9/-0)
- `test/gemm_reduce/gemm_reduce_fp16.cpp`  (+0/-6)
- `profiler/src/profiler.cpp`  (+5/-0)

## Key added lines (kernel files)

**`example/18_batched_gemm_reduce/batched_gemm_reduce_xdl_fp16.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F16 = ck::half_t;
using F32 = float;
```

**`include/ck/tensor_operation/gpu/device/device_batched_gemm_reduce_xdl_cshuffle.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <typename GridwiseGemm,
```

**`include/ck/tensor_operation/gpu/device/device_batched_gemm_xdl.hpp`**
```
const index_t batch_count,
__builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);
static constexpr auto MakeBlock2CTileMap(index_t batch_count,
const CGridDesc_M_N& c_grid_desc_m_n,
```

**`include/ck/tensor_operation/gpu/device/device_gemm_reduce.hpp`**
```
D1ReduceOperation d1_reduce_op,
ck::index_t BatchCount = 1) = 0;
```

**`include/ck/tensor_operation/gpu/device/device_gemm_reduce_xdl_cshuffle.hpp`**
```
D1ReduceOperation d1_reduce_op,
index_t /* KBatch */ = 1) override
```
