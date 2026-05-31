# Diff summary

- **files changed:** 18
- **lines:** +873 / -1703
- **kernel-ish files:** 17

## Files (by churn)

- `composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v2r5.hpp`  (+0/-635)
- `composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v2r6.hpp`  (+0/-617)
- `composable_kernel/include/tensor_operation/gridwise_gemm_pipeline_v1.hpp`  (+325/-0)
- `composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v2r3.hpp`  (+77/-62)
- `composable_kernel/include/tensor_operation/threadwise_tensor_slice_transfer_v3r1.hpp`  (+43/-93)
- `composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v3r1.hpp`  (+67/-52)
- `composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v3r2.hpp`  (+67/-52)
- `composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v3r3.hpp`  (+63/-50)
- `example/1_gemm_xdl/gemm_xdl.cpp`  (+61/-40)
- `example/4_conv2d_fwd_xdl/conv2d_fwd_xdl.cpp`  (+35/-38)
- `composable_kernel/include/tensor_operation/blockwise_tensor_slice_transfer_v4r1.hpp`  (+19/-38)
- `device_operation/src/device_gemm_xdl_c_shuffle_2_stage_f16_f16_f16_mk_nk_mn_instance.cpp`  (+56/-0)
- `device_operation/src/device_gemm_xdl_f16_f16_f16_mk_nk_mn_instance.cpp`  (+32/-17)
- `device_operation/include/device_gemm_xdl.hpp`  (+10/-3)
- `profiler/include/profile_gemm_impl.hpp`  (+9/-2)

## Key added lines (kernel files)

**`composable_kernel/include/tensor_description/tensor_descriptor.hpp`**
```
static_assert(NewTransforms::Size() == NewLowerDimensionOldVisibleIdss::Size() &&
NewTransforms::Size() == NewUpperDimensionNewVisibleIdss::Size(),
"wrong! inconsitent number of transform");
```

**`composable_kernel/include/tensor_operation/blockwise_tensor_slice_transfer_v4r1.hpp`**
```
bool ThreadTransferDstResetCoordinateAfterRun,
index_t NumThreadScratch = 1>
template <typename SrcBuffer, index_t ThreadScratchId = 0>
__device__ void RunRead(const SrcDesc& src_desc,
```

**`composable_kernel/include/tensor_operation/gridwise_gemm_pipeline_v1.hpp`**
```
namespace ck {
template <typename AGridDesc,
typename ABlockDesc,
typename ABlockTransfer,
```

**`composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v2r3.hpp`**
```
bool HasMainK0BlockLoop>
GridwiseGemm::template Run<HasMainK0BlockLoop>(p_a_grid,
p_b_grid,
p_c_grid,
```

**`composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v3r1.hpp`**
```
bool HasMainK0BlockLoop>
GridwiseGemm::template Run<HasMainK0BlockLoop>(
index_t CBlockTransferScalarPerVector_NWaveNPerXdl,
index_t NumPrefetch = 1>
```
