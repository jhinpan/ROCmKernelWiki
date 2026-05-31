# Diff summary

- **files changed:** 16 (diff was byte-capped; summary is partial)
- **lines:** +3972 / -1100
- **kernel-ish files:** 15

## Files (by churn)

- `composable_kernel/include/tensor_operation/gridwise_gemm_dlops_v3.hpp`  (+1920/-0)
- `host/driver_offline/include/driver_convolution_add_forward_implicit_gemm_v5r1_dlops_nc0hwc1_kc0yxc1_nk0hwk1.hpp`  (+565/-0)
- `host/driver_offline/include/driver_convolution_forward_implicit_gemm_v5r1_dlops_nc0hwc1_kc0yxc1_nk0hwk1.hpp`  (+500/-0)
- `host/driver_offline/include/driver_convolution_forward_implicit_gemm_v5r1_dlops_nchw_kcyx_nkhw_outpad.hpp`  (+0/-364)
- `host/driver_offline/include/driver_convolution_forward_implicit_gemm_v5r1_dlops_nchw_kcyx_nkhw.hpp`  (+0/-349)
- `host/driver_offline/include/device_convolution_add_forward_implicit_gemm_v5r1_dlops_nc0hwc1_kc0yxc1_nk0hwk1.hpp`  (+220/-0)
- `host/driver_offline/include/device_convolution_maxpool_forward_implicit_gemm_v5r1_dlops_nc0hwc1_kc0yxc1_nk0hwk1.hpp`  (+212/-0)
- `composable_kernel/include/tensor_operation/threadwise_gemm_dlops_v3.hpp`  (+104/-96)
- `host/driver_offline/include/device_convolution_forward_implicit_gemm_v5r1_dlops_nc0hwc1_kc0yxc1_nk0hwk1.hpp`  (+196/-0)
- `composable_kernel/include/tensor_operation/blockwise_gemm_dlops_v3.hpp`  (+92/-100)
- `host/driver_offline/include/device_convolution_forward_implicit_gemm_v5r1_dlops_nchw_kcyx_nkhw.hpp`  (+0/-190)
- `host/driver_offline/include/driver_convolution_maxpool_forward_implicit_gemm_v5r1_dlops_nc0hwc1_kc0yxc1_nk0hwk1.hpp`  (+101/-0)
- `composable_kernel/include/tensor_operation/threadwise_tensor_slice_transfer.hpp`  (+35/-0)
- `composable_kernel/include/utility/config.hpp`  (+10/-1)
- `host/driver_offline/CMakeLists.txt`  (+9/-0)

## Key added lines (kernel files)

**`composable_kernel/include/tensor_operation/blockwise_gemm_dlops_v3.hpp`**
```
typename ABlockDesc_E1_K1_E2,
typename BBlockDesc_E1_N_Ho_Wo_E2,
typename CThreadDesc_K_N_Ho_Wo,
index_t KPerThreadLoop>
```

**`composable_kernel/include/tensor_operation/gridwise_gemm_dlops_v3.hpp`**
```
namespace ck {
template <typename GridwiseGemm,
typename FloatAB,
typename FloatC,
```

**`composable_kernel/include/tensor_operation/threadwise_gemm_dlops_v3.hpp`**
```
typename AThreadDesc_E1_K_E2,
typename BThreadDesc_E1_N_Ho_Wo_E2,
typename CThreadDesc_K_N_Ho_Wo,
typename enable_if<AThreadDesc_E1_K_E2::IsKnownAtCompileTime() &&
```

**`composable_kernel/include/tensor_operation/threadwise_tensor_slice_transfer.hpp`**
```
else if constexpr(DstInMemOp == InMemoryDataOperationEnum_t::Add)
typename vector_type_maker<DstData, DstScalarPerVector>::type tmp;
tmp.template AsType<dst_vector_t>()(Number<0>{}) =
dst_buf.template Get<dst_vector_t>(dst_coord_.GetOffset(), is_dst_valid);
```

**`composable_kernel/include/utility/amd_buffer_addressing.hpp`**
```
llvm_amdgcn_raw_buffer_store_fp32x4(as_type<float4_t>(src_thread_data),
dst_wave_buffer_resource,
dst_thread_addr_offset,
dst_wave_addr_offset,
```
