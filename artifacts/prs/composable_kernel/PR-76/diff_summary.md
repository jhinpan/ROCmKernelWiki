# Diff summary

- **files changed:** 54 (diff was byte-capped; summary is partial)
- **lines:** +2209 / -767
- **kernel-ish files:** 49

## Files (by churn)

- `device_operation/include/device_gemm_xdl_c_shuffle_bias_activation_add.hpp`  (+574/-0)
- `device_operation/include/device_gemm_xdl_c_shuffle_bias_activation.hpp`  (+163/-186)
- `example/3_gemm_xdl_bias_relu_add/gemm_xdl_bias_relu_add.cpp`  (+93/-183)
- `example/2_gemm_xdl_bias_relu/gemm_xdl_bias_relu.cpp`  (+235/-0)
- `composable_kernel/include/tensor_operation/element_wise_operation.hpp`  (+58/-137)
- `device_operation/CMakeLists.txt`  (+111/-0)
- `profiler/CMakeLists.txt`  (+13/-75)
- `profiler/include/profile_conv_fwd_bias_relu_add_impl.hpp`  (+19/-64)
- `example/2_gemm_xdl_bias_relu/README.md`  (+61/-0)
- `device_operation/src/device_gemm_xdl_c_shuffle_bias_relu_add_f16_f16_f16_mk_nk_mn_instance.cpp`  (+57/-0)
- `device_operation/src/device_gemm_xdl_c_shuffle_bias_relu_f16_f16_f16_mk_nk_mn_instance.cpp`  (+57/-0)
- `device_operation/src/device_gemm_xdl_c_shuffle_f16_f16_f16_mk_nk_mn_instance.cpp`  (+57/-0)
- `example/1_gemm_xdl/gemm_xdl.cpp`  (+17/-36)
- `device_operation/src/device_gemm_xdl_c_shuffle_bias_relu_add_f16_f16_f16_km_kn_mn_instance.cpp`  (+52/-0)
- `device_operation/src/device_gemm_xdl_c_shuffle_bias_relu_add_f16_f16_f16_km_nk_mn_instance.cpp`  (+52/-0)

## Key added lines (kernel files)

**`composable_kernel/include/tensor_operation/element_wise_operation.hpp`**
```
__host__ __device__ void operator()(float& y, const float& x) const { y = x; }
__host__ __device__ void operator()(half_t& y, const half_t& x) const { y = x; }
__host__ __device__ constexpr void operator()(float& y, const float& x0, const float& x1) const
const float a = x0 + x1;
```

**`composable_kernel/include/tensor_operation/threadwise_tensor_slice_transfer.hpp`**
```
SrcData dst_v;
dst_element_op_(dst_v, src_buf[Number<src_offset>{}]);
dst_vector.template AsType<DstData>()(i) = type_convert<DstData>(dst_v);
```

**`composable_kernel/include/tensor_operation/threadwise_tensor_slice_transfer_v1r4.hpp`**
```
DstData dst_v;
dst_element_op_(dst_v, src_v, dst0_v, dst1_v);
```

**`composable_kernel/include/tensor_operation/threadwise_tensor_slice_transfer_v3r1.hpp`**
```
SrcData src_v;
src_element_op_(src_v, src_vector_container.template AsType<SrcData>()[i]);
src_vector_container.template AsType<SrcData>()(i) = src_v;
DstData dst_v;
```

**`device_operation/include/device_conv2d_fwd_xdl_c_shuffle_bias_activation_add_nhwc_kyxc_nhwk.hpp`**
```
using GridDescs = decltype(MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(
using AGridDesc_K0_M_K1 = remove_cvref_t<decltype(GridDescs{}[I0])>;
using BGridDesc_K0_N_K1 = remove_cvref_t<decltype(GridDescs{}[I1])>;
using CGridDesc_M_N     = remove_cvref_t<decltype(GridDescs{}[I2])>;
```
