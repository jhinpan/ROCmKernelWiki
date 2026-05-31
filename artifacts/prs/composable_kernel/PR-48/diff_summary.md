# Diff summary

- **files changed:** 34 (diff was byte-capped; summary is partial)
- **lines:** +3816 / -333
- **kernel-ish files:** 30

## Files (by churn)

- `external/half/include/half.hpp`  (+1394/-0)
- `device_operation/include/device_conv_fwd_xdl_nhwc_kyxc_nhwk.hpp`  (+601/-0)
- `device_operation/include/device_gemm_xdl.hpp`  (+442/-0)
- `composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v2r3.hpp`  (+125/-122)
- `example/1_gemm_xdl/gemm_xdl.cpp`  (+202/-0)
- `README.md`  (+0/-173)
- `device_operation/include/device_conv.hpp`  (+78/-0)
- `device_operation/device_conv_xdl_instance_f16_f16_f16_nhwc_kyxc_nhwk.cpp`  (+64/-0)
- `device_operation/device_conv_xdl_instance_f32_f32_f32_nhwc_kyxc_nhwk.cpp`  (+64/-0)
- `device_operation/device_gemm_xdl_instance_f16_f16_f16_mk_nk_mn.cpp`  (+63/-0)
- `device_operation/device_gemm_xdl_instance_f32_f32_f32_mk_nk_mn.cpp`  (+63/-0)
- `device_operation/device_gemm_xdl_instance_f16_f16_f16_km_kn_mn.cpp`  (+58/-0)
- `device_operation/device_gemm_xdl_instance_f16_f16_f16_km_nk_mn.cpp`  (+58/-0)
- `device_operation/device_gemm_xdl_instance_f16_f16_f16_mk_kn_mn.cpp`  (+58/-0)
- `device_operation/device_gemm_xdl_instance_f32_f32_f32_km_kn_mn.cpp`  (+58/-0)

## Key added lines (kernel files)

**`composable_kernel/include/problem_transform/transform_forward_convolution_into_gemm_v4r4r4_nhwc_kyxc_nhwk.hpp`**
```
__host__ __device__ constexpr auto transform_forward_convolution_into_gemm_v4r4r4_nhwc_kyxc_nhwk(
```

**`composable_kernel/include/tensor_operation/blockwise_gemm_xdlops.hpp`**
```
__host__ __device__ static constexpr auto GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2()
__host__ __device__ static constexpr auto GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2()
constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2 =
return xdlops_gemm.MakeCDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(c_block_desc_m0_n0_m1_n1_m2_n2);
```

**`composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v2r3.hpp`**
```
typename AGridDesc_K0_M_K1,
typename BGridDesc_K0_N_K1,
typename CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2,
typename Block2CTileMap,
```

**`composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v2r4.hpp`**
```
return BlockwiseGemm::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(c_m_n_grid_desc);
blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();
```

**`composable_kernel/include/tensor_operation/xdlops_gemm.hpp`**
```
template <typename CDesc_M0_N0_M1_N1_M2_N2>
MakeCDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(const CDesc_M0_N0_M1_N1_M2_N2& c_desc_m0_n0_m1_n1_m2_n2)
const auto M0 = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I0);
const auto N0 = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I1);
```
