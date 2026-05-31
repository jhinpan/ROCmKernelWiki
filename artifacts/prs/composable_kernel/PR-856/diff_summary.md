# Diff summary

- **files changed:** 116
- **lines:** +710 / -457
- **kernel-ish files:** 107

## Files (by churn)

- `CMakeLists.txt`  (+70/-43)
- `include/ck/config.h.in`  (+102/-0)
- `client_example/CMakeLists.txt`  (+37/-16)
- `library/include/ck/library/tensor_operation_instance/gpu/convolution_backward_data.hpp`  (+24/-24)
- `include/ck/version.h.in`  (+40/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v3r3.hpp`  (+12/-16)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_xdl_cshuffle.hpp`  (+14/-10)
- `include/ck/tensor_operation/gpu/grid/gemm_layernorm/gridwise_gemm_multiple_d_welford_first_half_xdl_cshuffle.hpp`  (+14/-10)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm.hpp`  (+11/-11)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v3r2.hpp`  (+9/-12)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_contraction_multiple_d_xdl_cshuffle.hpp`  (+12/-8)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_multi_d_xdl.hpp`  (+12/-8)
- `include/ck/tensor_operation/gpu/device/impl/device_contraction_multiple_d_xdl_cshuffle.hpp`  (+12/-8)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle.hpp`  (+12/-8)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_contraction_multiple_d_xdl_cshuffle.hpp`  (+12/-8)

## Key added lines (kernel files)

**`include/ck/tensor_description/multi_index_transform.hpp`**
```
using LowLengthsMagicDivisorMultipiler = decltype(generate_tuple(
lambda_merge_generate_MagicDivision_calculate_magic_multiplier<LowLengths>{},
Number<NDimLow>{}));
using LowLengthsMagicDivisorShift = decltype(generate_tuple(
```

**`include/ck/tensor_operation/gpu/block/blockwise_softmax.hpp`**
```
using ThreadSliceDesc_M = decltype(make_naive_tensor_descriptor_packed(
make_tuple(ThreadSliceDesc_M_K{}.GetLength(I0))));
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_contraction_multiple_d_xdl_cshuffle.hpp`**
```
using AGridDesc_AK0_M_AK1 =
remove_cvref_t<decltype(GridwiseGemm::MakeDefaultAGridDescriptor_AK0_M_AK1(
AGridDesc_M_K{}))>;
using BGridDesc_BK0_N_BK1 =
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_e_permute_xdl.hpp`**
```
using AGridDesc_AK0_M_AK1 =
remove_cvref_t<decltype(GridwiseGemm::MakeDefaultAGridDescriptor_AK0_M_AK1(
AGridDesc_M_K{}))>;
using BGridDesc_BK0_N_BK1 =
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_multi_d_xdl.hpp`**
```
using AGridDesc_AK0_M_AK1 =
remove_cvref_t<decltype(GridwiseGemm::MakeDefaultAGridDescriptor_AK0_M_AK1(
AGridDesc_M_K{}))>;
using BGridDesc_BK0_N_BK1 =
```
