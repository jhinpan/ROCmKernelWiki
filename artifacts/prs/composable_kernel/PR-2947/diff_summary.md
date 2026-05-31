# Diff summary

- **files changed:** 13 (diff was byte-capped; summary is partial)
- **lines:** +5223 / -47
- **kernel-ish files:** 12

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_two_stage_wmma_cshuffle_v3.hpp`  (+1578/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_wmma_cshuffle_v3.hpp`  (+1417/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_multiple_d_wmma_cshuffle_v3.hpp`  (+1258/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_wmma_cshuffle_v3.hpp`  (+764/-0)
- `example/20_grouped_conv_bwd_weight/grouped_conv_bwd_weight_v3_wmma_bf16.cpp`  (+100/-0)
- `example/20_grouped_conv_bwd_weight/grouped_conv_bwd_weight_v3_wmma_fp16.cpp`  (+33/-22)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_explicit.hpp`  (+32/-13)
- `example/20_grouped_conv_bwd_weight/run_grouped_conv_bwd_weight_example.inc`  (+16/-2)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_dl.hpp`  (+5/-4)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_wmma_cshuffle.hpp`  (+5/-4)
- `example/20_grouped_conv_bwd_weight/CMakeLists.txt`  (+5/-2)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_xdl_cshuffle_v3.hpp`  (+5/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_two_stage_xdl_cshuffle.hpp`  (+5/-0)

## Key added lines (kernel files)

**`example/20_grouped_conv_bwd_weight/grouped_conv_bwd_weight_v3_wmma_bf16.cpp`**
```
using InDataType = BF16;
using WeiDataType = F32;
using OutDataType = BF16;
using AccDataType = F32;
```

**`example/20_grouped_conv_bwd_weight/grouped_conv_bwd_weight_v3_wmma_fp16.cpp`**
```
ck::tensor_operation::device::DeviceGroupedConvBwdWeight_Wmma_CShuffleV3<
ck::tuple_element_t<NDimSpatial - 1,
ck::Tuple<ck::tensor_layout::convolution::GNWC,
ck::tensor_layout::convolution::NHWGC,
```

**`example/20_grouped_conv_bwd_weight/run_grouped_conv_bwd_weight_example.inc`**
```
float max_accumulated_value =
const ck::index_t num_accums         = out.GetElementSize() / conv_param.K_;
const ck::index_t num_accums_split_k = split_k;
double rtol = ck::utils::get_relative_threshold<InDataType, WeiDataType, AccDataType>(
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_wmma_cshuffle_v3.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <typename GridwiseGemm,
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_xdl_cshuffle_v3.hpp`**
```
template <typename EType>
void SetEPointer(void* ptr)
this->p_c_grid = static_cast<EType*>(ptr);
```
