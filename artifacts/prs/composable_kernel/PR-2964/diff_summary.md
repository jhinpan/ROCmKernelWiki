# Diff summary

- **files changed:** 13 (diff was byte-capped; summary is partial)
- **lines:** +3102 / -27
- **kernel-ish files:** 11

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_wmma_cshuffle_v3.hpp`  (+2353/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_instance.hpp`  (+273/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3.hpp`  (+138/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_scaleadd_ab_instance.hpp`  (+135/-0)
- `example/62_convnd_activ/multi_AB/convnd_fwd_activ_multi_ab_common.hpp`  (+63/-0)
- `example/62_convnd_activ/multi_AB/conv_fwd_wmma_cshufflev3_scaleadd_ab_bf16.cpp`  (+27/-0)
- `example/62_convnd_activ/multi_AB/conv_fwd_wmma_cshufflev3_scaleadd_ab_fp16.cpp`  (+27/-0)
- `example/62_convnd_activ/multi_AB/conv_fwd_wmma_cshufflev3_scaleadd_ab_int8.cpp`  (+27/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3_common.hpp`  (+15/-9)
- `include/ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_v7r3.hpp`  (+19/-2)
- `experimental/builder/test/CMakeLists.txt`  (+8/-8)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp`  (+8/-8)
- `example/62_convnd_activ/multi_AB/CMakeLists.txt`  (+9/-0)

## Key added lines (kernel files)

**`example/62_convnd_activ/multi_AB/conv_fwd_wmma_cshufflev3_scaleadd_ab_bf16.cpp`**
```
using DataType    = ck::bhalf_t;
using AccDataType = float;
using InDataType  = DataType;
using WeiDataType = DataType;
```

**`example/62_convnd_activ/multi_AB/conv_fwd_wmma_cshufflev3_scaleadd_ab_fp16.cpp`**
```
using DataType    = ck::half_t;
using AccDataType = float;
using InDataType  = DataType;
using WeiDataType = DataType;
```

**`example/62_convnd_activ/multi_AB/conv_fwd_wmma_cshufflev3_scaleadd_ab_int8.cpp`**
```
using DataType    = int8_t;
using AccDataType = int32_t;
using InDataType  = DataType;
using WeiDataType = DataType;
```

**`example/62_convnd_activ/multi_AB/convnd_fwd_activ_multi_ab_common.hpp`**
```
template <typename DataType,
typename AccDataType,
typename InDataTypes,
typename WeiDataTypes,
```

**`include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_wmma_cshuffle_v3.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace {
```
