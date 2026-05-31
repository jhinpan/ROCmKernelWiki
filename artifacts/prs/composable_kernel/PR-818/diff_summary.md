# Diff summary

- **files changed:** 22
- **lines:** +3593 / -4
- **kernel-ish files:** 18

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_splitk_cshuffle.hpp`  (+1086/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl_fixed_nk.hpp`  (+836/-0)
- `example/15_grouped_gemm/grouped_gemm_xdl_fixed_nk_bias_fp16.cpp`  (+353/-0)
- `example/15_grouped_gemm/grouped_gemm_xdl_fixed_nk_fp16.cpp`  (+329/-0)
- `client_example/21_grouped_gemm_bias/grouped_gemm_fixed_nk_bias_fp16.cpp`  (+244/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle.hpp`  (+172/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_gemm_bias.hpp`  (+146/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_bias/device_grouped_gemm_xdl_fixed_nk_bias_f16_f16_f32_mk_kn_mn_instance.cpp`  (+83/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_bias/device_grouped_gemm_xdl_fixed_nk_bias_f16_f16_f32_mk_nk_mn_instance.cpp`  (+83/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_bias/device_grouped_gemm_xdl_fixed_nk_bias_f16_f16_f16_mk_nk_mn_instance.cpp`  (+82/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_bias/device_grouped_gemm_xdl_fixed_nk_bias_f16_f16_f16_mk_kn_mn_instance.cpp`  (+79/-0)
- `include/ck/tensor_operation/gpu/device/device_grouped_gemm_fixed_nk.hpp`  (+63/-0)
- `library/src/utility/device_memory.cpp`  (+10/-0)
- `example/15_grouped_gemm/CMakeLists.txt`  (+6/-1)
- `include/ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp`  (+7/-0)

## Key added lines (kernel files)

**`client_example/21_grouped_gemm_bias/grouped_gemm_fixed_nk_bias_fp16.cpp`**
```
using F16 = ck::half_t;
using F32 = float;
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
```

**`example/15_grouped_gemm/grouped_gemm_xdl_fixed_nk_bias_fp16.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F16 = ck::half_t;
using F32 = float;
```

**`example/15_grouped_gemm/grouped_gemm_xdl_fixed_nk_fp16.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F16 = ck::half_t;
using F32 = float;
```

**`include/ck/tensor_operation/gpu/device/device_grouped_gemm_fixed_nk.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <index_t NumDTensor = 0>
```

**`include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl_fixed_nk.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <typename GridwiseGemm,
```
