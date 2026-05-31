# Diff summary

- **files changed:** 19
- **lines:** +3798 / -1036
- **kernel-ish files:** 13

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/device_batched_contraction_multiple_d_xdl_cshuffle.hpp`  (+1019/-0)
- `include/ck/tensor_operation/gpu/device/device_grouped_contraction_multiple_d_xdl_cshuffle.hpp`  (+908/-0)
- `example/28_grouped_gemm_bias_e_permute/grouped_gemm_bias_e_permute_xdl_fp16.cpp`  (+483/-0)
- `example/29_batched_gemm_bias_e_permute/batched_gemm_bias_e_permute_xdl_fp16.cpp`  (+418/-0)
- `example/25_gemm_bias_e_permute/gemm_bias_e_permute_m3n2_xdl_fp16.cpp`  (+404/-0)
- `example/25_gemm_bias_e_permute/gemm_bias_e_permute_m2n3_xdl_fp16.cpp`  (+396/-0)
- `example/25_gemm_bias_e_permute/gemm_bias_e_permute_xdl_fp16.cpp`  (+0/-284)
- `example/28_grouped_gemm_bias/grouped_gemm_bias_xdl_fp16.cpp`  (+0/-280)
- `example/29_batched_gemm_multi_d/batched_gemm_bias_xdl_fp16.cpp`  (+0/-248)
- `example/29_batched_gemm_multi_d/batched_gemm_xdl_fp16.cpp`  (+0/-217)
- `include/ck/tensor_operation/gpu/device/device_grouped_contraction_multiple_d.hpp`  (+72/-0)
- `include/ck/tensor_operation/gpu/device/device_batched_contraction_multiple_d.hpp`  (+64/-0)
- `include/ck/tensor_operation/gpu/device/tensor_specialization.hpp`  (+28/-0)
- `example/CMakeLists.txt`  (+2/-2)
- `example/25_gemm_bias_e_permute/CMakeLists.txt`  (+2/-1)

## Key added lines (kernel files)

**`example/25_gemm_bias_e_permute/gemm_bias_e_permute_m2n3_xdl_fp16.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F16 = ck::half_t;
using F32 = float;
```

**`example/25_gemm_bias_e_permute/gemm_bias_e_permute_m3n2_xdl_fp16.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F16 = ck::half_t;
using F32 = float;
```

**`example/28_grouped_gemm_bias_e_permute/grouped_gemm_bias_e_permute_xdl_fp16.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F16 = ck::half_t;
using F32 = float;
```

**`example/29_batched_gemm_bias_e_permute/batched_gemm_bias_e_permute_xdl_fp16.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F16 = ck::half_t;
using F32 = float;
```

**`include/ck/tensor_operation/gpu/device/device_batched_contraction_multiple_d.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <index_t NumDimG,
```
