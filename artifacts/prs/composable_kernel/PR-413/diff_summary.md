# Diff summary

- **files changed:** 12
- **lines:** +2916 / -6
- **kernel-ish files:** 11

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gemm_layernorm/gridwise_gemm_multiple_d_welford_first_half_xdl_cshuffle.hpp`  (+1111/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_layernorm_xdl_cshuffle.hpp`  (+1072/-0)
- `include/ck/tensor_operation/gpu/grid/gemm_layernorm/gridwise_welford_second_half_layernorm2d.hpp`  (+394/-0)
- `example/21_gemm_layernorm/gemm_bias_relu_add_layernorm_xdl_welford_fp16.cpp`  (+262/-0)
- `include/ck/tensor_operation/gpu/device/device_gemm_multiple_d_layernorm.hpp`  (+67/-0)
- `example/21_gemm_layernorm/CMakeLists.txt`  (+4/-3)
- `library/include/ck/library/reference_tensor_operation/cpu/reference_layernorm.hpp`  (+4/-1)
- `include/ck/tensor_operation/gpu/grid/gridwise_elementwise_layernorm_welford_variance.hpp`  (+1/-1)
- `include/ck/tensor_operation/gpu/grid/gridwise_normalization_welford_variance.hpp`  (+1/-1)
- `example/21_gemm_layernorm/gemm_bias_relu_add_layernorm_xdl_naive_fp16.cpp`  (+0/-0)
- `example/21_gemm_layernorm/gemm_layernorm_xdl_naive_fp16.cpp`  (+0/-0)
- `example/21_gemm_layernorm/gemm_xdl_layernorm_naive_single_kernel_fp16.cpp`  (+0/-0)

## Key added lines (kernel files)

**`example/21_gemm_layernorm/gemm_bias_relu_add_layernorm_xdl_welford_fp16.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F16 = ck::half_t;
using F32 = float;
```

**`include/ck/tensor_operation/gpu/device/device_gemm_multiple_d_layernorm.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <typename ALayout,
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_layernorm_xdl_cshuffle.hpp`**
```
namespace ck {
template <typename GridwiseGemmWelford,
typename ABDataType,
typename DsPointer,
```

**`include/ck/tensor_operation/gpu/grid/gemm_layernorm/gridwise_gemm_multiple_d_welford_first_half_xdl_cshuffle.hpp`**
```
namespace ck {
template <typename ABDataType,
typename AccDataType,
typename CShuffleDataType,
```

**`include/ck/tensor_operation/gpu/grid/gemm_layernorm/gridwise_welford_second_half_layernorm2d.hpp`**
```
namespace ck {
template <typename EMeanVarDataType,
typename HDataType,
typename GammaDataType,
```
