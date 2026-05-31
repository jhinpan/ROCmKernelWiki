# Diff summary

- **files changed:** 42 (diff was byte-capped; summary is partial)
- **lines:** +344 / -224
- **kernel-ish files:** 42

## Files (by churn)

- `example/ck_tile/03_gemm/run_gemm_example.inc`  (+38/-36)
- `include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_multiple_d_gemm_multiple_d_xdl_cshuffle_v1.hpp`  (+25/-7)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_two_stage_xdl_instance.hpp`  (+12/-20)
- `library/src/tensor_operation_instance/gpu/conv2d_fwd/device_conv2d_fwd_xdl_c_shuffle_nhwc_kyxc_nhwk_f16_instance.cpp`  (+12/-16)
- `library/src/tensor_operation_instance/gpu/conv2d_fwd_bias_relu/device_conv2d_fwd_xdl_c_shuffle_bias_relu_nhwc_kyxc_nhwk_f16_instance.cpp`  (+12/-16)
- `library/src/tensor_operation_instance/gpu/conv2d_fwd_bias_relu_add/device_conv2d_fwd_xdl_c_shuffle_bias_relu_add_nhwc_kyxc_nhwk_f16_instance.cpp`  (+12/-16)
- `include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_multiple_d_softmax_gemm_xdl_cshuffle_v1.hpp`  (+19/-5)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_split_k_multiple_d_xdl_cshuffle.hpp`  (+18/-4)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_b_scale.hpp`  (+13/-4)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_splitk_cshuffle.hpp`  (+10/-3)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_split_k_multiple_d_xdl_cshuffle_v2.hpp`  (+10/-3)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v2.hpp`  (+10/-3)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_waveletmodel_cshuffle.hpp`  (+10/-3)
- `library/src/tensor_operation_instance/gpu/batched_gemm_softmax_gemm/device_batched_gemm_softmax_gemm_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance.cpp`  (+4/-8)
- `include/ck/tensor_operation/gpu/grid/gemm_layernorm/gridwise_gemm_multiple_d_welford_first_half_xdl_cshuffle.hpp`  (+9/-2)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/run_gemm_example.inc`**
```
template <typename ADataType,
typename BDataType,
typename AccDataType,
typename CDataType,
```

**`include/ck/tensor_operation/gpu/grid/gemm_layernorm/gridwise_gemm_multiple_d_welford_first_half_xdl_cshuffle.hpp`**
```
constexpr auto lcm_AK1_BK1 = math::lcm(AK1, BK1);
constexpr bool is_single_rate_mfma =
((is_same<ABDataType, half_t>::value || is_same<ABDataType, bhalf_t>::value) &&
lcm_AK1_BK1 <= 4)
```

**`include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_gemm_xdl_cshuffle_v1.hpp`**
```
constexpr auto lcm_AK1_BK1 = math::lcm(AK1, BK1);
constexpr bool is_single_rate_mfma =
((is_same<FloatAB, half_t>::value || is_same<FloatAB, bhalf_t>::value) &&
lcm_AK1_BK1 <= 4)
```

**`include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_multiple_d_gemm_multiple_d_xdl_cshuffle_v1.hpp`**
```
constexpr bool is_single_rate_mfma =
((is_same<A0B0B1DataType, half_t>::value || is_same<A0B0B1DataType, bhalf_t>::value) &&
math::lcm(A0K1, B0K1) <= 4)
constexpr auto mfma = MfmaSelector<A0B0B1DataType,
```

**`include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_multiple_d_softmax_gemm_xdl_cshuffle_v1.hpp`**
```
constexpr bool is_single_rate_mfma =
((is_same<FloatAB, half_t>::value || is_same<FloatAB, bhalf_t>::value) &&
math::lcm(AK1, BK1) <= 4)
constexpr auto mfma =
```
