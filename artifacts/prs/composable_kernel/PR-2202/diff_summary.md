# Diff summary

- **files changed:** 34
- **lines:** +549 / -181
- **kernel-ish files:** 34

## Files (by churn)

- `include/ck/tensor_operation/gpu/warp/xdlops_gemm.hpp`  (+114/-17)
- `include/ck/utility/amd_xdlops.hpp`  (+90/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_multiple_d_gemm_multiple_d_xdl_cshuffle_v1.hpp`  (+20/-13)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_split_k_multiple_d_xdl_cshuffle.hpp`  (+22/-10)
- `include/ck/tensor_operation/gpu/grid/gridwise_moe_gemm.hpp`  (+21/-8)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_multi_d_b_preshuffle.hpp`  (+19/-6)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle_lds_direct_load.hpp`  (+12/-9)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_abd_xdl_cshuffle.hpp`  (+11/-9)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle.hpp`  (+11/-9)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_b_preshuffle.hpp`  (+14/-6)
- `include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_multiple_d_softmax_gemm_xdl_cshuffle_v1.hpp`  (+13/-6)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_waveletmodel_cshuffle.hpp`  (+12/-5)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_bwd_weight.hpp`  (+11/-6)
- `include/ck/tensor_operation/gpu/grid/gemm_layernorm/gridwise_gemm_multiple_d_welford_first_half_xdl_cshuffle.hpp`  (+11/-5)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_splitk_cshuffle.hpp`  (+11/-5)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_dequant_v3.hpp`**
```
static constexpr auto xdlops_gemm =
XdlopsGemm<ComputeDataType, MPerXDL, NPerXDL, KPack, BDataType>{};
```

**`include/ck/tensor_operation/gpu/grid/gemm_layernorm/gridwise_gemm_multiple_d_welford_first_half_xdl_cshuffle.hpp`**
```
(is_same<ABDataType, int8_t>::value && lcm_AK1_BK1 <= 8) ||
((is_same<ABDataType, f8_t>::value || is_same<ABDataType, bf8_t>::value) &&
lcm_AK1_BK1 < 32))
constexpr auto is_scale_mfma = false;
```

**`include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_gemm_xdl_cshuffle_v1.hpp`**
```
(is_same<FloatAB, int8_t>::value && lcm_AK1_BK1 <= 8) ||
((is_same<FloatAB, f8_t>::value || is_same<FloatAB, bf8_t>::value) &&
lcm_AK1_BK1 < 32))
constexpr auto is_scale_mfma = false;
```

**`include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_multiple_d_gemm_multiple_d_xdl_cshuffle_v1.hpp`**
```
(is_same<A0B0B1DataType, int8_t>::value && lcm_A0K1_B0K1 <= 8) ||
((is_same<A0B0B1DataType, f8_t>::value || is_same<A0B0B1DataType, bf8_t>::value) &&
lcm_A0K1_B0K1 < 32))
constexpr auto is_scale_mfma = false;
```

**`include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_multiple_d_softmax_gemm_xdl_cshuffle_v1.hpp`**
```
(is_same<FloatAB, int8_t>::value && lcm_AK1_BK1 <= 8) ||
((is_same<FloatAB, f8_t>::value || is_same<FloatAB, bf8_t>::value) &&
lcm_AK1_BK1 < 32))
constexpr auto is_scale_mfma = false;
```
