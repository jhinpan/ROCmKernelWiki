# Diff summary

- **files changed:** 46 (diff was byte-capped; summary is partial)
- **lines:** +844 / -330
- **kernel-ish files:** 45

## Files (by churn)

- `example/01_gemm/run_gemm_example_streamk.inc`  (+270/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_comp_instance.hpp`  (+94/-23)
- `example/30_grouped_conv_fwd_multiple_d/run_grouped_conv_fwd_bias_relu_add_example.inc`  (+64/-38)
- `example/01_gemm/run_gemm_example.inc`  (+1/-77)
- `include/ck/tensor_operation/gpu/warp/xdlops_gemm.hpp`  (+33/-23)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_merged_groups_instance.hpp`  (+41/-15)
- `library/src/tensor_operation_instance/gpu/batched_gemm_softmax_gemm_permute/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance.cpp`  (+42/-4)
- `library/src/tensor_operation_instance/gpu/batched_gemm/device_batched_gemm_xdl_f16_f16_f16_gmk_gkn_gmn_instance.cpp`  (+32/-8)
- `library/src/tensor_operation_instance/gpu/batched_gemm/device_batched_gemm_xdl_f16_f16_f16_gmk_gnk_gmn_instance.cpp`  (+32/-8)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_two_stage_xdl_instance.hpp`  (+8/-24)
- `library/src/tensor_operation_instance/gpu/batched_gemm/device_batched_gemm_xdl_f16_f16_f16_gkm_gkn_gmn_instance.cpp`  (+19/-4)
- `library/src/tensor_operation_instance/gpu/batched_gemm/device_batched_gemm_xdl_f16_f16_f16_gkm_gnk_gmn_instance.cpp`  (+19/-4)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_abd_xdl_cshuffle.hpp`  (+16/-4)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle.hpp`  (+16/-4)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle_lds_direct_load.hpp`  (+16/-4)

## Key added lines (kernel files)

**`example/01_gemm/gemm_xdl_streamk.cpp`**
```
using DeviceGemmStreamK2 = ck::tensor_operation::device::DeviceGemmXdlStreamK
< ADataType, BDataType, CDataType, AccDataType, ALayout, BLayout, CLayout,  AElementOp,  BElementOp,  CElementOp,    256
using DeviceGemmInstance  = DeviceGemmStreamK;
using DeviceGemmInstance2 = DeviceGemmStreamK2;
```

**`example/01_gemm/run_gemm_example.inc`**
```
if constexpr(std::is_same<ProblemType, ProblemSize>::value)
```

**`example/01_gemm/run_gemm_example_streamk.inc`**
```
template <typename ProblemType>
bool run_gemm(const ProblemType& problem_size, const ExecutionConfig& config)
static_assert(sizeof(ck::int4_t) == sizeof(int8_t));
using namespace ck::literals;
```

**`example/30_grouped_conv_fwd_multiple_d/run_grouped_conv_fwd_bias_relu_add_example.inc`**
```
using DeviceConvFwdInstance2 =
64,          // NPerBlock
1,           // NXdlPerWave
128,         // NPerBlock
```

**`include/ck/tensor_operation/gpu/grid/gemm_layernorm/gridwise_gemm_multiple_d_welford_first_half_xdl_cshuffle.hpp`**
```
(((is_same<ABDataType, half_t>::value || is_same<ABDataType, bhalf_t>::value) &&
lcm_AK1_BK1 <= 4) ||
(is_same<ABDataType, int8_t>::value && lcm_AK1_BK1 <= 8))
```
