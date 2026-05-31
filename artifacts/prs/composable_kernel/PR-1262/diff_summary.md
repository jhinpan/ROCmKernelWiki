# Diff summary

- **files changed:** 37
- **lines:** +1235 / -174
- **kernel-ish files:** 35

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp`  (+162/-137)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_universal.hpp`  (+159/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn.hpp`  (+101/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn.hpp`  (+91/-0)
- `example/01_gemm/run_gemm_example_v2.inc`  (+87/-1)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v3.hpp`  (+40/-17)
- `example/01_gemm/gemm_xdl_bf16_v3.cpp`  (+48/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_mem_v1_mnkpadding_instance.cpp`  (+25/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_mem_v2_mnkpadding_instance.cpp`  (+25/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn_mem_v1_mnkpadding_instance.cpp`  (+25/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn_mem_v2_mnkpadding_instance.cpp`  (+25/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_comp_kpadding_instance.cpp`  (+24/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_comp_mnkpadding_instance.cpp`  (+24/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_comp_mnpadding_instance.cpp`  (+24/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn_mem_v1_default_instance.cpp`  (+24/-0)

## Key added lines (kernel files)

**`example/01_gemm/gemm_xdl_bf16_v3.cpp`**
```
using ADataType        = ck::bhalf_t;
using BDataType        = ck::bhalf_t;
using AccDataType      = float;
using CShuffleDataType = ck::bhalf_t;
```

**`example/01_gemm/run_gemm_example_v2.inc`**
```
template <typename DataType>
inline __host__ __device__ constexpr double get_rtol()
if constexpr(std::is_same_v<DataType, float>)
return 1e-3;
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v2.hpp`**
```
static constexpr index_t WgpPerCU =
(4 * warpSize / BlockSize) >= 1 ? 4 * warpSize / BlockSize : 1;
32768 / WgpPerCU,
static constexpr index_t WgpPerCU =
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v3.hpp`**
```
(mfma_cycle - 4 + 2 * ds_read_a_issue_cycle - 1) / (2 * ds_read_a_issue_cycle);
(mfma_cycle - 4 + 2 * ds_read_b_issue_cycle - 1) / (2 * ds_read_b_issue_cycle);
constexpr auto num_dsread_a_mfma =
(num_ds_read_inst_a + ds_read_a_mfma_rate - 1) / ds_read_a_mfma_rate;
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v5.hpp`**
```
(mfma_cycle - 4 + 2 * ds_read_a_issue_cycle - 1) / (2 * ds_read_a_issue_cycle);
(mfma_cycle - 4 + 2 * ds_read_b_issue_cycle - 1) / (2 * ds_read_b_issue_cycle);
```
