# Diff summary

- **files changed:** 35
- **lines:** +1048 / -75
- **kernel-ish files:** 34

## Files (by churn)

- `library/include/ck/library/tensor_operation_instance/gpu/gemm_universal.hpp`  (+151/-0)
- `test/gemm_universal/test_gemm_universal_ut_cases.inc`  (+128/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_km_kn_mn.hpp`  (+84/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_km_nk_mn.hpp`  (+84/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn.hpp`  (+11/-25)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn.hpp`  (+11/-19)
- `library/src/tensor_operation_instance/gpu/gemm_universal/CMakeLists.txt`  (+28/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_km_kn_mn_mem_v1_mnkpadding_instance.cpp`  (+25/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_km_kn_mn_mem_v2_mnkpadding_instance.cpp`  (+25/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_km_nk_mn_mem_v1_mkpadding_instance.cpp`  (+25/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_km_nk_mn_mem_v2_mkpadding_instance.cpp`  (+25/-0)
- `test/gemm_universal/test_gemm_universal_xdl.cpp`  (+25/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_km_kn_mn_comp_kpadding_instance.cpp`  (+24/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_km_kn_mn_comp_mnkpadding_instance.cpp`  (+24/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_km_kn_mn_comp_mnpadding_instance.cpp`  (+24/-0)

## Key added lines (kernel files)

**`example/01_gemm/run_gemm_example_v2.inc`**
```
ave_time =
invoker.Run(argument, StreamConfig{nullptr, config.time_kernel, 0, 5, 10, true, 4});
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3.hpp`**
```
const auto a_grid_desc_ak0_m_ak1 = GridwiseGemm::MakeAGridDescriptor_AK0_M_AK1(
arg_.M, arg_.MPadded, arg_.K, arg_.KPadded, arg_.StrideA, arg_.AK0);
const auto b_grid_desc_bk0_n_bk1 = GridwiseGemm::MakeBGridDescriptor_BK0_N_BK1(
arg_.K, arg_.KPadded, arg_.N, arg_.NPadded, arg_.StrideB, arg_.BK0);
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp`**
```
const auto a_grid_desc_ak0_m_ak1 = GridwiseGemm::MakeAGridDescriptor_AK0_M_AK1(
arg_.M, arg_.MPadded, arg_.K, arg_.KPadded, arg_.StrideA, arg_.AK0);
const auto b_grid_desc_bk0_n_bk1 = GridwiseGemm::MakeBGridDescriptor_BK0_N_BK1(
arg_.K, arg_.KPadded, arg_.N, arg_.NPadded, arg_.StrideB, arg_.BK0);
```

**`include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3.hpp`**
```
__host__ __device__ static auto MakeAGridDescriptor_AK0_M_AK1(
__host__ __device__ static auto MakeBGridDescriptor_BK0_N_BK1(
a_k_split_offset = blockIdx.z * karg.KRead * karg.StrideA;
b_k_split_offset = blockIdx.z * karg.KRead * karg.StrideB;
```

**`include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_multi_d.hpp`**
```
__host__ __device__ static auto MakeAGridDescriptor_AK0_M_AK1(
__host__ __device__ static auto MakeBGridDescriptor_BK0_N_BK1(
a_k_split_offset = blockIdx.z * karg.KRead * karg.StrideA;
b_k_split_offset = blockIdx.z * karg.KRead * karg.StrideB;
```
