# Diff summary

- **files changed:** 57 (diff was byte-capped; summary is partial)
- **lines:** +2855 / -230
- **kernel-ish files:** 52

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_streamk_v3.hpp`  (+733/-85)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_streamk_v3.hpp`  (+309/-73)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_universal_streamk.hpp`  (+315/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_f16_f8_f16/device_gemm_xdl_universal_streamk_f16_f8_f16_mk_nk_mn.hpp`  (+90/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_f8_f16_f16/device_gemm_xdl_universal_streamk_f8_f16_f16_mk_nk_mn.hpp`  (+90/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_f8_f16_f16/device_gemm_xdl_universal_streamk_f8_f16_f16_mk_kn_mn.hpp`  (+85/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_f16_f8_f16/device_gemm_xdl_universal_streamk_f16_f8_f16_mk_kn_mn.hpp`  (+84/-0)
- `example/01_gemm/gemm_xdl_fp8_streamk_v3.cpp`  (+58/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/CMakeLists.txt`  (+44/-1)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/gemm_universal_streamk/device_gemm_xdl_universal_streamk_f16_f16_f16/device_gemm_xdl_universal_streamk_f16_f16_f16_mk_kn_mn.hpp`  (+0/-41)
- `example/01_gemm/run_gemm_example_streamk_v2.inc`  (+40/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/gemm_universal_streamk/CMakeLists.txt`  (+0/-26)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_f16_f8_f16/device_gemm_xdl_universal_streamk_f16_f8_f16_mk_kn_mn_mem_v1_default_instance.cpp`  (+25/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_f16_f8_f16/device_gemm_xdl_universal_streamk_f16_f8_f16_mk_kn_mn_mem_v1_kpadding_instance.cpp`  (+25/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_f16_f8_f16/device_gemm_xdl_universal_streamk_f16_f8_f16_mk_kn_mn_mem_v1_mnkpadding_instance.cpp`  (+25/-0)

## Key added lines (kernel files)

**`example/01_gemm/common.hpp`**
```
ck::index_t NumSKBlocks = -1; // number of stream-k blocks
```

**`example/01_gemm/gemm_xdl_fp16_streamk_v3.cpp`**
```
using CShuffleDataType = float;
using ReferenceGemmInstanceGPU = ck::tensor_operation::device::ReferenceGemm<ALayout,
ADataType,
BDataType,
```

**`example/01_gemm/gemm_xdl_fp8_streamk_v3.cpp`**
```
using ADataType        = ck::f8_t;
using BDataType        = ck::f8_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
```

**`example/01_gemm/run_gemm_example_streamk_v2.inc`**
```
Tensor<CDataType> c_m_n_device_ref_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
DeviceMem c_m_n_device_ref_buf(sizeof(CDataType) *
c_m_n_device_ref_result.mDesc.GetElementSpaceSize());
std::size_t workspace_size = gemm.GetWorkSpaceSize(&argument);
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_streamk_v3.hpp`**
```
if constexpr(GridwiseGemm::Block2CTileMap_streamk::ReductionStrategy ==
StreamKReductionStrategy::Atomic)
hip_check_error(hipMemsetAsync(
arg.p_c_grid, 0, arg.M * arg.N * sizeof(CDataType), stream_config.stream_id_));
```
