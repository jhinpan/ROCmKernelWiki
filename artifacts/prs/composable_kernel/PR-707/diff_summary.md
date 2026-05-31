# Diff summary

- **files changed:** 20
- **lines:** +1262 / -470
- **kernel-ish files:** 18

## Files (by churn)

- `test/gemm_split_k/gemm_split_k.cpp`  (+0/-261)
- `test/grouped_gemm/test_grouped_gemm_util.hpp`  (+249/-0)
- `test/gemm_split_k/test_gemm_splitk_ut_cases.inc`  (+217/-0)
- `test/grouped_gemm/test_grouped_gemm_interface.cpp`  (+202/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r4r2.hpp`  (+112/-68)
- `test/grouped_gemm/test_grouped_gemm_ut_cases.inc`  (+180/-0)
- `profiler/include/profiler/profile_grouped_gemm_impl.hpp`  (+73/-45)
- `test/gemm_split_k/test_gemm_splitk_util.hpp`  (+78/-0)
- `test/grouped_gemm/grouped_gemm_fp16.cpp`  (+0/-69)
- `test/gemm_split_k/test_gemm_splitk.cpp`  (+66/-0)
- `test/grouped_gemm/test_grouped_gemm_splitk.cpp`  (+34/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl_splitk_cshuffle.hpp`  (+18/-10)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_splitk_c_shuffle.hpp`  (+11/-3)
- `test/grouped_gemm/CMakeLists.txt`  (+7/-3)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_irregular_instance.cpp`  (+3/-3)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_splitk_c_shuffle.hpp`**
```
static constexpr index_t NumGemmKPrefetchStage = 1;
static constexpr LoopScheduler LoopSched       = make_default_loop_scheduler();
static constexpr PipelineVersion PipelineVer   = PipelineVersion::v2;
NumGemmKPrefetchStage,
```

**`include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl_splitk_cshuffle.hpp`**
```
ck::index_t NumGemmKPrefetchStage,
NumGemmKPrefetchStage,
CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
LoopSched,
```

**`include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r4r2.hpp`**
```
index_t NumGemmKPrefetchStage,
typename CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
LoopScheduler LoopSched     = make_default_loop_scheduler(),
PipelineVersion PipelineVer = PipelineVersion::v1>
```

**`library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_xdl_f16_f16_f16_mk_kn_mn_instance.cpp`**
```
DeviceGroupedGemm_Xdl<    Row,    Row, Empty_Tuple,    Row,   F16,   F16,     F32,      F16, Empty_Tuple,   F16, PassThr
```

**`library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_irregular_instance.cpp`**
```
DeviceGroupedGemmXdlSplitKCShuffle<    Row,    Row, Empty_Tuple,    Row,   F16,   F16,     F32,      F16, Empty_Tuple,  
DeviceGroupedGemmXdlSplitKCShuffle<    Row,    Row, Empty_Tuple,    Row,   F16,   F16,     F32,      F16, Empty_Tuple,  
DeviceGroupedGemmXdlSplitKCShuffle<    Row,    Row, Empty_Tuple,    Row,   F16,   F16,     F32,      F16, Empty_Tuple,  
```
