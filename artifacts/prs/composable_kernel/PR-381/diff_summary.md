# Diff summary

- **files changed:** 28 (diff was byte-capped; summary is partial)
- **lines:** +566 / -202
- **kernel-ish files:** 28

## Files (by churn)

- `library/src/tensor_operation_instance/gpu/batched_gemm/device_batched_gemm_xdl_f16_f16_f16_gmk_gkn_gmn_instance.cpp`  (+64/-21)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_f16_f16_f16_km_kn_mn_instance.cpp`  (+61/-20)
- `library/src/tensor_operation_instance/gpu/batched_gemm/device_batched_gemm_xdl_f16_f16_f16_gmk_gnk_gmn_instance.cpp`  (+52/-17)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_2_stage_f16_f16_f16_mk_nk_mn_instance.cpp`  (+52/-17)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_f16_f16_f16_km_nk_mn_instance.cpp`  (+49/-20)
- `library/src/tensor_operation_instance/gpu/batched_gemm/device_batched_gemm_xdl_f16_f16_f16_gkm_gkn_gmn_instance.cpp`  (+37/-12)
- `library/src/tensor_operation_instance/gpu/batched_gemm/device_batched_gemm_xdl_f16_f16_f16_gkm_gnk_gmn_instance.cpp`  (+37/-12)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp`  (+43/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r3.hpp`  (+18/-15)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_xdl.hpp`  (+21/-3)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl.hpp`  (+19/-3)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle.hpp`  (+15/-3)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp`  (+15/-3)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v1.hpp`  (+5/-10)
- `include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_gemm_xdl_cshuffle_v1.hpp`  (+8/-5)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_xdl.hpp`**
```
ck::index_t CThreadTransferDstScalarPerVector,
ck::index_t NumGemmKPrefetchStage = 1,
ck::LoopScheduler LoopSched       = make_default_loop_scheduler(),
ck::PipelineVersion PipelineVer   = ck::PipelineVersion::v1>
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle.hpp`**
```
LoopScheduler LoopSched     = make_default_loop_scheduler(),
PipelineVersion PipelineVer = PipelineVersion::v1>
LoopSched,
PipelineVer>;
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl.hpp`**
```
ck::index_t NumPrefetch         = 1,
ck::LoopScheduler LoopSched     = make_default_loop_scheduler(),
ck::PipelineVersion PipelineVer = ck::PipelineVersion::v1>
NumPrefetch,
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp`**
```
LoopScheduler LoopSched     = make_default_loop_scheduler(),
PipelineVersion PipelineVer = PipelineVersion::v1>
LoopSched,
PipelineVer>;
```

**`include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_gemm_xdl_cshuffle_v1.hpp`**
```
LoopScheduler LoopSched,
PipelineVersion PipelineVer = PipelineVersion::v1>
using GridwiseGemmPipe = remove_cvref_t<decltype(
GridwiseGemmPipeline_Selector<PipelineVer, NumGemmKPrefetchStage>())>;
```
