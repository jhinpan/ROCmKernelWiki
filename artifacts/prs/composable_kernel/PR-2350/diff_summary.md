# Diff summary

- **files changed:** 14 (diff was byte-capped; summary is partial)
- **lines:** +3437 / -1646
- **kernel-ish files:** 12

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3.hpp`  (+181/-1352)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3_common.hpp`  (+1420/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3_b_scale.hpp`  (+551/-0)
- `example/01_gemm/gemm_wmma_fp16_pk_i4_v3_b_scale.cpp`  (+367/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3_b_scale.hpp`  (+302/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3_common.hpp`  (+265/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3.hpp`  (+20/-213)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v1.hpp`  (+111/-44)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v3.hpp`  (+67/-32)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_base.hpp`  (+74/-2)
- `library/src/tensor_operation_instance/gpu/gemm_b_scale/device_gemm_b_scale_wmma_f16_i4_f16/device_gemm_b_scale_wmma_f16_i4_f16_mk_nk_mn.hpp`  (+50/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_b_scale.hpp`  (+23/-1)
- `library/src/tensor_operation_instance/gpu/gemm_b_scale/CMakeLists.txt`  (+4/-2)
- `example/01_gemm/CMakeLists.txt`  (+2/-0)

## Key added lines (kernel files)

**`example/01_gemm/gemm_wmma_fp16_pk_i4_v3_b_scale.cpp`**
```
using ADataType        = ck::half_t;
using BDataType        = ck::pk_i4_t;
using BScaleDataType   = ck::half_t;
using AccDataType      = float;
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_base.hpp`**
```
struct Empty
__device__ Empty(){};
template <index_t NBuffer>
__device__ void GlobalLoad(bool cond)
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v1.hpp`**
```
using typename Base::Empty;
typename CThreadBuffer,
typename BScaleStruct>
BScaleStruct& b_scale_struct,
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v3.hpp`**
```
using typename Base::Empty;
template <typename ABlockBuffer,
typename AThreadBuffer,
typename BBlockBuffer,
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3.hpp`**
```
using DeviceGemmCommon = DeviceGemm_Wmma_CShuffleV3_Common<GridwiseGemm,
ADataType,
BDataType,
CDataType,
```
