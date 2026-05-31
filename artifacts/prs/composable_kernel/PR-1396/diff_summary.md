# Diff summary

- **files changed:** 14
- **lines:** +278 / -433
- **kernel-ish files:** 12

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3r1.hpp`  (+168/-401)
- `example/35_splitK_gemm/gemm_xdl_splitk_reduce_bf16.cpp`  (+57/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_2d_reduction_threadwise_multi_d.hpp`  (+11/-8)
- `example/35_splitK_gemm/gemm_xdl_splitk_reduce_multi_d_fp16.cpp`  (+9/-5)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3.hpp`  (+9/-4)
- `example/35_splitK_gemm/gemm_xdl_splitk_reduce_multi_d_bf16.cpp`  (+8/-4)
- `example/35_splitK_gemm/run_gemm_splitk_reduce_multi_d_example.inc`  (+6/-5)
- `include/ck/tensor_operation/gpu/device/impl/device_reduce_threadwise_multi_d.hpp`  (+4/-2)
- `example/12_reduce/reduce_threadwise_multi_d_impl.hpp`  (+2/-1)
- `example/35_splitK_gemm/CMakeLists.txt`  (+2/-0)
- `example/35_splitK_gemm/gemm_xdl_splitk_reduce_bf16A_i8B.cpp`  (+1/-1)
- `cmake/gtest.cmake`  (+1/-0)
- `include/ck/tensor_operation/gpu/device/device_gemm_v2.hpp`  (+0/-1)
- `profiler/include/profiler/profile_gemm_universal_reduce_impl.hpp`  (+0/-1)

## Key added lines (kernel files)

**`example/12_reduce/reduce_threadwise_multi_d_impl.hpp`**
```
Sequence<1>>; // OutDstVectorSize
```

**`example/35_splitK_gemm/gemm_xdl_splitk_reduce_bf16.cpp`**
```
using ADataType        = ck::bhalf_t;
using BDataType        = ck::bhalf_t;
using AccDataType      = float;
using CShuffleDataType = ck::bhalf_t;
```

**`example/35_splitK_gemm/gemm_xdl_splitk_reduce_bf16A_i8B.cpp`**
```
AElementOp, BElementOp, CElementOp, GemmDefault,
```

**`example/35_splitK_gemm/gemm_xdl_splitk_reduce_multi_d_bf16.cpp`**
```
AElementOp, BElementOp, CDEElementOp, GemmDefault,
using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
BDataType,
CDataType,
```

**`example/35_splitK_gemm/gemm_xdl_splitk_reduce_multi_d_fp16.cpp`**
```
AElementOp, BElementOp, CDEElementOp, GemmDefault,
ck::BlockGemmPipelineScheduler::Intrawave,ck::BlockGemmPipelineVersion::v2>;
using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
BDataType,
```
