# Diff summary

- **files changed:** 21
- **lines:** +518 / -39
- **kernel-ish files:** 21

## Files (by churn)

- `library/include/ck/library/reference_tensor_operation/gpu/reference_gemm.hpp`  (+245/-0)
- `example/01_gemm/run_gemm_example.inc`  (+40/-6)
- `example/01_gemm/common.hpp`  (+17/-16)
- `example/01_gemm/gemm_xdl_bf16.cpp`  (+15/-1)
- `example/01_gemm/gemm_xdl_bf16_rtn.cpp`  (+15/-1)
- `example/01_gemm/gemm_xdl_fp8.cpp`  (+14/-0)
- `example/01_gemm/gemm_dl_fp16.cpp`  (+12/-1)
- `example/01_gemm/gemm_dl_fp32.cpp`  (+12/-1)
- `example/01_gemm/gemm_dl_int8.cpp`  (+12/-1)
- `example/01_gemm/gemm_wmma_fp16.cpp`  (+12/-1)
- `example/01_gemm/gemm_xdl_fp16.cpp`  (+12/-1)
- `example/01_gemm/gemm_xdl_fp16_fp8.cpp`  (+12/-1)
- `example/01_gemm/gemm_xdl_fp16_v2.cpp`  (+12/-1)
- `example/01_gemm/gemm_xdl_fp64.cpp`  (+12/-1)
- `example/01_gemm/gemm_xdl_fp8_bf8.cpp`  (+12/-1)

## Key added lines (kernel files)

**`example/01_gemm/common.hpp`**
```
ck::index_t StrideA = 0;
ck::index_t StrideB = 0;
ck::index_t StrideC = 0;
ck::index_t StrideA = 0;
```

**`example/01_gemm/gemm_dl_fp16.cpp`**
```
using ReferenceGemmInstanceGPU = ck::tensor_operation::device::ReferenceGemm<ALayout,
ADataType,
BDataType,
CDataType,
```

**`example/01_gemm/gemm_dl_fp32.cpp`**
```
using ReferenceGemmInstanceGPU = ck::tensor_operation::device::ReferenceGemm<ALayout,
ADataType,
BDataType,
CDataType,
```

**`example/01_gemm/gemm_dl_int8.cpp`**
```
using ReferenceGemmInstanceGPU = ck::tensor_operation::device::ReferenceGemm<ALayout,
ADataType,
BDataType,
CDataType,
```

**`example/01_gemm/gemm_dpp_fp16.cpp`**
```
using ReferenceGemmInstanceGPU = ck::tensor_operation::device::
ReferenceGemm<ALayout, BLayout, CLayout, ADataType, BDataType, CDataType, AccDataType, AElementOp, BElementOp, CElementO
```
