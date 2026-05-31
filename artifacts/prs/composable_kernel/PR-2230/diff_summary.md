# Diff summary

- **files changed:** 39 (diff was byte-capped; summary is partial)
- **lines:** +2725 / -139
- **kernel-ish files:** 36

## Files (by churn)

- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v1.hpp`  (+638/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_universal_wmma.inc`  (+302/-2)
- `example/01_gemm/gemm_wmma_fp16_pk_i4_v3.cpp`  (+302/-0)
- `example/01_gemm/gemm_wmma_bf16_pk_i4_v3.cpp`  (+253/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_universal.hpp`  (+182/-1)
- `library/src/tensor_operation_instance/gpu/gemm_universal/CMakeLists.txt`  (+168/-1)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v3.hpp`  (+44/-44)
- `example/01_gemm/gemm_wmma_fp8_v3.cpp`  (+67/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_wmma_universal_bf16_i4_bf16/device_gemm_wmma_universal_bf16_i4_bf16_mk_nk_mn.hpp`  (+59/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_wmma_universal_bf16_i4_bf16/device_gemm_wmma_universal_bf16_i4_bf16_km_nk_mn.hpp`  (+58/-0)
- `example/01_gemm/gemm_wmma_fp16_fp8_v3.cpp`  (+52/-0)
- `example/01_gemm/gemm_wmma_bf16_v3.cpp`  (+47/-0)
- `example/01_gemm/gemm_wmma_fp16_v3.cpp`  (+47/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3.hpp`  (+30/-8)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_base.hpp`  (+14/-14)

## Key added lines (kernel files)

**`example/01_gemm/gemm_wmma_bf16_pk_i4_v3.cpp`**
```
using ADataType        = ck::bhalf_t;
using BDataType        = ck::pk_i4_t;
using AccDataType      = float;
using CShuffleDataType = ck::bhalf_t;
```

**`example/01_gemm/gemm_wmma_bf16_v3.cpp`**
```
using ADataType        = ck::bhalf_t;
using BDataType        = ck::bhalf_t;
using AccDataType      = float;
using CShuffleDataType = ck::bhalf_t;
```

**`example/01_gemm/gemm_wmma_fp16_fp8_v3.cpp`**
```
using ADataType        = ck::half_t;
using BDataType        = ck::f8_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
```

**`example/01_gemm/gemm_wmma_fp16_pk_i4_v3.cpp`**
```
using ADataType        = ck::half_t;
using BDataType        = ck::pk_i4_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
```

**`example/01_gemm/gemm_wmma_fp16_v3.cpp`**
```
using ADataType        = ck::half_t;
using BDataType        = ck::half_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
```
