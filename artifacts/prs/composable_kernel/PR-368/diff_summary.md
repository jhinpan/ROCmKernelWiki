# Diff summary

- **files changed:** 13
- **lines:** +2129 / -347
- **kernel-ish files:** 12

## Files (by churn)

- `example/16_gemm_multi_d_multi_reduces/gemm_reduce_xdl_common.hpp`  (+498/-0)
- `example/16_gemm_multi_d_multi_reduces/gemm_add_addsquare_xdl_int8.cpp`  (+368/-0)
- `example/16_gemm_multi_d_multi_reduces/gemm_mean_meansquare_xdl_fp16.cpp`  (+102/-182)
- `example/16_gemm_multi_d_multi_reduces/gemm_max_xdl_fp16.cpp`  (+100/-160)
- `example/16_gemm_multi_d_multi_reduces/gemm_mean_meansquare_xdl_bf16.cpp`  (+174/-0)
- `example/16_gemm_multi_d_multi_reduces/gemm_mean_meansquare_xdl_fp32.cpp`  (+174/-0)
- `example/16_gemm_multi_d_multi_reduces/gemm_max_xdl_int4.cpp`  (+172/-0)
- `example/16_gemm_multi_d_multi_reduces/gemm_max_xdl_bf16.cpp`  (+167/-0)
- `example/16_gemm_multi_d_multi_reduces/gemm_max_xdl_fp32.cpp`  (+166/-0)
- `example/16_gemm_multi_d_multi_reduces/gemm_max_xdl_int8.cpp`  (+166/-0)
- `example/16_gemm_multi_d_multi_reduces/CMakeLists.txt`  (+38/-1)
- `include/ck/utility/reduction_operator.hpp`  (+3/-3)
- `library/include/ck/library/utility/host_tensor.hpp`  (+1/-1)

## Key added lines (kernel files)

**`example/16_gemm_multi_d_multi_reduces/gemm_add_addsquare_xdl_int8.cpp`**
```
using ADataType         = INT8;
using BDataType         = INT8;
using GemmAccDataType   = INT32;
using CShuffleDataType  = INT32;
```

**`example/16_gemm_multi_d_multi_reduces/gemm_max_xdl_bf16.cpp`**
```
using ADataType         = BF16;
using BDataType         = BF16;
using GemmAccDataType   = F32;
using CShuffleDataType  = F32;
```

**`example/16_gemm_multi_d_multi_reduces/gemm_max_xdl_fp16.cpp`**
```
<ALayout,                   // ALayout
BLayout,                   // BLayout
ELayout,                   // ELayout
ADataType,                 // ADataType
```

**`example/16_gemm_multi_d_multi_reduces/gemm_max_xdl_fp32.cpp`**
```
using ADataType         = F32;
using BDataType         = F32;
using GemmAccDataType   = F32;
using CShuffleDataType  = F32;
```

**`example/16_gemm_multi_d_multi_reduces/gemm_max_xdl_int4.cpp`**
```
using ADataType         = INT4;
using ADataKernelType   = INT8;
using BDataType         = INT4;
using BDataKernelType   = INT8;
```
