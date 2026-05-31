# Diff summary

- **files changed:** 15
- **lines:** +504 / -1422
- **kernel-ish files:** 14

## Files (by churn)

- `example/01_gemm/gemm_xdl_bf16.cpp`  (+21/-221)
- `example/01_gemm/gemm_xdl_int8.cpp`  (+18/-213)
- `example/01_gemm/gemm_xdl_fp64.cpp`  (+15/-208)
- `example/01_gemm/gemm_xdl_fp16.cpp`  (+20/-191)
- `example/01_gemm/gemm_dl_fp16.cpp`  (+12/-185)
- `example/01_gemm/gemm_dl_fp32.cpp`  (+12/-184)
- `example/01_gemm/gemm_dl_int8.cpp`  (+12/-182)
- `example/01_gemm/run_gemm_example.inc`  (+151/-0)
- `example/01_gemm/common.hpp`  (+89/-0)
- `example/01_gemm/gemm_xdl_int4.cpp`  (+46/-0)
- `example/01_gemm/gemm_dl_int4.cpp`  (+45/-0)
- `example/01_gemm/gemm_xdl_skip_b_lds_fp16.cpp`  (+15/-28)
- `example/01_gemm/CMakeLists.txt`  (+28/-0)
- `library/include/ck/library/utility/host_tensor.hpp`  (+11/-6)
- `library/include/ck/library/utility/check_err.hpp`  (+9/-4)

## Key added lines (kernel files)

**`example/01_gemm/common.hpp`**
```
struct ProblemSize final
ck::index_t M = 3840;
ck::index_t N = 4096;
ck::index_t K = 4096;
```

**`example/01_gemm/gemm_dl_fp16.cpp`**
```
using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmDl
```

**`example/01_gemm/gemm_dl_fp32.cpp`**
```
using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmDl
```

**`example/01_gemm/gemm_dl_int4.cpp`**
```
using ADataType       = ck::int4_t;
using BDataType       = ck::int4_t;
using CDataType       = ck::int4_t;
using KernelADataType = int8_t;
```

**`example/01_gemm/gemm_dl_int8.cpp`**
```
using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmDl
```
