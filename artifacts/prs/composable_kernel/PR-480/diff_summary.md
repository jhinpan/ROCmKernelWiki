# Diff summary

- **files changed:** 17
- **lines:** +816 / -297
- **kernel-ish files:** 16

## Files (by churn)

- `test/gemm/gemm_standalone_xdl_fp16.cpp`  (+162/-0)
- `test/gemm/gemm_util.hpp`  (+65/-42)
- `test/gemm/instance/gemm_f16_nn_instance.cpp`  (+86/-0)
- `test/gemm/instance/gemm_f16_nt_instance.cpp`  (+86/-0)
- `test/gemm/instance/gemm_f16_tn_instance.cpp`  (+86/-0)
- `test/gemm/instance/gemm_f16_tt_instance.cpp`  (+86/-0)
- `test/gemm/gemm_bf16.cpp`  (+6/-51)
- `test/gemm/gemm_fp16.cpp`  (+6/-51)
- `test/gemm/gemm_fp32.cpp`  (+6/-51)
- `test/gemm/gemm_fp64.cpp`  (+6/-51)
- `test/gemm/gemm_int8.cpp`  (+6/-51)
- `test/gemm/instance/gemm_f16_nn_instance.hpp`  (+41/-0)
- `test/gemm/instance/gemm_f16_nt_instance.hpp`  (+41/-0)
- `test/gemm/instance/gemm_f16_tn_instance.hpp`  (+41/-0)
- `test/gemm/instance/gemm_f16_tt_instance.hpp`  (+41/-0)

## Key added lines (kernel files)

**`test/gemm/gemm_bf16.cpp`**
```
using ADataType   = ck::bhalf_t;
using BDataType   = ck::bhalf_t;
using CDataType   = ck::bhalf_t;
using AccDataType = float;
```

**`test/gemm/gemm_fp16.cpp`**
```
using ADataType   = ck::half_t;
using BDataType   = ck::half_t;
using CDataType   = ck::half_t;
using AccDataType = float;
```

**`test/gemm/gemm_fp32.cpp`**
```
using ADataType   = float;
using BDataType   = float;
using CDataType   = float;
using AccDataType = float;
```

**`test/gemm/gemm_fp64.cpp`**
```
using ADataType   = double;
using BDataType   = double;
using CDataType   = double;
using AccDataType = double;
```

**`test/gemm/gemm_int8.cpp`**
```
using ADataType   = int8_t;
using BDataType   = int8_t;
using CDataType   = int8_t;
using AccDataType = int32_t;
```
