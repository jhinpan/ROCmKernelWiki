# Diff summary

- **files changed:** 27
- **lines:** +1911 / -89
- **kernel-ish files:** 22

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3r1.hpp`  (+562/-0)
- `example/35_splitK_gemm/run_gemm_wmma_splitk_reduce_multi_d_example.inc`  (+214/-0)
- `example/35_splitK_gemm/run_gemm_wmma_splitk_reduce_example.inc`  (+191/-0)
- `example/35_splitK_gemm/common.hpp`  (+82/-0)
- `example/35_splitK_gemm/run_gemm_splitk_reduce_multi_d_example.inc`  (+0/-82)
- `library/src/tensor_operation_instance/gpu/gemm_universal_reduce/device_gemm_wmma_universal_bf16_i8_bf16/device_gemm_wmma_universal_bf16_i8_bf16_mk_kn_mn.hpp`  (+73/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_universal_reduce.hpp`  (+71/-1)
- `library/src/tensor_operation_instance/gpu/gemm_universal_reduce/device_gemm_wmma_universal_bf16_bf16_bf16/device_gemm_wmma_universal_bf16_bf16_bf16_mk_kn_mn.hpp`  (+72/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_reduce/device_gemm_wmma_universal_f16_f16_f16/device_gemm_wmma_universal_f16_f16_f16_mk_kn_mn.hpp`  (+72/-0)
- `example/35_splitK_gemm/gemm_wmma_splitk_reduce_bf16.cpp`  (+59/-0)
- `example/35_splitK_gemm/gemm_wmma_splitk_reduce_bf16A_i8B.cpp`  (+59/-0)
- `example/35_splitK_gemm/gemm_wmma_splitk_reduce_multi_d_bf16.cpp`  (+59/-0)
- `example/35_splitK_gemm/gemm_wmma_splitk_reduce_multi_d_fp16.cpp`  (+59/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_reduce/device_gemm_wmma_universal_bf16_i8_bf16/device_gemm_wmma_universal_bf16_i8_bf16_mk_kn_mn_comp_default_instance.cpp`  (+59/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_reduce/device_gemm_wmma_universal_bf16_bf16_bf16/device_gemm_wmma_universal_bf16_bf16_bf16_mk_kn_mn_comp_default_instance.cpp`  (+58/-0)

## Key added lines (kernel files)

**`example/35_splitK_gemm/common.hpp`**
```
template <typename DataType>
inline __host__ __device__ constexpr double get_rtol()
if constexpr(std::is_same_v<DataType, float>)
return 1e-3;
```

**`example/35_splitK_gemm/gemm_wmma_splitk_reduce_bf16.cpp`**
```
using ADataType        = ck::bhalf_t;
using BDataType        = ck::bhalf_t;
using AccDataType      = float;
using CShuffleDataType = ck::bhalf_t;
```

**`example/35_splitK_gemm/gemm_wmma_splitk_reduce_bf16A_i8B.cpp`**
```
using ADataType        = ck::bhalf_t;
using BDataType        = int8_t;
using AccDataType      = float;
using CShuffleDataType = ck::bhalf_t;
```

**`example/35_splitK_gemm/gemm_wmma_splitk_reduce_multi_d_bf16.cpp`**
```
using ADataType        = ck::bhalf_t;
using BDataType        = ck::bhalf_t;
using AccDataType      = float;
using CShuffleDataType = ck::bhalf_t;
```

**`example/35_splitK_gemm/gemm_wmma_splitk_reduce_multi_d_fp16.cpp`**
```
using ADataType        = ck::half_t;
using BDataType        = ck::half_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
```
