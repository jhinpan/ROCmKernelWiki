# Diff summary

- **files changed:** 44 (diff was byte-capped; summary is partial)
- **lines:** +4199 / -12
- **kernel-ish files:** 41

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3r1.hpp`  (+703/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_universal_reduce.hpp`  (+457/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_reduce_threadwise_multi_d.hpp`  (+412/-0)
- `example/35_splitK_gemm/run_gemm_splitk_reduce_multi_d_example.inc`  (+309/-0)
- `example/12_reduce/reduce_threadwise_multi_d_impl.hpp`  (+307/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_2d_reduction_threadwise_multi_d.hpp`  (+260/-0)
- `example/12_reduce/reduce_threadwise_multi_d.cpp`  (+229/-0)
- `example/35_splitK_gemm/common.hpp`  (+101/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_reduce/device_gemm_xdl_universal_bf16_bf16_bf16/device_gemm_xdl_universal_bf16_bf16_bf16_mk_kn_mn.hpp`  (+99/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_reduce/device_gemm_xdl_universal_f16_f16_f16/device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn.hpp`  (+99/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_reduce/device_gemm_xdl_universal_bf16_i8_bf16/device_gemm_xdl_universal_bf16_i8_bf16_mk_kn_mn.hpp`  (+88/-0)
- `include/ck/tensor_operation/gpu/device/device_reduce_multi_d.hpp`  (+69/-0)
- `example/35_splitK_gemm/gemm_xdl_splitk_reduce_bf16.cpp`  (+58/-0)
- `example/35_splitK_gemm/gemm_xdl_splitk_reduce_bf16A_i8B.cpp`  (+58/-0)
- `example/35_splitK_gemm/gemm_xdl_splitk_reduce_multi_d_bf16.cpp`  (+58/-0)

## Key added lines (kernel files)

**`example/12_reduce/reduce_threadwise_multi_d.cpp`**
```
using namespace ck;
using namespace ck::tensor_operation::device;
static struct option long_options[] = {{"inLengths", required_argument, nullptr, 'D'},
{"verify", required_argument, nullptr, 'v'},
```

**`example/12_reduce/reduce_threadwise_multi_d_impl.hpp`**
```
template <typename InOutDataType,
typename AccDataType,
ck::ReduceTensorOp ReduceOpId,
ck::index_t Rank,
```

**`example/35_splitK_gemm/common.hpp`**
```
struct ProblemSizeSplitK final
ck::index_t M = 256;
ck::index_t N = 1024;
ck::index_t K = 512;
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
using ADataType        = ck::bhalf_t;
using BDataType        = int8_t;
using AccDataType      = float;
using CShuffleDataType = ck::bhalf_t;
```
