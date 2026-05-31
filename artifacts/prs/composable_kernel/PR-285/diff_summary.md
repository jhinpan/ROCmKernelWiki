# Diff summary

- **files changed:** 24 (diff was byte-capped; summary is partial)
- **lines:** +1629 / -1095
- **kernel-ish files:** 22

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/device_gemm_bias_add_reduce_xdl_cshuffle.hpp`  (+193/-138)
- `include/ck/tensor_operation/gpu/device/device_batched_gemm_reduce_xdl_cshuffle.hpp`  (+185/-115)
- `client_example/03_gemm_layernorm/gemm_add_add_layernorm.cpp`  (+270/-0)
- `include/ck/tensor_operation/gpu/device/device_gemm_reduce_xdl_cshuffle.hpp`  (+166/-95)
- `example/21_gemm_layernorm/gemm_bias_relu_add_layernorm_xdl_fp16.cpp`  (+111/-108)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_bias_add_reduce_xdl_cshuffle_v1.hpp`  (+99/-99)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_reduce_xdl_cshuffle_v1.hpp`  (+84/-84)
- `example/21_gemm_layernorm/gemm_layernorm_xdl_fp16.cpp`  (+86/-79)
- `example/16_gemm_reduce/gemm_reduce_xdl_mean_squaremean_fp16.cpp`  (+74/-67)
- `example/18_batched_gemm_reduce/batched_gemm_reduce_xdl_fp16.cpp`  (+62/-52)
- `include/ck/tensor_operation/gpu/device/device_5ary_elementwise.hpp`  (+56/-42)
- `example/16_gemm_reduce/gemm_reduce_xdl_max_fp16.cpp`  (+47/-42)
- `include/ck/tensor_operation/gpu/device/device_gemm_reduce.hpp`  (+11/-68)
- `include/ck/tensor_operation/gpu/device/device_batched_gemm_reduce.hpp`  (+0/-54)
- `library/include/ck/library/tensor_operation_instance/gpu/device_elementwise_instance.hpp`  (+49/-0)

## Key added lines (kernel files)

**`client_example/03_gemm_layernorm/gemm_add_add_layernorm.cpp`**
```
using F16 = ck::half_t;
using F32 = float;
using ADataType            = F16;
using BDataType            = F16;
```

**`example/16_gemm_reduce/gemm_reduce_xdl_max_fp16.cpp`**
```
using ReduceDataType    = F64;
using ReducePtrsGlobal  = ck::Tuple<ReduceDataType*>;
using AElementOp       = ck::tensor_operation::element_wise::PassThrough;
using BElementOp       = ck::tensor_operation::element_wise::PassThrough;
```

**`example/16_gemm_reduce/gemm_reduce_xdl_mean_squaremean_fp16.cpp`**
```
using ReduceDataType    = F32;
using ReducePtrsGlobal  = ck::Tuple<ReduceDataType*, ReduceDataType*>;
using AElementOp = ck::tensor_operation::element_wise::PassThrough;
using BElementOp = ck::tensor_operation::element_wise::PassThrough;
```

**`example/18_batched_gemm_reduce/batched_gemm_reduce_xdl_fp16.cpp`**
```
using ReduceDataType    = F32;
using ReducePtrsGlobal  = ck::Tuple<ReduceDataType*, ReduceDataType*>;
using AElementOp = ck::tensor_operation::element_wise::PassThrough;
using BElementOp = ck::tensor_operation::element_wise::PassThrough;
```

**`example/19_binary_elementwise/broadcast_add_2d_amn_bn.cpp`**
```
std::array<const void*, 2> input = {a_m_n_device_buf.GetDeviceBuffer(),
b_n_device_buf.GetDeviceBuffer()};
std::array<void*, 1> output      = {c_m_n_device_buf.GetDeviceBuffer()};
std::vector<ck::index_t> a_strides = {Stride, 1};
```
