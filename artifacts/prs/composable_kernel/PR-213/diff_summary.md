# Diff summary

- **files changed:** 19 (diff was byte-capped; summary is partial)
- **lines:** +364 / -391
- **kernel-ish files:** 19

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_reduce_xdl_cshuffle_v1.hpp`  (+39/-26)
- `include/ck/tensor_operation/gpu/device/device_batched_gemm_reduce_xdl_cshuffle.hpp`  (+18/-32)
- `example/16_gemm_reduce/gemm_reduce_xdl_fp16.cpp`  (+28/-21)
- `example/18_batched_gemm_reduce/batched_gemm_reduce_xdl_fp16.cpp`  (+26/-21)
- `library/src/tensor_operation_instance/gpu/batched_gemm_reduce/device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gkm_gkn_gmn_instance.cpp`  (+22/-24)
- `library/src/tensor_operation_instance/gpu/batched_gemm_reduce/device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gkm_gnk_gmn_instance.cpp`  (+22/-24)
- `library/src/tensor_operation_instance/gpu/batched_gemm_reduce/device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gmk_gkn_gmn_instance.cpp`  (+22/-24)
- `library/src/tensor_operation_instance/gpu/gemm_reduce/device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_km_kn_mn_instance.cpp`  (+22/-24)
- `library/src/tensor_operation_instance/gpu/gemm_reduce/device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_km_nk_mn_instance.cpp`  (+22/-24)
- `library/src/tensor_operation_instance/gpu/gemm_reduce/device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_mk_kn_mn_instance.cpp`  (+22/-24)
- `profiler/include/profile_batched_gemm_reduce_impl.hpp`  (+25/-21)
- `library/src/tensor_operation_instance/gpu/batched_gemm_reduce/device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gmk_gnk_gmn_instance.cpp`  (+19/-21)
- `library/src/tensor_operation_instance/gpu/gemm_reduce/device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_mk_nk_mn_instance.cpp`  (+19/-21)
- `include/ck/tensor_operation/gpu/device/device_gemm_reduce_xdl_cshuffle.hpp`  (+14/-24)
- `profiler/include/profile_gemm_reduce_impl.hpp`  (+20/-17)

## Key added lines (kernel files)

**`example/16_gemm_reduce/gemm_reduce_xdl_fp16.cpp`**
```
using AElementOp  = ck::tensor_operation::element_wise::PassThrough;
using BElementOp  = ck::tensor_operation::element_wise::PassThrough;
using CElementOp  = ck::tensor_operation::element_wise::PassThrough;
using D0ReduceOp  = ck::reduce::Add<float>;
```

**`example/18_batched_gemm_reduce/batched_gemm_reduce_xdl_fp16.cpp`**
```
using AElementOp  = ck::tensor_operation::element_wise::PassThrough;
using BElementOp  = ck::tensor_operation::element_wise::PassThrough;
using CElementOp  = ck::tensor_operation::element_wise::PassThrough;
using D0ReduceOp  = ck::reduce::Add<float>;
```

**`include/ck/tensor_operation/gpu/device/device_batched_gemm_reduce_xdl_cshuffle.hpp`**
```
typename D1ElementwiseOperation,
const D1ElementwiseOperation d1_element_op,
d1_element_op,
ignore = d1_element_op;
```

**`include/ck/tensor_operation/gpu/device/device_batched_gemm_xdl.hpp`**
```
ignore = compute_ptr_offset_of_batch;
b_grid_desc_k0_n_k1_.GetElementSpaceSize(),
c_grid_desc_m_n_.GetElementSpaceSize()},
```

**`include/ck/tensor_operation/gpu/device/device_gemm_reduce.hpp`**
```
typename D1ElementwiseOperation>
D1ElementwiseOperation d1_element_op,
typename D1ElementwiseOperation>
D1ElementwiseOperation>>;
```
