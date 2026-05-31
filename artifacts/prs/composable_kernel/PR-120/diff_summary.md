# Diff summary

- **files changed:** 23
- **lines:** +1312 / -899
- **kernel-ish files:** 20

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_xdlops_v2r3.hpp`  (+0/-649)
- `include/ck/tensor_operation/gpu/device/device_batched_gemm_xdl.hpp`  (+279/-160)
- `profiler/src/profile_batched_gemm.cpp`  (+162/-3)
- `test/batched_gemm/batched_gemm_fp16.cpp`  (+137/-0)
- `test/batched_gemm/batched_gemm_util.hpp`  (+106/-0)
- `profiler/include/profile_batched_gemm_impl.hpp`  (+76/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm/device_batched_gemm_xdl_int8_int8_int8_gkm_gkn_gmn_instance.cpp`  (+66/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm/device_batched_gemm_xdl_int8_int8_int8_gkm_gnk_gmn_instance.cpp`  (+66/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm/device_batched_gemm_xdl_int8_int8_int8_gmk_gkn_gmn_instance.cpp`  (+66/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm/device_batched_gemm_xdl_int8_int8_int8_gmk_gnk_gmn_instance.cpp`  (+58/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm/device_batched_gemm_xdl_f32_f32_f32_gmk_gnk_gmn_instance.cpp`  (+56/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm/device_batched_gemm_xdl_f32_f32_f32_gkm_gkn_gmn_instance.cpp`  (+51/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm/device_batched_gemm_xdl_f32_f32_f32_gkm_gnk_gmn_instance.cpp`  (+51/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm/device_batched_gemm_xdl_f32_f32_f32_gmk_gkn_gmn_instance.cpp`  (+51/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm/device_batched_gemm_xdl_f16_f16_f16_gmk_gkn_gmn_instance.cpp`  (+25/-21)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/device/device_batched_gemm_xdl.hpp`**
```
template <typename GridwiseGemm,
typename FloatAB,
typename FloatC,
typename AGridDesc_K0_M_K1,
```

**`library/include/ck/library/host_tensor/host_tensor.hpp`**
```
float check_error(const Tensor<T>& ref, const Tensor<T>& result)
return max_diff;
```

**`library/include/ck/library/host_tensor/host_tensor_generator.hpp`**
```
float min_value = 0;
float max_value = 1;
```

**`library/src/tensor_operation_instance/gpu/batched_gemm/device_batched_gemm_xdl_f16_f16_f16_gkm_gkn_gmn_instance.cpp`**
```
using device_batched_gemm_xdl_f16_f16_f16_gkm_gkn_gmn_instances = std::tuple<
DeviceBatchedGemmXdl<  F16,   F16,   F16,     F32,     Col,      Row,    Row, PassThrough, PassThrough, PassThrough,   2
DeviceBatchedGemmXdl<  F16,   F16,   F16,     F32,     Col,      Row,    Row, PassThrough, PassThrough, PassThrough,   2
DeviceBatchedGemmXdl<  F16,   F16,   F16,     F32,     Col,      Row,    Row, PassThrough, PassThrough, PassThrough,   1
```

**`library/src/tensor_operation_instance/gpu/batched_gemm/device_batched_gemm_xdl_f16_f16_f16_gkm_gnk_gmn_instance.cpp`**
```
using device_batched_gemm_xdl_f16_f16_f16_gkm_gnk_gmn_instances = std::tuple<
DeviceBatchedGemmXdl<  F16,   F16,   F16,     F32,     Col,      Col,    Row, PassThrough, PassThrough, PassThrough,   2
DeviceBatchedGemmXdl<  F16,   F16,   F16,     F32,     Col,      Col,    Row, PassThrough, PassThrough, PassThrough,   2
DeviceBatchedGemmXdl<  F16,   F16,   F16,     F32,     Col,      Col,    Row, PassThrough, PassThrough, PassThrough,   1
```
