# Diff summary

- **files changed:** 32
- **lines:** +356 / -175
- **kernel-ish files:** 19

## Files (by churn)

- `library/src/tensor_operation_instance/gpu/gemm/CMakeLists.txt`  (+89/-80)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm.hpp`  (+24/-22)
- `library/src/tensor_operation_instance/gpu/batched_gemm_multi_d/CMakeLists.txt`  (+22/-18)
- `library/src/tensor_operation_instance/gpu/CMakeLists.txt`  (+36/-3)
- `library/src/tensor_operation_instance/gpu/conv2d_bwd_data/CMakeLists.txt`  (+17/-10)
- `CMakeLists.txt`  (+25/-0)
- `client_example/CMakeLists.txt`  (+25/-0)
- `test/gemm/CMakeLists.txt`  (+14/-10)
- `library/include/ck/library/tensor_operation_instance/gpu/convolution_backward_data.hpp`  (+14/-7)
- `Jenkinsfile`  (+16/-0)
- `example/01_gemm/CMakeLists.txt`  (+10/-4)
- `profiler/src/profile_gemm.cpp`  (+11/-3)
- `profiler/src/profile_conv_bwd_data.cpp`  (+8/-0)
- `profiler/src/profile_batched_gemm_multi_d.cpp`  (+5/-1)
- `test/batched_gemm_multi_d/test_batched_gemm_multi_d.cpp`  (+4/-2)

## Key added lines (kernel files)

**`library/include/ck/library/tensor_operation_instance/gpu/gemm.hpp`**
```
void add_device_gemm_xdl_c_shuffle_i8_i8_i8_km_kn_mn_instances(
std::vector<std::unique_ptr<
DeviceGemm<Col, Row, Row, int8_t, int8_t, int8_t, PassThrough, PassThrough, PassThrough>>>&
instances);
```

**`profiler/src/profile_batched_gemm_multi_d.cpp`**
```
using F16 = ck::half_t;
```

**`profiler/src/profile_gemm.cpp`**
```
using F32 = float;
using F16 = ck::half_t;
using BF16 = ck::bhalf_t;
```
