# Diff summary

- **files changed:** 30
- **lines:** +2291 / -268
- **kernel-ish files:** 25

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_wmma_splitk_cshuffle_v3.hpp`  (+827/-0)
- `test/grouped_gemm/test_grouped_gemm_util.hpp`  (+41/-206)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_gemm.hpp`  (+211/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_wmma_splitk_instance.hpp`  (+205/-0)
- `test/grouped_gemm/test_grouped_gemm_interface_xdl.hpp`  (+205/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3.hpp`  (+90/-21)
- `example/15_grouped_gemm/grouped_gemm_wmma_splitk_bf16.cpp`  (+72/-0)
- `example/15_grouped_gemm/grouped_gemm_wmma_splitk_fp16.cpp`  (+71/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_wmma_universal_f16_f8_f16_mk_kn_mn_instance.cpp`  (+57/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_wmma_universal_f8_f16_f16_mk_kn_mn_instance.cpp`  (+57/-0)
- `library/src/tensor_operation_instance/gpu/CMakeLists.txt`  (+22/-20)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_wmma_universal_f16_f16_f16_mk_kn_mn_instance.cpp`  (+38/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_wmma_universal_f16_f16_f16_mk_nk_mn_instance.cpp`  (+38/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_wmma_universal_bf16_bf16_bf16_km_kn_mn_instance.cpp`  (+37/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_wmma_universal_bf16_bf16_bf16_km_nk_mn_instance.cpp`  (+37/-0)

## Key added lines (kernel files)

**`example/15_grouped_gemm/grouped_gemm_wmma_splitk_bf16.cpp`**
```
using ::ck::DeviceMem;
using ::ck::hip_check_error;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;
```

**`example/15_grouped_gemm/grouped_gemm_wmma_splitk_fp16.cpp`**
```
using ::ck::DeviceMem;
using ::ck::hip_check_error;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;
```

**`example/15_grouped_gemm/run_grouped_gemm_example.inc`**
```
ck::index_t k_batch;
gemm.SetKBatchSize(&argument, problem_size.k_batch);
problem_size.k_batch = 1;
else if(argc == 4 || argc == 6 || argc == 7)
```

**`include/ck/tensor_operation/gpu/device/device_base.hpp`**
```
virtual __host__ __device__ ~BaseArgument() {}
```

**`include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_wmma_splitk_cshuffle_v3.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <typename GridwiseGemm,
```
