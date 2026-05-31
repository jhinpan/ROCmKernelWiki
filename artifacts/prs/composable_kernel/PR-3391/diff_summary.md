# Diff summary

- **files changed:** 22
- **lines:** +2957 / -500
- **kernel-ish files:** 18

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_gemm_multiple_d_wmma_cshuffle_v3.hpp`  (+1072/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_gemm_wmma_cshuffle_v3.hpp`  (+428/-103)
- `example/37_batched_gemm_add_add_relu_gemm_add/batched_gemm_add_add_relu_gemm_add_xdl_fp16.cpp`  (+7/-390)
- `profiler/include/profiler/profile_batched_gemm_multiple_d_gemm_multiple_d_impl.hpp`  (+387/-0)
- `example/37_batched_gemm_add_add_relu_gemm_add/batched_gemm_multiple_d_gemm_multiple_d.inc`  (+350/-0)
- `example/37_batched_gemm_add_add_relu_gemm_add/batched_gemm_add_add_relu_gemm_add_wmma_fp16.cpp`  (+135/-0)
- `test/batched_gemm_multiple_d_gemm_multiple_d/test_batched_gemm_multiple_d_gemm_multiple_d.hpp`  (+121/-0)
- `test/batched_gemm_multiple_d_gemm_multiple_d/test_batched_gemm_multiple_d_gemm_multiple_d_ut_cases.inc`  (+88/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_add_relu_gemm_add/device_batched_gemm_add_relu_gemm_add_wmma_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance.cpp`  (+72/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_add_relu_gemm_add/device_batched_gemm_add_relu_gemm_add_wmma_cshuffle_f16_f16_f16_f16_gmk_gnk_gon_gmo_instance.cpp`  (+72/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/batched_gemm_add_relu_gemm_add.hpp`  (+62/-1)
- `example/37_batched_gemm_add_add_relu_gemm_add/element_ops.h`  (+58/-0)
- `test/batched_gemm_multiple_d_gemm_multiple_d/test_batched_gemm_add_relu_gemm_add.cpp`  (+27/-0)
- `include/ck/tensor_operation/operator_transform/transform_contraction_to_gemm.hpp`  (+24/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_gemm_wmma_cshuffle_v3.hpp`  (+16/-4)

## Key added lines (kernel files)

**`example/37_batched_gemm_add_add_relu_gemm_add/batched_gemm_add_add_relu_gemm_add_wmma_fp16.cpp`**
```
Computes C_m_o = Relu(A0[m, k] * B0[n, k] + D00[m, n] + D01[mn]) * B1[n, o] + D1[m, o]
using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;
```

**`example/37_batched_gemm_add_add_relu_gemm_add/batched_gemm_add_add_relu_gemm_add_xdl_fp16.cpp`**
```
using AccDataType       = F32;
AccDataType,
AccDataType,
int main(int argc, char* argv[]) { return run_example(argc, argv); }
```

**`example/37_batched_gemm_add_add_relu_gemm_add/batched_gemm_multiple_d_gemm_multiple_d.inc`**
```
int run_example(int argc, char* argv[])
bool do_verification = true;
int init_method      = 1;
bool time_kernel     = false;
```

**`example/37_batched_gemm_add_add_relu_gemm_add/element_ops.h`**
```
struct AddAddRelu
__host__ __device__ void
operator()(ck::half_t& e, const ck::half_t& c, const ck::half_t& d0, const ck::half_t& d1) const
const ck::half_t x = c + d0 + d1;
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_gemm_wmma_cshuffle_v3.hpp`**
```
Tuple<>{}, // p_d0s_grid
Tuple<>{}, // p_d1s_grid
Tuple<>{}, // D0sGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
Tuple<>{}, // D1sGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
```
