# Diff summary

- **files changed:** 20
- **lines:** +1229 / -14
- **kernel-ish files:** 17

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3_b_preshuffle.hpp`  (+303/-0)
- `example/01_gemm/run_gemm_wmma_bpreshuffle_example.inc`  (+206/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_preshuffle/device_gemm_wmma_universal_preshuffle_f8_f8_bf16/device_gemm_wmma_universal_preshuffle_f8_f8_bf16_mk_wmma_mn.hpp`  (+106/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_preshuffle/device_gemm_wmma_universal_preshuffle_f8_f8_f16/device_gemm_wmma_universal_preshuffle_f8_f8_f16_mk_wmma_mn.hpp`  (+106/-0)
- `example/01_gemm/gemm_wmma_fp8_bpreshuffle.cpp`  (+72/-0)
- `example/01_gemm/gemm_wmma_fp16_bpreshuffle.cpp`  (+70/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_universal_preshuffle.inc`  (+45/-2)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_universal_preshuffle.hpp`  (+34/-9)
- `library/src/tensor_operation_instance/gpu/gemm_universal_preshuffle/device_gemm_wmma_universal_preshuffle_f8_f8_bf16/device_gemm_wmma_universal_preshuffle_f8_f8_bf16_mk_wmma_mn_default_instance_p1.cpp`  (+33/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_preshuffle/device_gemm_wmma_universal_preshuffle_f8_f8_bf16/device_gemm_wmma_universal_preshuffle_f8_f8_bf16_mk_wmma_mn_default_instance_p2.cpp`  (+33/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_preshuffle/device_gemm_wmma_universal_preshuffle_f8_f8_bf16/device_gemm_wmma_universal_preshuffle_f8_f8_bf16_mk_wmma_mn_default_instance_p3.cpp`  (+33/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_preshuffle/device_gemm_wmma_universal_preshuffle_f8_f8_bf16/device_gemm_wmma_universal_preshuffle_f8_f8_bf16_mk_wmma_mn_default_instance_p4.cpp`  (+33/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_preshuffle/device_gemm_wmma_universal_preshuffle_f8_f8_f16/device_gemm_wmma_universal_preshuffle_f8_f8_f16_mk_wmma_mn_default_instance_p1.cpp`  (+33/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_preshuffle/device_gemm_wmma_universal_preshuffle_f8_f8_f16/device_gemm_wmma_universal_preshuffle_f8_f8_f16_mk_wmma_mn_default_instance_p2.cpp`  (+33/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_preshuffle/device_gemm_wmma_universal_preshuffle_f8_f8_f16/device_gemm_wmma_universal_preshuffle_f8_f8_f16_mk_wmma_mn_default_instance_p3.cpp`  (+33/-0)

## Key added lines (kernel files)

**`example/01_gemm/gemm_wmma_fp16_bpreshuffle.cpp`**
```
using F16 = ck::half_t;
using F32 = float;
using ADataType        = F16;
using BDataType        = F16;
```

**`example/01_gemm/gemm_wmma_fp8_bpreshuffle.cpp`**
```
using F8  = ck::f8_t;
using F16 = ck::half_t;
using F32 = float;
using ADataType        = F8;
```

**`example/01_gemm/run_gemm_wmma_bpreshuffle_example.inc`**
```
template <typename ProblemType>
bool run_gemm(const ProblemType& problem_size, const ExecutionConfig& config)
using namespace ck::literals;
auto M       = problem_size.M;
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3_b_preshuffle.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <typename ALayout,
```

**`library/include/ck/library/tensor_operation_instance/gpu/gemm_universal_preshuffle.hpp`**
```
add_device_gemm_universal_preshuffle_wmma_f8_f8_bf16_mk_wmma_mn_default_instances_p1(
op_ptrs);
add_device_gemm_universal_preshuffle_wmma_f8_f8_bf16_mk_wmma_mn_default_instances_p2(
op_ptrs);
```
