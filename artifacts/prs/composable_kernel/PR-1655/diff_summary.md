# Diff summary

- **files changed:** 21
- **lines:** +2655 / -11
- **kernel-ish files:** 17

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_xdl_cshuffle_v3.hpp`  (+1014/-0)
- `example/24_batched_gemm/run_batched_gemm_example_rowwise.inc`  (+280/-0)
- `profiler/include/profiler/profile_gemm_universal_batched_impl.hpp`  (+280/-0)
- `profiler/src/profile_gemm_universal_batched.cpp`  (+187/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_universal_batched.hpp`  (+185/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_batched/device_batched_gemm_xdl_universal_f8_f8_bf16/device_batched_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn.hpp`  (+109/-0)
- `example/24_batched_gemm/batched_gemm_xdl_fp8_rowwise_v3.cpp`  (+106/-0)
- `example/24_batched_gemm/batched_gemm_xdl_bf16_v3.cpp`  (+99/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_batched/device_batched_gemm_xdl_universal_bf16_bf16_bf16/device_batched_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn.hpp`  (+95/-0)
- `include/ck/tensor_operation/gpu/device/device_batched_gemm_multi_d.hpp`  (+42/-1)
- `example/24_batched_gemm/run_batched_gemm_example.inc`  (+26/-10)
- `library/src/tensor_operation_instance/gpu/gemm_universal_batched/device_batched_gemm_xdl_universal_bf16_bf16_bf16/device_batched_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn_mem_v1_default_instance.cpp`  (+33/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_batched/device_batched_gemm_xdl_universal_bf16_bf16_bf16/device_batched_gemm_xdl_universal_bf16_bf16_bf16_mk_nk_mn_mem_v2_default_instance.cpp`  (+33/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_batched/device_batched_gemm_xdl_universal_f8_f8_bf16/device_batched_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_mem_v1_default_instance.cpp`  (+33/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_batched/device_batched_gemm_xdl_universal_f8_f8_bf16/device_batched_gemm_xdl_universal_f8_f8_bf16_mk_nk_mn_mem_v2_default_instance.cpp`  (+33/-0)

## Key added lines (kernel files)

**`example/24_batched_gemm/batched_gemm_xdl_bf16_v3.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using BF16 = ck::bhalf_t;
using F32  = float;
```

**`example/24_batched_gemm/batched_gemm_xdl_fp8_rowwise_v3.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F8   = ck::f8_t;
using BF16 = ck::bhalf_t;
```

**`example/24_batched_gemm/run_batched_gemm_example.inc`**
```
problem_size.K = 128 * (dis(gen) + 2);
problem_size.batch_count = 2;
else if(argc == 8)
config.do_verification   = std::stoi(argv[1]);
```

**`example/24_batched_gemm/run_batched_gemm_example_rowwise.inc`**
```
struct ProblemSize final
ck::index_t M = 3840;
ck::index_t N = 4096;
ck::index_t K = 4096;
```

**`include/ck/tensor_operation/gpu/device/device_batched_gemm_multi_d.hpp`**
```
template <typename ALayout,
typename BLayout,
typename DsLayout,
typename ELayout,
```
