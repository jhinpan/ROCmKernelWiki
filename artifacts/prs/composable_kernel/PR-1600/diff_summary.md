# Diff summary

- **files changed:** 43 (diff was byte-capped; summary is partial)
- **lines:** +1210 / -1190
- **kernel-ish files:** 42

## Files (by churn)

- `profiler/include/profiler/profile_grouped_gemm_two_stage_impl.hpp`  (+0/-294)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_tile_loop/device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_instance.cpp`  (+0/-234)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_gemm.hpp`  (+184/-1)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_xdl_splitk_instance.hpp`  (+138/-0)
- `include/ck/tensor_operation/gpu/device/device_grouped_gemm_multiple_d_splitk.hpp`  (+0/-136)
- `include/ck/tensor_operation/gpu/device/device_grouped_gemm.hpp`  (+131/-1)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_irregular_instance.cpp`  (+0/-123)
- `profiler/include/profiler/profile_grouped_gemm_impl.hpp`  (+67/-54)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_splitk_xdl_cshuffle_two_stage.hpp`  (+58/-35)
- `include/ck/tensor_operation/gpu/device/device_grouped_gemm_tile_loop.hpp`  (+6/-86)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl_fixed_nk.hpp`  (+55/-17)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_irregular_instance.cpp`  (+3/-52)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_instance.cpp`  (+4/-47)
- `include/ck/tensor_operation/gpu/device/device_grouped_gemm_fixed_nk.hpp`  (+13/-37)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_instance.cpp`  (+4/-43)

## Key added lines (kernel files)

**`client_example/31_grouped_gemm_bf16Aint8B/grouped_gemm_multiply_bias_fastgelu_xdl_bf16_i8.cpp`**
```
ck::tensor_operation::device::GroupedGemmKernelArgument<NumDTensor>;
```

**`client_example/31_grouped_gemm_bf16Aint8B/grouped_gemm_multiply_xdl_bf16_i8.cpp`**
```
ck::tensor_operation::device::GroupedGemmKernelArgument<NumDTensor>;
```

**`example/15_grouped_gemm/grouped_gemm_multiple_d_splitk_xdl_fp16.cpp`**
```
gemm.SetKBatchSize(&argument, config.k_batch);
gemm.SetDeviceKernelArgs(&argument, gemm_arg_dev_mem.GetDeviceBuffer());
```

**`example/15_grouped_gemm/grouped_gemm_multiple_d_xdl_fp16.cpp`**
```
using KernelArguments = ck::tensor_operation::device::GroupedGemmKernelArgument<NumDs>;
```

**`example/15_grouped_gemm/grouped_gemm_xdl_fixed_nk_bias_fp16.cpp`**
```
gemm.SetDeviceKernelArgs(&argument, gemm_kernel_args_dev.GetDeviceBuffer());
```
