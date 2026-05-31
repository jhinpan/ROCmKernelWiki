# Diff summary

- **files changed:** 34 (diff was byte-capped; summary is partial)
- **lines:** +2533 / -876
- **kernel-ish files:** 31

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_wmma_cshuffle_tile_loop_v3.hpp`  (+689/-0)
- `example/15_grouped_gemm/run_grouped_gemm_multiple_d_example.inc`  (+341/-0)
- `example/15_grouped_gemm/grouped_gemm_multiple_d_xdl_fp16.cpp`  (+2/-335)
- `profiler/include/profiler/profile_grouped_gemm_multiply_tile_loop_impl.hpp`  (+28/-303)
- `profiler/include/profiler/profile_grouped_gemm_tile_loop_generic_impl.hpp`  (+329/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_gemm_tile_loop/device_grouped_gemm_tile_loop_wmma_instance.hpp`  (+215/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_xdl_cshuffle_tile_loop.hpp`  (+59/-119)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_gemm_tile_loop/device_grouped_gemm_tile_loop_multiply_wmma_instance.hpp`  (+159/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_gemm_tile_loop_multiply.hpp`  (+88/-2)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_wmma_splitk_instance.hpp`  (+40/-43)
- `example/15_grouped_gemm/grouped_gemm_multiple_d_wmma_fp16.cpp`  (+76/-0)
- `include/ck/tensor_operation/gpu/device/device_grouped_gemm_tile_loop.hpp`  (+58/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_tile_loop/device_grouped_gemm_wmma_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn_instance.cpp`  (+48/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_tile_loop/device_grouped_gemm_wmma_tile_loop_multiply_bias_bf16_i8_bf16_mk_kn_mn_instance.cpp`  (+48/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_tile_loop/device_grouped_gemm_wmma_tile_loop_multiply_bias_fastgelu_bf16_i8_bf16_mk_kn_mn_instance.cpp`  (+48/-0)

## Key added lines (kernel files)

**`example/15_grouped_gemm/grouped_gemm_multiple_d_wmma_fp16.cpp`**
```
using ::ck::DeviceMem;
using ::ck::hip_check_error;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;
```

**`example/15_grouped_gemm/grouped_gemm_multiple_d_xdl_fp16.cpp`**
```
int main(int argc, char* argv[]) { return !run_grouped_gemm_example(argc, argv); }
```

**`example/15_grouped_gemm/grouped_gemm_wmma_splitk_bf16.cpp`**
```
< ALayout, BLayout, DsLayout, ELayout, ADataType, BDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AEle
```

**`example/15_grouped_gemm/grouped_gemm_wmma_splitk_fp16.cpp`**
```
< ALayout, BLayout, DsLayout, ELayout, ADataType, BDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AEle
```

**`example/15_grouped_gemm/run_grouped_gemm_example.inc`**
```
printf("arg3: time kernel (0=no, 1=yes)\n");
printf("arg4: async hargs (0=no, 1=yes)\n");
```
