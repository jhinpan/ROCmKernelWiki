# Diff summary

- **files changed:** 235
- **lines:** +964 / -22
- **kernel-ish files:** 235

## Files (by churn)

- `library/include/ck/library/reference_tensor_operation/gpu/reference_gemm.hpp`  (+15/-14)
- `include/ck/host_utility/io.hpp`  (+8/-4)
- `example/62_convnd_activ/binary/convnd_fwd_xdl_bilinear_residual_fp16.cpp`  (+8/-0)
- `library/src/utility/device_memory.cpp`  (+5/-1)
- `library/src/utility/host_tensor.cpp`  (+5/-1)
- `example/01_gemm/common.hpp`  (+5/-0)
- `example/15_grouped_gemm/grouped_gemm_multiple_d_dl_fp16.cpp`  (+5/-0)
- `example/15_grouped_gemm/grouped_gemm_multiple_d_splitk_xdl_fp16.cpp`  (+5/-0)
- `example/15_grouped_gemm/grouped_gemm_multiple_d_xdl_fp16.cpp`  (+5/-0)
- `example/15_grouped_gemm/grouped_gemm_xdl_bf16.cpp`  (+5/-0)
- `example/15_grouped_gemm/grouped_gemm_xdl_fixed_nk_bias_fp16.cpp`  (+5/-0)
- `example/15_grouped_gemm/grouped_gemm_xdl_fixed_nk_fp16.cpp`  (+5/-0)
- `example/15_grouped_gemm/grouped_gemm_xdl_fixed_nk_fp16_fp8.cpp`  (+5/-0)
- `example/15_grouped_gemm/grouped_gemm_xdl_fp16.cpp`  (+5/-0)
- `example/15_grouped_gemm/grouped_gemm_xdl_fp32.cpp`  (+5/-0)

## Key added lines (kernel files)

**`client_example/24_grouped_conv_activation/grouped_convnd_fwd_convscale_reduce/common.hpp`**
```
using ::ck::HostTensorDescriptor;
```

**`client_example/31_grouped_gemm_bf16Aint8B/grouped_gemm_bias_fastgelu_xdl_bf16_i8.cpp`**
```
using ::ck::hip_check_error;
```

**`client_example/31_grouped_gemm_bf16Aint8B/grouped_gemm_fastgelu_xdl_bf16_i8.cpp`**
```
using ::ck::hip_check_error;
```

**`client_example/31_grouped_gemm_bf16Aint8B/grouped_gemm_multiply_bias_fastgelu_xdl_bf16_i8.cpp`**
```
using ::ck::hip_check_error;
```

**`client_example/31_grouped_gemm_bf16Aint8B/grouped_gemm_multiply_xdl_bf16_i8.cpp`**
```
using ::ck::hip_check_error;
```
