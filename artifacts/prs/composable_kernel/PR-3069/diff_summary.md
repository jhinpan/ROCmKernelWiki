# Diff summary

- **files changed:** 33
- **lines:** +195 / -199
- **kernel-ish files:** 33

## Files (by churn)

- `example/60_gemm_multi_ABD/gemm_multi_ABD_xdl_bias_fastgelu_bf16_i8.cpp`  (+15/-22)
- `example/60_gemm_multi_ABD/gemm_multi_ABD_xdl_fastgelu_bf16_i8.cpp`  (+15/-22)
- `example/60_gemm_multi_ABD/gemm_multi_ABD_xdl_multiply_bias_fastgelu_bf16_i8.cpp`  (+15/-22)
- `example/46_gemm_add_multiply/run_gemm_add_multiply_example.inc`  (+13/-19)
- `example/65_gemm_multiply_multiply/moe_gemm2_xdl_pk_i4.cpp`  (+10/-14)
- `example/04_gemm_add_add_fastgelu/run_gemm_add_add_fastgelu_example.inc`  (+9/-8)
- `example/03_gemm_bias_relu/gemm_bias_relu_xdl_fp16.cpp`  (+8/-8)
- `example/02_gemm_bilinear/gemm_bilinear_xdl_fp16.cpp`  (+6/-5)
- `example/65_gemm_multiply_multiply/moe_gemm1_xdl_pk_i4.cpp`  (+6/-5)
- `example/02_gemm_bilinear/gemm_bilinear_wmma_fp16.cpp`  (+5/-4)
- `example/02_gemm_bilinear/gemm_bilinear_wmma_int8.cpp`  (+5/-4)
- `example/18_batched_gemm_reduce/batched_gemm_reduce_xdl_fp16.cpp`  (+5/-4)
- `example/59_grouped_gemm_multi_ABD/grouped_gemm_multi_abd_xdl_fixed_nk_bias_bf16_i8.cpp`  (+5/-4)
- `example/59_grouped_gemm_multi_ABD/grouped_gemm_multi_abd_xdl_fixed_nk_bias_fp16.cpp`  (+5/-4)
- `example/65_gemm_multiply_multiply/gemm_multiply_multiply_xdl_fp8.cpp`  (+5/-4)

## Key added lines (kernel files)

**`example/02_gemm_bilinear/gemm_bilinear_wmma_fp16.cpp`**
```
using Row    = ck::tensor_layout::gemm::RowMajor;
using Col    = ck::tensor_layout::gemm::ColumnMajor;
using Bypass = ck::tensor_layout::BypassLayoutVerification;
return HostTensorDescriptor({row, col}, {stride, 1_uz}, Bypass{});
```

**`example/02_gemm_bilinear/gemm_bilinear_wmma_int8.cpp`**
```
using Row    = ck::tensor_layout::gemm::RowMajor;
using Col    = ck::tensor_layout::gemm::ColumnMajor;
using Bypass = ck::tensor_layout::BypassLayoutVerification;
return HostTensorDescriptor({row, col}, {stride, 1_uz}, Bypass{});
```

**`example/02_gemm_bilinear/gemm_bilinear_xdl_fp16.cpp`**
```
using Row    = ck::tensor_layout::gemm::RowMajor;
using Col    = ck::tensor_layout::gemm::ColumnMajor;
using Bypass = ck::tensor_layout::BypassLayoutVerification;
return HostTensorDescriptor({row, col}, {stride, 1_uz}, Bypass{});
```

**`example/03_gemm_bias_relu/gemm_bias_relu_xdl_fp16.cpp`**
```
using Row    = ck::tensor_layout::gemm::RowMajor;
using Col    = ck::tensor_layout::gemm::ColumnMajor;
using Bypass = ck::tensor_layout::BypassLayoutVerification;
return HostTensorDescriptor({row, col}, {stride, 1_uz}, Bypass{});
```

**`example/04_gemm_add_add_fastgelu/run_gemm_add_add_fastgelu_example.inc`**
```
using Bypass = ck::tensor_layout::BypassLayoutVerification;
return HostTensorDescriptor({row, col}, {stride, 1_uz}, Bypass{});
return HostTensorDescriptor({row, col}, {1_uz, stride}, Bypass{});
if(StrideA < 0)
```
