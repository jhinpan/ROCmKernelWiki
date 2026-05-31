# Diff summary

- **files changed:** 9
- **lines:** +30 / -22
- **kernel-ish files:** 9

## Files (by churn)

- `example/14_gemm_quantization/gemm_wmma_quantization_int8.cpp`  (+7/-6)
- `example/65_gemm_multiply_multiply/gemm_add_add_wmma_fp16.cpp`  (+5/-4)
- `example/65_gemm_multiply_multiply/run_gemm_multiply_multiply_wp_example.inc`  (+4/-2)
- `example/68_gemm_add/run_gemm_add_example_wmma.inc`  (+3/-2)
- `example/68_gemm_add/run_gemm_add_example_xdl.inc`  (+3/-2)
- `example/69_gemm_add_relu/run_gemm_add_relu_example_wmma.inc`  (+3/-2)
- `example/69_gemm_add_relu/run_gemm_add_relu_example_xdl.inc`  (+3/-2)
- `example/68_gemm_add/common.hpp`  (+1/-1)
- `example/69_gemm_add_relu/common.hpp`  (+1/-1)

## Key added lines (kernel files)

**`example/14_gemm_quantization/gemm_wmma_quantization_int8.cpp`**
```
using I8     = int8_t;
using I32    = int32_t;
using Row    = ck::tensor_layout::gemm::RowMajor;
using Col    = ck::tensor_layout::gemm::ColumnMajor;
```

**`example/65_gemm_multiply_multiply/gemm_add_add_wmma_fp16.cpp`**
```
using Row    = ck::tensor_layout::gemm::RowMajor;
using Col    = ck::tensor_layout::gemm::ColumnMajor;
using Bypass = ck::tensor_layout::BypassLayoutVerification;
return HostTensorDescriptor({row, col}, {stride, 1_uz}, Bypass{});
```

**`example/65_gemm_multiply_multiply/run_gemm_multiply_multiply_wp_example.inc`**
```
using Bypass = ck::tensor_layout::BypassLayoutVerification;
return ck::HostTensorDescriptor({row, col}, {stride, 1_uz}, Bypass{});
return ck::HostTensorDescriptor({row, col}, {1_uz, stride}, Bypass{});
```

**`example/68_gemm_add/common.hpp`**
```
else if(argc == 11)
```

**`example/68_gemm_add/run_gemm_add_example_wmma.inc`**
```
using Bypass = ck::tensor_layout::BypassLayoutVerification;
return HostTensorDescriptor({row, col}, {stride, 1_uz}, Bypass{});
return HostTensorDescriptor({row, col}, {1_uz, stride}, Bypass{});
```
