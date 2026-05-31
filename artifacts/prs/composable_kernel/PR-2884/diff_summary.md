# Diff summary

- **files changed:** 22
- **lines:** +74 / -65
- **kernel-ish files:** 22

## Files (by churn)

- `example/32_batched_gemm_scale_softmax_gemm/grouped_gemm_lower_triangle_scale_softmax_gemm_permute_xdl_fp16.cpp`  (+7/-7)
- `example/22_cgemm/cgemm_xdl_bf16.cpp`  (+6/-6)
- `example/22_cgemm/cgemm_xdl_fp16.cpp`  (+6/-6)
- `example/46_gemm_add_multiply/run_gemm_add_multiply_example.inc`  (+10/-1)
- `example/01_gemm/gemm_xdl_fp16_v2.cpp`  (+4/-4)
- `example/35_splitK_gemm/gemm_xdl_splitk_reduce_bf16.cpp`  (+4/-4)
- `example/35_splitK_gemm/gemm_xdl_splitk_reduce_bf16A_i8B.cpp`  (+4/-4)
- `example/35_splitK_gemm/gemm_xdl_splitk_reduce_multi_d_bf16.cpp`  (+4/-4)
- `example/35_splitK_gemm/gemm_xdl_splitk_reduce_multi_d_fp16.cpp`  (+4/-4)
- `example/01_gemm/gemm_xdl_fp16.cpp`  (+2/-2)
- `example/01_gemm/gemm_xdl_wavelet_fp16.cpp`  (+2/-2)
- `example/04_gemm_add_add_fastgelu/gemm_add_add_fastgelu_xdl_bf16.cpp`  (+2/-2)
- `example/04_gemm_add_add_fastgelu/gemm_add_add_fastgelu_xdl_fp16.cpp`  (+2/-2)
- `example/04_gemm_add_add_fastgelu/gemm_add_add_fastgelu_xdl_fp32.cpp`  (+2/-2)
- `example/15_grouped_gemm/grouped_gemm_xdl_bf16.cpp`  (+2/-2)

## Key added lines (kernel files)

**`example/01_gemm/gemm_xdl_fp16.cpp`**
```
< ALayout, BLayout, CLayout, ADataType, BDataType, CDataType, AccDataType, CShuffleDataType,  AElementOp,  BElementOp,  
```

**`example/01_gemm/gemm_xdl_fp16_v2.cpp`**
```
16,   16,
1, 1, S<1, 32, 1, 8>, 4,
```

**`example/01_gemm/gemm_xdl_wavelet_fp16.cpp`**
```
< ALayout, BLayout, CLayout, ADataType, BDataType, AccDataType,              F16, CDataType,  AElementOp,  BElementOp,  
```

**`example/04_gemm_add_add_fastgelu/gemm_add_add_fastgelu_xdl_bf16.cpp`**
```
< ALayout, BLayout, DsLayout, ELayout, ADataType, BDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AEle
```

**`example/04_gemm_add_add_fastgelu/gemm_add_add_fastgelu_xdl_fp16.cpp`**
```
< ALayout, BLayout, DsLayout, ELayout, ADataType, BDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AEle
```
