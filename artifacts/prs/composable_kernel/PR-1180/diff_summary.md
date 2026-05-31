# Diff summary

- **files changed:** 66
- **lines:** +82 / -80
- **kernel-ish files:** 66

## Files (by churn)

- `client_example/03_gemm_layernorm/gemm_add_add_layernorm_naive.cpp`  (+4/-2)
- `client_example/01_gemm/gemm.cpp`  (+2/-2)
- `client_example/02_gemm_add_add_fastgelu/gemm_add_add_fastgelu.cpp`  (+2/-2)
- `client_example/02_gemm_add_add_fastgelu/gemm_add_add_fastgelu_generic.cpp`  (+2/-2)
- `client_example/02_gemm_add_add_fastgelu/gemm_add_fastgelu.cpp`  (+2/-2)
- `client_example/02_gemm_add_add_fastgelu/gemm_add_fastgelu_generic.cpp`  (+2/-2)
- `client_example/02_gemm_add_add_fastgelu/gemm_fastgelu.cpp`  (+2/-2)
- `client_example/02_gemm_add_add_fastgelu/gemm_fastgelu_generic.cpp`  (+2/-2)
- `client_example/03_gemm_layernorm/gemm_add_relu_add_layernorm_welford.cpp`  (+2/-2)
- `client_example/20_splitk_gemm/splitK_gemm_fp16_f8.cpp`  (+2/-2)
- `client_example/21_grouped_gemm_bias/grouped_gemm_fixed_nk_bias_fp16.cpp`  (+2/-2)
- `client_example/22_grouped_gemm/grouped_gemm_fixed_nk_fp16.cpp`  (+2/-2)
- `client_example/22_grouped_gemm/grouped_gemm_fixed_nk_fp8.cpp`  (+2/-2)
- `client_example/22_grouped_gemm/grouped_gemm_fixed_nk_i8.cpp`  (+2/-2)
- `client_example/04_contraction/contraction_bilinear_fp32.cpp`  (+1/-1)

## Key added lines (kernel files)

**`client_example/01_gemm/gemm.cpp`**
```
if constexpr(std::is_same<Layout, Row>::value)
```

**`client_example/02_gemm_add_add_fastgelu/gemm_add_add_fastgelu.cpp`**
```
if constexpr(std::is_same<Layout, Row>::value)
```

**`client_example/02_gemm_add_add_fastgelu/gemm_add_add_fastgelu_generic.cpp`**
```
if constexpr(std::is_same<Layout, Row>::value)
```

**`client_example/02_gemm_add_add_fastgelu/gemm_add_fastgelu.cpp`**
```
if constexpr(std::is_same<Layout, Row>::value)
```

**`client_example/02_gemm_add_add_fastgelu/gemm_add_fastgelu_generic.cpp`**
```
if constexpr(std::is_same<Layout, Row>::value)
```
