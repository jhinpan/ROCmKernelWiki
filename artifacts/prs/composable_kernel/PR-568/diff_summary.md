# Diff summary

- **files changed:** 29
- **lines:** +1635 / -36
- **kernel-ish files:** 23

## Files (by churn)

- `profiler/include/profiler/profile_gemm_add_relu_add_layernorm_impl.hpp`  (+346/-0)
- `client_example/03_gemm_layernorm/gemm_add_relu_add_layernorm_welford.cpp`  (+244/-0)
- `profiler/src/profile_gemm_add_relu_add_layernorm.cpp`  (+215/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_add_relu_add_layernorm.hpp`  (+172/-0)
- `library/src/tensor_operation_instance/gpu/gemm_add_relu_add_layernorm/device_gemm_add_relu_add_xdl_c_shuffle_layernorm_f16_km_kn_mn_mn_mn_instance.cpp`  (+130/-0)
- `library/src/tensor_operation_instance/gpu/gemm_add_relu_add_layernorm/device_gemm_add_relu_add_xdl_c_shuffle_layernorm_f16_km_nk_mn_mn_mn_instance.cpp`  (+130/-0)
- `library/src/tensor_operation_instance/gpu/gemm_add_relu_add_layernorm/device_gemm_add_relu_add_xdl_c_shuffle_layernorm_f16_mk_kn_mn_mn_mn_instance.cpp`  (+130/-0)
- `library/src/tensor_operation_instance/gpu/gemm_add_relu_add_layernorm/device_gemm_add_relu_add_xdl_c_shuffle_layernorm_f16_mk_nk_mn_mn_mn_instance.cpp`  (+127/-0)
- `test/gemm_layernorm/test_gemm_add_relu_add_layernorm_fp16.cpp`  (+77/-0)
- `example/21_gemm_layernorm/gemm_bias_relu_add_layernorm_xdl_welford_fp16.cpp`  (+10/-9)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_layernorm_xdl_cshuffle.hpp`  (+14/-2)
- `test/normalization/CMakeLists.txt`  (+6/-7)
- `client_example/03_gemm_layernorm/CMakeLists.txt`  (+5/-2)
- `test/gemm_layernorm/CMakeLists.txt`  (+7/-0)
- `library/src/tensor_operation_instance/gpu/gemm_add_relu_add_layernorm/CMakeLists.txt`  (+6/-0)

## Key added lines (kernel files)

**`client_example/01_gemm/gemm.cpp`**
```
if constexpr(std::is_same<Layout, ck::tensor_layout::gemm::RowMajor>::value)
```

**`client_example/02_gemm_add_add_fastgelu/gemm_add_add_fastgelu.cpp`**
```
if constexpr(std::is_same<Layout, ck::tensor_layout::gemm::RowMajor>::value)
```

**`client_example/02_gemm_add_add_fastgelu/gemm_add_fastgelu.cpp`**
```
if constexpr(std::is_same<Layout, ck::tensor_layout::gemm::RowMajor>::value)
```

**`client_example/02_gemm_add_add_fastgelu/gemm_fastgelu.cpp`**
```
if constexpr(std::is_same<Layout, ck::tensor_layout::gemm::RowMajor>::value)
```

**`client_example/03_gemm_layernorm/gemm_add_add_layernorm_naive.cpp`**
```
if constexpr(std::is_same<Layout, ck::tensor_layout::gemm::RowMajor>::value)
```
