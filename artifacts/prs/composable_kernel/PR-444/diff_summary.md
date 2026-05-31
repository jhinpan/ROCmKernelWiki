# Diff summary

- **files changed:** 77
- **lines:** +279 / -249
- **kernel-ish files:** 77

## Files (by churn)

- `library/include/ck/library/utility/check_err.hpp`  (+46/-43)
- `library/include/ck/library/utility/ranges.hpp`  (+60/-0)
- `example/32_batched_gemm_scale_softmax_gemm/run_grouped_gemm_scale_softmax_gemm_permute.inc`  (+12/-12)
- `library/include/ck/library/utility/iterator.hpp`  (+22/-0)
- `example/16_gemm_multi_d_multi_reduces/gemm_reduce_xdl_common.hpp`  (+9/-12)
- `example/22_cgemm/cgemm_xdl_common.hpp`  (+8/-8)
- `test/reference_conv_fwd/reference_conv_fwd.cpp`  (+7/-8)
- `example/18_batched_gemm_reduce/batched_gemm_reduce_xdl_fp16.cpp`  (+6/-7)
- `example/34_batchnorm/batchnorm_forward_nhwc.cpp`  (+5/-8)
- `example/10_convnd_fwd_multiple_d_multiple_reduce/run_convnd_fwd_max_example.inc`  (+4/-7)
- `test/gemm/gemm_util.hpp`  (+5/-5)
- `example/16_gemm_multi_d_multi_reduces/gemm_add_add_mean_meansquare_xdl_fp16.cpp`  (+3/-6)
- `example/21_gemm_layernorm/gemm_xdl_layernorm_single_kernel_fp16.cpp`  (+3/-6)
- `example/35_splitK_gemm/run_splitK_gemm_example.inc`  (+3/-6)
- `profiler/include/profile_batched_gemm_reduce_impl.hpp`  (+3/-6)

## Key added lines (kernel files)

**`example/01_gemm/gemm_xdl_skip_b_lds_fp16.cpp`**
```
ck::utils::check_err(c_m_n_device_result, c_m_n_host_result);
```

**`example/01_gemm/run_gemm_example.inc`**
```
return ck::utils::check_err(c_m_n_device_result_converted, c_m_n_host_result);
return ck::utils::check_err(c_m_n_device_result, c_m_n_host_result);
```

**`example/02_gemm_bilinear/gemm_bilinear_xdl_fp16.cpp`**
```
return ck::utils::check_err(e_m_n_device_result, e_m_n_host_result) ? 0 : 1;
```

**`example/03_gemm_bias_relu/gemm_bias_relu_xdl_fp16.cpp`**
```
return ck::utils::check_err(e_m_n_device_result, e_m_n_host_result) ? 0 : 1;
```

**`example/04_gemm_add_add_fastgelu/run_gemm_add_add_fastgelu_example.inc`**
```
return ck::utils::check_err(e_m_n_device_result_converted, e_m_n_host_result);
return ck::utils::check_err(e_m_n_device_result, e_m_n_host_result);
```
