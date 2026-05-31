# Diff summary

- **files changed:** 103
- **lines:** +657 / -649
- **kernel-ish files:** 103

## Files (by churn)

- `library/include/ck/library/utility/check_err.hpp`  (+46/-43)
- `library/include/ck/library/utility/ranges.hpp`  (+60/-0)
- `library/include/ck/library/utility/host_tensor.hpp`  (+28/-26)
- `library/include/ck/library/utility/algorithm.hpp`  (+43/-0)
- `example/18_batched_gemm_reduce/batched_gemm_reduce_xdl_fp16.cpp`  (+15/-19)
- `profiler/include/profile_batched_gemm_reduce_impl.hpp`  (+12/-18)
- `profiler/include/profile_gemm_bias_add_reduce_impl.hpp`  (+13/-17)
- `example/25_gemm_bias_e_permute/gemm_bias_e_permute_g1m2n3k1_xdl_fp16.cpp`  (+7/-21)
- `example/25_gemm_bias_e_permute/gemm_bias_e_permute_g1m3n2k1_xdl_fp16.cpp`  (+7/-21)
- `example/29_batched_gemm_bias_e_permute/batched_gemm_bias_e_permute_xdl_fp16.cpp`  (+7/-21)
- `profiler/include/profile_gemm_reduce_impl.hpp`  (+12/-15)
- `example/26_contraction/contraction_bilinear_xdl_fp32.cpp`  (+7/-19)
- `example/28_grouped_gemm_bias_e_permute/grouped_gemm_bias_e_permute_xdl_fp16.cpp`  (+7/-19)
- `example/34_batchnorm/batchnorm_forward_nhwc.cpp`  (+10/-16)
- `example/22_cgemm/cgemm_xdl_common.hpp`  (+13/-12)

## Key added lines (kernel files)

**`example/01_gemm/gemm_xdl_skip_b_lds_fp16.cpp`**
```
using namespace ck::literals;
return HostTensorDescriptor({row, col}, {stride, 1_uz});
return HostTensorDescriptor({row, col}, {1_uz, stride});
ck::utils::check_err(c_m_n_device_result, c_m_n_host_result);
```

**`example/01_gemm/run_gemm_example.inc`**
```
return ck::utils::check_err(c_m_n_device_result_converted, c_m_n_host_result);
return ck::utils::check_err(c_m_n_device_result, c_m_n_host_result);
```

**`example/02_gemm_bilinear/gemm_bilinear_xdl_fp16.cpp`**
```
using namespace ck::literals;
return HostTensorDescriptor({row, col}, {stride, 1_uz});
return HostTensorDescriptor({row, col}, {1_uz, stride});
Tensor<CShuffleDataType> c_m_n({M, N});
```

**`example/03_gemm_bias_relu/gemm_bias_relu_xdl_fp16.cpp`**
```
using namespace ck::literals;
return HostTensorDescriptor({row, col}, {stride, 1_uz});
return HostTensorDescriptor({row, col}, {1_uz, stride});
return ck::utils::check_err(e_m_n_device_result, e_m_n_host_result) ? 0 : 1;
```

**`example/04_gemm_add_add_fastgelu/run_gemm_add_add_fastgelu_example.inc`**
```
Tensor<AccDataType> c_m_n({M, N});
return ck::utils::check_err(e_m_n_device_result_converted, e_m_n_host_result);
return ck::utils::check_err(e_m_n_device_result, e_m_n_host_result);
```
