# Diff summary

- **files changed:** 120 (diff was byte-capped; summary is partial)
- **lines:** +1644 / -792
- **kernel-ish files:** 120

## Files (by churn)

- `include/ck/library/utility/host_tensor.hpp`  (+477/-30)
- `include/ck/tensor_operation/gpu/device/tensor_layout.hpp`  (+90/-75)
- `include/ck/library/utility/validation_common.hpp`  (+0/-50)
- `example/60_gemm_multi_ABD/gemm_multi_ABD_xdl_multiply_bias_fastgelu_bf16_i8.cpp`  (+28/-19)
- `example/60_gemm_multi_ABD/gemm_multi_ABD_xdl_bias_fastgelu_bf16_i8.cpp`  (+27/-18)
- `example/60_gemm_multi_ABD/gemm_multi_ABD_xdl_fastgelu_bf16_i8.cpp`  (+27/-18)
- `example/66_complex_contraction_bilinear/run_complex_contraction_bilinear_example.inc`  (+21/-16)
- `example/30_grouped_conv_fwd_multiple_d/common.hpp`  (+24/-12)
- `example/30_grouped_conv_fwd_multiple_d/common_wmma.hpp`  (+24/-12)
- `example/32_batched_gemm_scale_softmax_gemm/run_cross_attention_wmma.inc`  (+29/-6)
- `example/32_batched_gemm_scale_softmax_gemm/run_self_attention_wmma.inc`  (+29/-6)
- `profiler/include/profiler/profile_gemm_blockscale_wp_impl.hpp`  (+17/-17)
- `profiler/include/profiler/profile_gemm_multiply_add_impl.hpp`  (+22/-12)
- `example/32_batched_gemm_scale_softmax_gemm/run_batched_gemm_scale_softmax_gemm_permute_wmma.inc`  (+28/-5)
- `example/32_batched_gemm_scale_softmax_gemm/run_grouped_query_attention_forward_wmma.inc`  (+28/-5)

## Key added lines (kernel files)

**`example/01_gemm/run_gemm_example.inc`**
```
return HostTensorDescriptor({row, col}, {stride, 1_uz}, layout);
return HostTensorDescriptor({row, col}, {1_uz, stride}, layout);
```

**`example/03_gemm_bias_relu/gemm_bias_relu_xdl_fp16.cpp`**
```
const auto StrideD = std::is_same<decltype(ELayout{}), ck::tensor_layout::gemm::RowMajor>::value
? d_m_n.mDesc.GetStrides()[0]
: d_m_n.mDesc.GetStrides()[1];
std::array<ck::index_t, 1>{static_cast<int>(StrideD)},
```

**`example/04_gemm_add_add_fastgelu/run_gemm_add_add_fastgelu_example.inc`**
```
ProblemSize ps =
problem_size; // make mutable copy because default stride values of 0 need to be updated
auto& [M, N, K, StrideA, StrideB, StrideD0, StrideD1, StrideE] = ps;
auto fetch_leading_stride = [](const auto& tensor, auto layout_tag) -> int {
```

**`example/13_pool2d_fwd/pool2d_fwd_common.hpp`**
```
return HostTensorDescriptor({N_, C_, H, W}, {C_ * H * W, H * W, W, 1_uz}, layout);
return HostTensorDescriptor({N_, C_, H, W}, {C_ * H * W, 1_uz, W * C_, C_}, layout);
```

**`example/14_gemm_quantization/gemm_dl_quantization_int8.cpp`**
```
std::vector<std::size_t>({stride, 1_uz}),
std::vector<std::size_t>({1_uz, stride}),
```
