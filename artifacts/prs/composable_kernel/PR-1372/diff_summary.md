# Diff summary

- **files changed:** 24
- **lines:** +243 / -50
- **kernel-ish files:** 19

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle_v3.hpp`  (+15/-14)
- `include/ck/utility/amd_smfmac.hpp`  (+28/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_two_stage_xdl_cshuffle.hpp`  (+10/-10)
- `client_example/25_wrapper/wrapper_basic_gemm.cpp`  (+15/-2)
- `client_example/25_wrapper/wrapper_optimized_gemm.cpp`  (+14/-2)
- `example/30_grouped_conv_fwd_multiple_d/grouped_conv_fwd_bias_relu_add_wmma_fp16.cpp`  (+12/-1)
- `example/30_grouped_conv_fwd_multiple_d/grouped_conv_fwd_bias_relu_add_wmma_int8.cpp`  (+12/-1)
- `example/32_batched_gemm_scale_softmax_gemm/batched_gemm_lower_triangle_scale_softmax_gemm_permute_wmma_fp16.cpp`  (+12/-1)
- `example/32_batched_gemm_scale_softmax_gemm/batched_gemm_scale_softmax_gemm_permute_wmma_fp16.cpp`  (+12/-1)
- `example/32_batched_gemm_scale_softmax_gemm/cross_attention_forward_wmma_fp16.cpp`  (+12/-1)
- `example/32_batched_gemm_scale_softmax_gemm/grouped_query_attention_forward_wmma_fp16.cpp`  (+12/-1)
- `example/32_batched_gemm_scale_softmax_gemm/multi_query_attention_forward_wmma_fp16.cpp`  (+12/-1)
- `example/32_batched_gemm_scale_softmax_gemm/self_attention_forward_wmma_fp16.cpp`  (+12/-1)
- `example/38_grouped_conv_bwd_data_multiple_d/grouped_conv_bwd_data_wmma_fp16.cpp`  (+12/-1)
- `example/02_gemm_bilinear/gemm_bilinear_wmma_fp16.cpp`  (+9/-0)

## Key added lines (kernel files)

**`client_example/25_wrapper/wrapper_basic_gemm.cpp`**
```
bool is_supported = ck::is_xdl_supported();
if(!is_supported)
std::cout << "WARNING: xdl example not supported on the platform " << ck::get_device_name()
<< std::endl;
```

**`client_example/25_wrapper/wrapper_optimized_gemm.cpp`**
```
bool is_supported = ck::is_xdl_supported();
if(!is_supported)
std::cout << "WARNING: xdl example not supported on the platform " << ck::get_device_name()
<< std::endl;
```

**`example/02_gemm_bilinear/gemm_bilinear_wmma_fp16.cpp`**
```
bool is_supported = ck::is_gfx11_supported();
if(!is_supported)
std::cout << "WARNING: wmma example not supported on the platform " << ck::get_device_name()
<< std::endl;
```

**`example/02_gemm_bilinear/gemm_bilinear_wmma_int8.cpp`**
```
bool is_supported = ck::is_gfx11_supported();
if(!is_supported)
std::cout << "WARNING: wmma example not supported on the platform " << ck::get_device_name()
<< std::endl;
```

**`example/30_grouped_conv_fwd_multiple_d/grouped_conv_fwd_bias_relu_add_wmma_fp16.cpp`**
```
int main(int argc, char* argv[])
bool is_supported = ck::is_gfx11_supported();
if(!is_supported)
std::cout << "WARNING: wmma example not supported on the platform " << ck::get_device_name()
```
