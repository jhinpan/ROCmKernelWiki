# Diff summary

- **files changed:** 21
- **lines:** +375 / -189
- **kernel-ish files:** 21

## Files (by churn)

- `include/ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_global.hpp`  (+160/-89)
- `include/ck/tensor_operation/gpu/grid/gridwise_ab_transfer_wave_tiles.hpp`  (+26/-20)
- `include/ck/tensor_operation/gpu/grid/gridwise_ab_transfer_wave_tiles_interleave.hpp`  (+22/-21)
- `example/60_gemm_multi_ABD/gemm_multi_ABD_wmma_fastgelu_bf16_i8.cpp`  (+26/-4)
- `example/60_gemm_multi_ABD/gemm_multi_ABD_wmma_bias_fastgelu_bf16_i8.cpp`  (+26/-3)
- `example/60_gemm_multi_ABD/gemm_multi_ABD_wmma_multiply_bias_fastgelu_bf16_i8.cpp`  (+26/-3)
- `example/60_gemm_multi_ABD/gemm_multi_ABD_wmma_fp16.cpp`  (+25/-2)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3_common.hpp`  (+6/-21)
- `library/src/tensor_operation_instance/gpu/gemm_multi_abd/device_gemm_wmma_multi_abd_bf16_i8_bf16_mk_kn_mn_common.hpp`  (+11/-8)
- `library/src/tensor_operation_instance/gpu/gemm_multi_abd/device_gemm_wmma_multi_abd_bf16_i8_bf16_mk_nk_mn_common.hpp`  (+9/-6)
- `include/ck/tensor_operation/gpu/grid/gridwise_ab_transfer_thread_tiles.hpp`  (+13/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_ab_transfer_thread_tiles_preshuffle.hpp`  (+13/-0)
- `library/src/tensor_operation_instance/gpu/gemm_multi_abd/device_gemm_wmma_multi_abd_bias_gelu_bf16_i8_bf16_mk_nk_mn_v1_instance.cpp`  (+4/-4)
- `library/src/tensor_operation_instance/gpu/gemm_multi_abd/device_gemm_wmma_multi_abd_bf16_i8_bf16_mk_kn_mn_v1_instance.cpp`  (+1/-1)
- `library/src/tensor_operation_instance/gpu/gemm_multi_abd/device_gemm_wmma_multi_abd_bias_bf16_i8_bf16_mk_kn_mn_v1_instance.cpp`  (+1/-1)

## Key added lines (kernel files)

**`example/60_gemm_multi_ABD/gemm_multi_ABD_wmma_bias_fastgelu_bf16_i8.cpp`**
```
S<8, 16, 1>,
ck::BlockGemmPipelineVersion::v1>;
auto f_get_default_stride =
[](std::size_t row, std::size_t col, ck::index_t stride, auto layout) {
```

**`example/60_gemm_multi_ABD/gemm_multi_ABD_wmma_fastgelu_bf16_i8.cpp`**
```
S<8, 16, 1>,
ck::BlockGemmPipelineVersion::v1>;
else if(argc == 10)
auto f_get_default_stride =
```

**`example/60_gemm_multi_ABD/gemm_multi_ABD_wmma_fp16.cpp`**
```
S<4, 16, 1>,
auto f_get_default_stride =
[](std::size_t row, std::size_t col, ck::index_t stride, auto layout) {
if(stride == -1 || stride == 0)
```

**`example/60_gemm_multi_ABD/gemm_multi_ABD_wmma_multiply_bias_fastgelu_bf16_i8.cpp`**
```
S<8, 16, 1>,
ck::BlockGemmPipelineVersion::v1>;
auto f_get_default_stride =
[](std::size_t row, std::size_t col, ck::index_t stride, auto layout) {
```

**`include/ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_global.hpp`**
```
template <typename SrcDescs,
typename SrcDatas,
bool DoTranspose,
index_t NumThreadScratch = 1>
```
