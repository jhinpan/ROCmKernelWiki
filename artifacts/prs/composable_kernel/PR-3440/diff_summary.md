# Diff summary

- **files changed:** 7
- **lines:** +873 / -71
- **kernel-ish files:** 6

## Files (by churn)

- `example/65_gemm_multiply_multiply/moe_gemm1_xdl_fp8_blockscale_splitk.cpp`  (+535/-0)
- `library/include/ck/library/reference_tensor_operation/cpu/reference_moe_gemm1_blockscale_splitk.hpp`  (+232/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_moe_gemm_blockscale.hpp`  (+74/-50)
- `include/ck/tensor_operation/gpu/device/impl/device_moe_gemm_blockscale.hpp`  (+27/-17)
- `example/65_gemm_multiply_multiply/moe_gemm1_xdl_fp8_blockscale.cpp`  (+2/-2)
- `example/65_gemm_multiply_multiply/moe_gemm2_xdl_fp8_blockscale.cpp`  (+2/-2)
- `example/65_gemm_multiply_multiply/CMakeLists.txt`  (+1/-0)

## Key added lines (kernel files)

**`example/65_gemm_multiply_multiply/moe_gemm1_xdl_fp8_blockscale.cpp`**
```
ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1, ActOP, Nswizzle, true, false, MulRoutedWeig
ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, ActOP, Nswizzle, true, false, MulRoutedWeig
```

**`example/65_gemm_multiply_multiply/moe_gemm1_xdl_fp8_blockscale_splitk.cpp`**
```
using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;
template <ck::index_t... Is>
```

**`example/65_gemm_multiply_multiply/moe_gemm2_xdl_fp8_blockscale.cpp`**
```
ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1, 0, false, false, false, MulRoutedWeight, in
ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, 0, false, false, false, MulRoutedWeight, in
```

**`include/ck/tensor_operation/gpu/device/impl/device_moe_gemm_blockscale.hpp`**
```
bool IsSplitK                               = false,
IsSplitK,
std::tie(gdx, gdy, gdz) = GridwiseGemm::CalculateGridSize(
arg.M, arg.N * (IsInputGemm && IsSplitK ? 2 : 1), arg.K, arg.KBatch);
```

**`include/ck/tensor_operation/gpu/grid/gridwise_moe_gemm_blockscale.hpp`**
```
karg.p_a_scale_grid + splitk_batch_offset.ascale_k_split_offset,
karg.p_b_scale_grid + splitk_batch_offset.bscale_k_split_offset,
karg.p_a_scale_grid + splitk_batch_offset.ascale_k_split_offset,
karg.p_b_scale_grid + splitk_batch_offset.bscale_k_split_offset,
```
