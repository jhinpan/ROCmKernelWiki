# Diff summary

- **files changed:** 13
- **lines:** +2912 / -140
- **kernel-ish files:** 12

## Files (by churn)

- `include/ck_tile/ops/flatmm/pipeline/mixed_prec_flatmm_pipeline_agmem_bgmem_creg_v1.hpp`  (+1170/-0)
- `example/65_gemm_multiply_multiply/moe_gemm1_xdl_fp8_blockscale_splitk.cpp`  (+539/-0)
- `include/ck_tile/ops/flatmm/pipeline/mixed_prec_flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp`  (+510/-1)
- `library/include/ck/library/reference_tensor_operation/cpu/reference_moe_gemm1_blockscale_splitk.hpp`  (+232/-0)
- `include/ck_tile/ops/flatmm/kernel/moe_flatmm_kernel.hpp`  (+170/-47)
- `include/ck_tile/core/tensor/tile_scatter_gather.hpp`  (+182/-1)
- `include/ck/tensor_operation/gpu/grid/gridwise_moe_gemm_blockscale.hpp`  (+74/-51)
- `include/ck/tensor_operation/gpu/device/impl/device_moe_gemm_blockscale.hpp`  (+27/-17)
- `include/ck_tile/ops/gemm/warp/warp_gemm.hpp`  (+0/-19)
- `example/65_gemm_multiply_multiply/moe_gemm1_xdl_fp8_blockscale.cpp`  (+2/-2)
- `example/65_gemm_multiply_multiply/moe_gemm2_xdl_fp8_blockscale.cpp`  (+2/-2)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_moe_blockscale_b_preshuffle_v1.hpp`  (+3/-0)
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

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_moe_blockscale_b_preshuffle_v1.hpp`**
```
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_sched_barrier(0);
```

**`include/ck/tensor_operation/gpu/device/impl/device_moe_gemm_blockscale.hpp`**
```
bool IsSplitK                               = false,
IsSplitK,
std::tie(gdx, gdy, gdz) = GridwiseGemm::CalculateGridSize(
arg.M, arg.N * (IsInputGemm && IsSplitK ? 2 : 1), arg.K, arg.KBatch);
```
