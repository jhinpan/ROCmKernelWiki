# Diff summary

- **files changed:** 7
- **lines:** +1352 / -272
- **kernel-ish files:** 7

## Files (by churn)

- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_mx_moe_v1.hpp`  (+891/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_moe_mx_gemm_bpreshuffle.hpp`  (+261/-207)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_mx_moe_v3.hpp`  (+168/-56)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_mx_moe_selector.hpp`  (+23/-1)
- `example/67_gemm_microscaling/moe_gemm1_xdl_mx_fp4_bpreshuffle.cpp`  (+6/-6)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_mx_moe_gufusion_v3.hpp`  (+2/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_moe_mx_gemm_bpreshuffle.hpp`  (+1/-1)

## Key added lines (kernel files)

**`example/67_gemm_microscaling/moe_gemm1_xdl_mx_fp4_bpreshuffle.cpp`**
```
static constexpr ck::index_t MPerBlock = 32;
MPerBlock,  128,  KPerBlock,
ck::index_t N       = 7168;
ck::index_t K       = 256;
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_mx_moe_gufusion_v3.hpp`**
```
if constexpr(MPerBlock >= 64)
HotLoopScheduler();
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_mx_moe_selector.hpp`**
```
return BlockwiseGemmXdlops_pipeline_bpreshuffle_mx_moe_v1<
BlkGemmPipeSche,
ThreadBlockSize,
ScaleBlockSize,
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_mx_moe_v1.hpp`**
```
namespace ck {
template <BlockGemmPipelineScheduler BlkGemmPipelineVer,
index_t ThreadBlockSize,
index_t ScaleBlockSize,
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_mx_moe_v3.hpp`**
```
constexpr auto num_total_stages = std::max(2, MRepeat);
if constexpr(num_total_stages > 2)
constexpr auto num_mfma_perstage      = num_mfma_inst / num_total_stages;
constexpr auto num_ds_read_a_perstage = num_ds_read_inst_a / num_total_stages;
```
