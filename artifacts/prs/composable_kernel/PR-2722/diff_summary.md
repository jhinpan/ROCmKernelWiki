# Diff summary

- **files changed:** 49
- **lines:** +607 / -302
- **kernel-ish files:** 49

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_moe_gemm.hpp`  (+54/-44)
- `include/ck/tensor_operation/gpu/warp/xdlops_gemm.hpp`  (+58/-16)
- `include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_softmax_gemm_xdl_cshuffle_v1.hpp`  (+35/-12)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_b_scale.hpp`  (+32/-13)
- `include/ck/tensor_operation/gpu/grid/gridwise_moe_gemm_blockscale.hpp`  (+25/-20)
- `include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_gemm_xdl_cshuffle_v1.hpp`  (+32/-10)
- `include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_multiple_d_gemm_multiple_d_xdl_cshuffle_v1.hpp`  (+31/-11)
- `include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_multiple_d_softmax_gemm_xdl_cshuffle_v1.hpp`  (+31/-11)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_preshuffle.hpp`  (+24/-18)
- `include/ck/utility/type_convert.hpp`  (+35/-1)
- `include/ck/tensor_operation/gpu/grid/gridwise_moe_mx_gemm_bpreshuffle.hpp`  (+17/-14)
- `include/ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp`  (+27/-1)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_multi_d_b_preshuffle.hpp`  (+16/-12)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_b_preshuffle.hpp`  (+13/-10)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_mx_bpreshuffle.hpp`  (+12/-11)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_dpp.hpp`**
```
static constexpr index_t MWaves   = MPerBlock / (MRepeat * MPerDpp);
static constexpr index_t NWaves   = NPerBlock / (NRepeat * NPerDpp);
static constexpr index_t WaveSize = BlockSize / MWaves / NWaves;
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_mx_pipeline_xdlops_base.hpp`**
```
static constexpr index_t MWaves   = MPerBlock / (MRepeat * MPerXDL);
static constexpr index_t NWaves   = NPerBlock / (NRepeat * NPerXDL);
static constexpr index_t WaveSize = BlockSize / MWaves / NWaves;
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops.hpp`**
```
static constexpr index_t WaveSize = BlockSize / (WaveNumM * WaveNumN);
static constexpr index_t MWaves = MPerBlock / (MRepeat * MPerXDL);
static constexpr index_t NWaves = NPerBlock / (NRepeat * NPerXDL);
static_assert(MWaves > 0);
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_dequant_v1.hpp`**
```
using Base::WaveSize;
constexpr index_t K1 = WaveSize / NPerXDL;
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_dequant_v3.hpp`**
```
using Base::WaveSize;
constexpr index_t K1 = WaveSize / NPerXDL;
```
