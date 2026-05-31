# Diff summary

- **files changed:** 10
- **lines:** +35 / -31
- **kernel-ish files:** 10

## Files (by churn)

- `include/ck_tile/ops/gemm_quant/pipeline/gemm_group_quant_utils.hpp`  (+10/-10)
- `include/ck_tile/ops/gemm_quant/block/block_universal_gemm_as_bs_bquant_cr.hpp`  (+6/-4)
- `include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`  (+5/-5)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_v3.hpp`  (+6/-4)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_wp_bquant_pipeline_ag_bg_cr_v2.hpp`  (+4/-4)
- `example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`  (+1/-1)
- `include/ck_tile/host.hpp`  (+1/-1)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_policy.hpp`  (+1/-1)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_fixtures.hpp`  (+1/-1)
- `include/ck_tile/host/tensor_shuffle_utils.hpp`  (+0/-0)

## Key added lines (kernel files)

**`include/ck_tile/ops/gemm_quant/block/block_universal_gemm_as_bs_bquant_cr.hpp`**
```
constexpr auto warp_size = get_warp_size();
static_for<0, WarpGemm::kM * WarpGemm::kN / warp_size, 1>{}(
[&](auto c_row) {
c_block_tensor.get_thread_buffer()[tbuf_offset + c_row] +=
```

**`include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`**
```
make_tuple(kargs.QK_B, kargs.N),
make_tuple(1, kargs.stride_BQ),
make_tuple(number<TilePartitioner::KPerBlock / GemmPipeline::QuantGroupSize>{},
number<TilePartitioner::NPerBlock>{}),
```

**`include/ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_policy.hpp`**
```
NPerBlock,
```

**`include/ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_v3.hpp`**
```
static_assert(KPerBlockBQ == BQDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
NPerBlock == BQDramBlockWindowTmp{}.get_window_lengths()[I1{}],
is_bq_col_major ? make_array(KPerBlockBQ, 0) : make_array(0, KPerBlockBQ);
constexpr index_t tail_count =
```

**`include/ck_tile/ops/gemm_quant/pipeline/gemm_group_quant_utils.hpp`**
```
static constexpr index_t Y  = YPerTile;
static constexpr index_t YR = 1;
static constexpr index_t X0 = NIterPerWarp;
static constexpr index_t X1 = NWarps;
```
