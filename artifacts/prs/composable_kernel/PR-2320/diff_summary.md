# Diff summary

- **files changed:** 31
- **lines:** +213 / -206
- **kernel-ish files:** 31

## Files (by churn)

- `include/ck_tile/ops/fused_moe/kernel/moe_sorting_kernel.hpp`  (+53/-53)
- `include/ck_tile/ops/fused_moe/pipeline/fused_moegemm_pipeline_flatmm_policy.hpp`  (+26/-26)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp`  (+19/-19)
- `include/ck_tile/ops/flatmm/block/flatmm_32x512x128_1x4x1_16x16x32.hpp`  (+13/-13)
- `example/ck_tile/02_layernorm2d/generate.py`  (+10/-10)
- `example/ck_tile/10_rmsnorm2d/generate.py`  (+10/-10)
- `example/ck_tile/11_add_rmsnorm2d_rdquant/add_rmsnorm2d_rdquant_fwd.hpp`  (+10/-10)
- `example/ck_tile/12_smoothquant/smoothquant.hpp`  (+10/-10)
- `example/ck_tile/14_moe_smoothquant/moe_smoothquant.hpp`  (+10/-10)
- `include/ck/tensor_operation/gpu/grid/gridwise_moe_gemm.hpp`  (+6/-5)
- `include/ck/tensor_operation/gpu/grid/gridwise_moe_gemm_blockscale.hpp`  (+5/-5)
- `include/ck/tensor_operation/gpu/grid/gridwise_moe_mx_gemm.hpp`  (+5/-5)
- `include/ck/ck.hpp`  (+6/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_b_preshuffle.hpp`  (+3/-3)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_multi_d_b_preshuffle.hpp`  (+3/-3)

## Key added lines (kernel files)

**`example/ck_tile/02_layernorm2d/generate.py`**
```
static constexpr bool is_warp_per_row = ThreadPerBlock_N_ <= WarpSize;
static_assert((ThreadPerBlock_M_ * ThreadPerBlock_N_) % WarpSize == 0);
(ThreadPerBlock_M_ * ThreadPerBlock_N_) / WarpSize;
static_assert(WarpSize % ThreadPerBlock_N_ == 0);
```

**`example/ck_tile/05_reduce/reduce.hpp`**
```
WarpSize * reduce_on_sequence(BlockWarps{}, multiplies{}, number<1>{});
```

**`example/ck_tile/10_rmsnorm2d/generate.py`**
```
static constexpr bool is_warp_per_row = ThreadPerBlock_N_ <= WarpSize;
static_assert((ThreadPerBlock_M_ * ThreadPerBlock_N_) % WarpSize == 0);
(ThreadPerBlock_M_ * ThreadPerBlock_N_) / WarpSize;
static_assert(WarpSize % ThreadPerBlock_N_ == 0);
```

**`example/ck_tile/11_add_rmsnorm2d_rdquant/add_rmsnorm2d_rdquant_fwd.hpp`**
```
static constexpr bool is_warp_per_row = ThreadPerBlock_N_ <= WarpSize;
static_assert((ThreadPerBlock_M_ * ThreadPerBlock_N_) % WarpSize == 0);
(ThreadPerBlock_M_ * ThreadPerBlock_N_) / WarpSize;
static_assert(WarpSize % ThreadPerBlock_N_ == 0);
```

**`example/ck_tile/12_smoothquant/smoothquant.hpp`**
```
static constexpr bool is_warp_per_row = ThreadPerBlock_N_ <= WarpSize;
static_assert((ThreadPerBlock_M_ * ThreadPerBlock_N_) % WarpSize == 0);
(ThreadPerBlock_M_ * ThreadPerBlock_N_) / WarpSize;
static_assert(WarpSize % ThreadPerBlock_N_ == 0);
```
