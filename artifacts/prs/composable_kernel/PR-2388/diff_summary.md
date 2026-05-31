# Diff summary

- **files changed:** 10
- **lines:** +112 / -110
- **kernel-ish files:** 10

## Files (by churn)

- `include/ck_tile/ops/fused_moe/kernel/moe_sorting_kernel.hpp`  (+55/-55)
- `example/ck_tile/10_rmsnorm2d/generate.py`  (+11/-11)
- `example/ck_tile/02_layernorm2d/generate.py`  (+10/-10)
- `example/ck_tile/12_smoothquant/smoothquant.hpp`  (+10/-10)
- `example/ck_tile/14_moe_smoothquant/moe_smoothquant.hpp`  (+10/-10)
- `include/ck_tile/core/tensor/tile_window_linear.hpp`  (+9/-10)
- `include/ck_tile/ops/norm_reduce/block/block_norm_reduce.hpp`  (+2/-2)
- `example/ck_tile/17_grouped_gemm/grouped_gemm_tileloop.cpp`  (+3/-0)
- `example/ck_tile/05_reduce/reduce.hpp`  (+1/-1)
- `include/ck_tile/ops/fused_moe/kernel/fused_moegemm_shape.hpp`  (+1/-1)

## Key added lines (kernel files)

**`example/ck_tile/02_layernorm2d/generate.py`**
```
static constexpr bool is_warp_per_row = ThreadPerBlock_N_ <= ck_tile::get_warp_size();
static_assert((ThreadPerBlock_M_ * ThreadPerBlock_N_) % ck_tile::get_warp_size() == 0);
(ThreadPerBlock_M_ * ThreadPerBlock_N_) / ck_tile::get_warp_size();
static_assert(ck_tile::get_warp_size() % ThreadPerBlock_N_ == 0);
```

**`example/ck_tile/05_reduce/reduce.hpp`**
```
ck_tile::get_warp_size() * reduce_on_sequence(BlockWarps{}, multiplies{}, number<1>{});
```

**`example/ck_tile/10_rmsnorm2d/generate.py`**
```
static constexpr bool is_warp_per_row = ThreadPerBlock_N_ <= ck_tile::get_warp_size();
static_assert((ThreadPerBlock_M_ * ThreadPerBlock_N_) % ck_tile::get_warp_size() == 0);
(ThreadPerBlock_M_ * ThreadPerBlock_N_) / ck_tile::get_warp_size();
static_assert(ck_tile::get_warp_size() % ThreadPerBlock_N_ == 0);
```

**`example/ck_tile/12_smoothquant/smoothquant.hpp`**
```
static constexpr bool is_warp_per_row = ThreadPerBlock_N_ <= ck_tile::get_warp_size();
static_assert((ThreadPerBlock_M_ * ThreadPerBlock_N_) % ck_tile::get_warp_size() == 0);
(ThreadPerBlock_M_ * ThreadPerBlock_N_) / ck_tile::get_warp_size();
static_assert(ck_tile::get_warp_size() % ThreadPerBlock_N_ == 0);
```

**`example/ck_tile/14_moe_smoothquant/moe_smoothquant.hpp`**
```
static constexpr bool is_warp_per_row = ThreadPerBlock_N_ <= ck_tile::get_warp_size();
static_assert((ThreadPerBlock_M_ * ThreadPerBlock_N_) % ck_tile::get_warp_size() == 0);
(ThreadPerBlock_M_ * ThreadPerBlock_N_) / ck_tile::get_warp_size();
static_assert(ck_tile::get_warp_size() % ThreadPerBlock_N_ == 0);
```
