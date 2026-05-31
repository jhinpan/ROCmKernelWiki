# Diff summary

- **files changed:** 50 (diff was byte-capped; summary is partial)
- **lines:** +4423 / -73
- **kernel-ish files:** 42

## Files (by churn)

- `include/ck_tile/ops/flatmm/block/flatmm_32x512x128_1x4x1_16x16x32.hpp`  (+615/-0)
- `include/ck_tile/ops/flatmm/block/uk/flatmm_sn_uk_gfx9_32x128x512_1x4x1_16x16x16.inc`  (+613/-0)
- `example/ck_tile/15_fused_moe/main.cpp`  (+603/-0)
- `include/ck_tile/ops/flatmm/block/flatmm_sn_32x128x512_1x4x1_16x16x32.hpp`  (+562/-0)
- `include/ck_tile/host/reference/reference_fused_moe.hpp`  (+196/-0)
- `include/ck_tile/core/tensor/tile_window_linear.hpp`  (+142/-17)
- `include/ck_tile/host/host_tensor.hpp`  (+103/-18)
- `include/ck_tile/core/utility/static_counter.hpp`  (+116/-0)
- `include/ck_tile/host/fill.hpp`  (+107/-6)
- `include/ck_tile/core/arch/amd_buffer_addressing.hpp`  (+103/-0)
- `include/ck_tile/ops/elementwise/unary_element_wise_operation.hpp`  (+99/-0)
- `include/ck_tile/core/tensor/buffer_view.hpp`  (+80/-6)
- `example/ck_tile/15_fused_moe/fused_moegemm.hpp`  (+84/-0)
- `example/ck_tile/15_fused_moe/instances/fused_moe_api.cpp`  (+80/-0)
- `include/ck_tile/core/tensor/tile_window.hpp`  (+72/-2)

## Key added lines (kernel files)

**`example/ck_tile/06_permute/alternative_impl/matrix_core_swizzle.cpp`**
```
matrix_core_permute_style::b_nr_kr_kw_nw_kv;
matrix_core_permute_style::b_nr_kr_kw_nw_kv;
```

**`example/ck_tile/06_permute/alternative_impl/matrix_core_swizzle_kernel.hpp`**
```
b_nr_kr_kw_nw_kv            = 2, // 0,1,3,4,2,5
b_nr_kr_waveflatten         = b_nr_kr_kw_nw_kv,
```

**`example/ck_tile/15_fused_moe/fused_moe.hpp`**
```
struct fused_moe_args
const void* a_ptr;              // [m, k], input token
const void* a_scale_ptr;        // [m, 1], token scale
const void* g_ptr;              // [e, n, k]/[e, 2*n, k], pre-shuffle([e, nr, kr, w])
```

**`example/ck_tile/15_fused_moe/fused_moegemm.hpp`**
```
template <typename I, typename W, typename O, typename ST, typename SW, typename SQ, typename KW>
struct FusedMoeGemmTypeConfig;
template <typename ST, typename SW, typename SQ, typename KW>
struct FusedMoeGemmTypeConfig<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t, ST, SW, SQ, KW>
```

**`example/ck_tile/15_fused_moe/fused_moesorting.hpp`**
```
struct fused_moesorting_trait
std::string index_type;
std::string weight_type; // currently always float
struct fused_moesorting_args : public ck_tile::MoeSortingHostArgs
```
