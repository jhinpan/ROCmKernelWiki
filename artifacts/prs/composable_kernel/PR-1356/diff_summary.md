# Diff summary

- **files changed:** 13
- **lines:** +581 / -243
- **kernel-ish files:** 13

## Files (by churn)

- `include/ck_tile/core/arch/amd_buffer_addressing.hpp`  (+333/-185)
- `include/ck_tile/core/tensor/tile_window.hpp`  (+90/-10)
- `include/ck_tile/core/tensor/tile_elementwise.hpp`  (+48/-8)
- `include/ck_tile/core/tensor/buffer_view.hpp`  (+34/-11)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async.hpp`  (+25/-8)
- `include/ck_tile/core/tensor/tensor_view.hpp`  (+15/-9)
- `include/ck_tile/core/tensor/load_tile.hpp`  (+13/-6)
- `include/ck_tile/core/config.hpp`  (+9/-0)
- `include/ck_tile/core/arch/arch.hpp`  (+3/-5)
- `example/ck_tile/01_fmha/codegen/ops/fmha_bwd.py`  (+3/-1)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`  (+3/-0)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`  (+3/-0)
- `include/ck_tile/core/tensor/null_tile_window.hpp`  (+2/-0)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/ops/fmha_bwd.py`**
```
if not per_dtypes:
per_dtypes += '    (void)t ; (void)s ; (void)a;'
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`**
```
if not per_dtypes:
per_dtypes += '    (void)t ; (void)s ; (void)a;'
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`**
```
if not per_dtypes:
per_dtypes += '    (void)t ; (void)s ; (void)a;'
```

**`include/ck_tile/core/arch/amd_buffer_addressing.hpp`**
```
template <index_t bytes, bool pre_nop = false>
template <bool pre_nop>
struct buffer_load<16, pre_nop>
index_t /*s_offset*/,
```

**`include/ck_tile/core/arch/arch.hpp`**
```
CK_TILE_DEVICE void s_nop(index_t cnt = 0)
asm volatile("s_nop %0" : : "n"(cnt) :);
__builtin_amdgcn_sched_barrier(cnt);
```
