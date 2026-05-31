# Diff summary

- **files changed:** 11
- **lines:** +747 / -266
- **kernel-ish files:** 11

## Files (by churn)

- `include/ck_tile/core/arch/amd_buffer_addressing.hpp`  (+435/-199)
- `include/ck_tile/core/tensor/tile_window.hpp`  (+94/-11)
- `include/ck_tile/core/tensor/tile_elementwise.hpp`  (+89/-10)
- `include/ck_tile/core/tensor/buffer_view.hpp`  (+34/-11)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async.hpp`  (+27/-9)
- `include/ck_tile/core/tensor/tensor_view.hpp`  (+15/-9)
- `example/ck_tile/01_fmha/generate.py`  (+17/-6)
- `include/ck_tile/core/tensor/load_tile.hpp`  (+13/-6)
- `include/ck_tile/core/config.hpp`  (+18/-0)
- `include/ck_tile/core/arch/arch.hpp`  (+3/-5)
- `include/ck_tile/core/tensor/null_tile_window.hpp`  (+2/-0)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/generate.py`**
```
if not per_dtypes:
per_dtypes += '    (void)t ; (void)s ; (void)a;'
if hdim == 256 or hdim == 32:
if bias == "bias":
```

**`include/ck_tile/core/arch/amd_buffer_addressing.hpp`**
```
namespace impl {
template<index_t N, typename T> struct buffer_load_trait;
template<typename T> struct buffer_load_trait<16, T> { using payload_t = fp32x4_t; };
template<typename T> struct buffer_load_trait<8 , T> { using payload_t = fp32x2_t; };
```

**`include/ck_tile/core/arch/arch.hpp`**
```
CK_TILE_DEVICE void s_nop(index_t cnt = 0)
asm volatile("s_nop %0" : : "n"(cnt) :);
__builtin_amdgcn_sched_barrier(cnt);
```

**`include/ck_tile/core/config.hpp`**
```
(HIP_VERSION_MAJOR == 6 && HIP_VERSION_MINOR == 2 && HIP_VERSION_PATCH >= 41133)
```

**`include/ck_tile/core/tensor/buffer_view.hpp`**
```
CK_TILE_HOST_DEVICE void init_raw() {}
int32x4_t cached_buf_res_;
: p_data_{}, buffer_size_{}, cached_buf_res_{0}, invalid_element_value_{}
: p_data_{p_data}, buffer_size_{buffer_size}, cached_buf_res_{0}, invalid_element_value_{0}
```
