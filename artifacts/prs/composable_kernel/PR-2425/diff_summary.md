# Diff summary

- **files changed:** 12
- **lines:** +224 / -142
- **kernel-ish files:** 11

## Files (by churn)

- `include/ck_tile/core/arch/amd_buffer_addressing.hpp`  (+55/-21)
- `include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp`  (+54/-21)
- `include/ck_tile/core/tensor/tile_window_linear.hpp`  (+29/-44)
- `include/ck_tile/core/tensor/tile_window.hpp`  (+24/-36)
- `example/ck_tile/36_copy/test_copy.hpp`  (+26/-11)
- `include/ck_tile/core/tensor/load_tile.hpp`  (+13/-0)
- `example/ck_tile/36_copy/test_copy.cpp`  (+6/-5)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp`  (+9/-0)
- `include/ck_tile/core/tensor/tensor_view.hpp`  (+4/-2)
- `include/ck_tile/core/tensor/buffer_view.hpp`  (+3/-1)
- `CHANGELOG.md`  (+1/-0)
- `example/ck_tile/03_gemm/gemm_utils.hpp`  (+0/-1)

## Key added lines (kernel files)

**`example/ck_tile/36_copy/test_copy.cpp`**
```
using BlockWaves         = ck_tile::sequence<2, 1>;
using BlockTile          = ck_tile::sequence<64, 8>;
using WaveTile           = ck_tile::sequence<64, 8>;
using Vector             = ck_tile::sequence<1, 2>;
```

**`example/ck_tile/36_copy/test_copy.hpp`**
```
template <typename XDataType_, typename BlockShape_, bool AsyncCopy_>
using XDataType                 = remove_cvref_t<XDataType_>;
using BlockShape                = remove_cvref_t<BlockShape_>;
static constexpr bool AsyncCopy = AsyncCopy_;
```

**`include/ck_tile/core/arch/amd_buffer_addressing.hpp`**
```
using as3_uint32_ptr = uint32_t __attribute__((address_space(3)))*;
as3_uint32_ptr lds_ptr,
CK_TILE_DEVICE void amd_async_buffer_load_impl(CK_TILE_LDS_ADDR T* smem,
constexpr index_t bytes = sizeof(T) * N;
```

**`include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp`**
```
using as3_uint32_ptr = uint32_t __attribute__((address_space(3)))*;
as3_uint32_ptr lds_ptr,
constexpr index_t bytes = sizeof(T) * N;
static_assert(bytes == 4 || bytes == 12 || bytes == 16,
```

**`include/ck_tile/core/tensor/buffer_view.hpp`**
```
const int32x4_t src_wave_buffer_resource =
make_wave_buffer_resource(p_data_, (buffer_size_) * sizeof(type));
src_wave_buffer_resource,
```
