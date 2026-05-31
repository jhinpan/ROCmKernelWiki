# Diff summary

- **files changed:** 21
- **lines:** +761 / -136
- **kernel-ish files:** 18

## Files (by churn)

- `test/ck_tile/memory_copy/test_copy.hpp`  (+157/-3)
- `include/ck_tile/core/numeric/pk_fp6.hpp`  (+109/-0)
- `include/ck_tile/ops/flatmm/pipeline/mx_flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp`  (+78/-31)
- `test/ck_tile/memory_copy/test_copy.cpp`  (+76/-25)
- `include/ck_tile/ops/flatmm/pipeline/mx_flatmm_pipeline_agmem_bgmem_creg_v1.hpp`  (+50/-40)
- `include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp`  (+53/-8)
- `include/ck_tile/host/check_err.hpp`  (+53/-0)
- `example/ck_tile/18_flatmm/mxgemm/run_mx_flatmm.inc`  (+37/-14)
- `include/ck_tile/core/tensor/buffer_view.hpp`  (+29/-6)
- `include/ck_tile/core/numeric/vector_type.hpp`  (+34/-0)
- `example/ck_tile/18_flatmm/mxgemm/mx_flatmm.hpp`  (+32/-0)
- `include/ck_tile/host/reference/reference_gemm.hpp`  (+22/-0)
- `example/ck_tile/18_flatmm/mxgemm/mx_flatmm.cpp`  (+13/-5)
- `include/ck_tile/core/arch/amd_buffer_addressing.hpp`  (+10/-1)
- `example/ck_tile/18_flatmm/mxgemm/mx_flatmm_instance.cmake`  (+2/-1)

## Key added lines (kernel files)

**`example/ck_tile/18_flatmm/mxgemm/mx_flatmm.cpp`**
```
int KPack =
std::is_same_v<dtype, ck_tile::pk_fp6x16_t> ? 32 : 16 * packed_size; // fp4/fp6:32 or fp8:16
int NLane = N_Warp_Tile;
int KLane = 64 / NLane;
```

**`example/ck_tile/18_flatmm/mxgemm/mx_flatmm.hpp`**
```
struct MXfp6_FlatmmConfig16
static constexpr ck_tile::index_t M_Tile = 128;
static constexpr ck_tile::index_t N_Tile = 256;
static constexpr ck_tile::index_t K_Tile = 256;
```

**`example/ck_tile/18_flatmm/mxgemm/run_mx_flatmm.inc`**
```
if constexpr(std::is_same_v<ADataType, ck_tile::pk_fp6x16_t>)
auto a_buffer_bytes = a_host.get_element_space_size_in_bytes();
auto b_buffer_bytes = b_origin_host.get_element_space_size_in_bytes();
ck_tile::FillUniformDistribution<>{-1.f, 1.f}(scale_a);
```

**`include/ck_tile/core/arch/amd_buffer_addressing.hpp`**
```
static_assert(N == 1 || N == 2 || N == 4 || N == 8 || N == 12 || N == 16 || N == 32 || N == 64,
else if constexpr(N == 12)
auto tmp = llvm_amdgcn_raw_buffer_load_i32x3(src_wave_buffer_resource,
src_thread_addr_offset,
```

**`include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp`**
```
CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_i32x3_(int32x3_t vdata,
int32x4_t rsrc,
index_t voffset,
```
