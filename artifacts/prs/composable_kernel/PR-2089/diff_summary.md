# Diff summary

- **files changed:** 14
- **lines:** +1803 / -0
- **kernel-ish files:** 10

## Files (by churn)

- `include/ck_tile/ops/flatmm/kernel/flatmm_kernel.hpp`  (+496/-0)
- `example/ck_tile/18_flatmm/run_flatmm_example.inc`  (+281/-0)
- `include/ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp`  (+265/-0)
- `include/ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1.hpp`  (+208/-0)
- `include/ck_tile/ops/flatmm/block/block_flatmm_asmem_bsmem_creg_v1.hpp`  (+187/-0)
- `example/ck_tile/18_flatmm/flatmm_basic.cpp`  (+102/-0)
- `example/ck_tile/18_flatmm/flatmm_basic.hpp`  (+100/-0)
- `include/ck_tile/ops/flatmm/pipeline/tile_flatmm_shape.hpp`  (+43/-0)
- `include/ck_tile/ops/flatmm/block/block_flatmm_asmem_bsmem_creg_v1_custom_policy.hpp`  (+38/-0)
- `example/ck_tile/18_flatmm/README.md`  (+35/-0)
- `example/ck_tile/18_flatmm/script/smoke_test_basic.sh`  (+34/-0)
- `example/ck_tile/18_flatmm/CMakeLists.txt`  (+7/-0)
- `include/ck_tile/ops/flatmm.hpp`  (+6/-0)
- `example/ck_tile/CMakeLists.txt`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/18_flatmm/flatmm_basic.cpp`**
```
template <typename ALayout, typename BLayout, typename CLayout>
float flatmm_calc(const ck_tile::FlatmmHostArgs& args, const ck_tile::stream_config& s)
constexpr bool kPadM = false;
constexpr bool kPadN = false;
```

**`example/ck_tile/18_flatmm/flatmm_basic.hpp`**
```
template <typename DataType>
struct GemmBasicTypeConfig;
template <>
struct GemmBasicTypeConfig<ck_tile::half_t>
```

**`example/ck_tile/18_flatmm/run_flatmm_example.inc`**
```
template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
ck_tile::tensor_layout::gemm::RowMajor>>{};
```

**`include/ck_tile/ops/flatmm/block/block_flatmm_asmem_bsmem_creg_v1.hpp`**
```
namespace ck_tile {
template <typename Problem_, typename BlockPolicy_>
struct BlockFlatmmASmemBSmemCRegV1
using Problem        = remove_cvref_t<Problem_>;
```

**`include/ck_tile/ops/flatmm/block/block_flatmm_asmem_bsmem_creg_v1_custom_policy.hpp`**
```
namespace ck_tile {
template <typename AType_,
typename BType_,
typename CType_,
```
