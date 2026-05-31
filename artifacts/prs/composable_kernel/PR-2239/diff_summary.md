# Diff summary

- **files changed:** 10
- **lines:** +319 / -204
- **kernel-ish files:** 9

## Files (by churn)

- `include/ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp`  (+73/-54)
- `include/ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1.hpp`  (+61/-58)
- `example/ck_tile/18_flatmm/flatmm_basic.hpp`  (+66/-43)
- `example/ck_tile/18_flatmm/run_flatmm_example.inc`  (+62/-29)
- `example/ck_tile/18_flatmm/flatmm_basic.cpp`  (+35/-9)
- `include/ck_tile/ops/gemm/kernel/gemm_kernel.hpp`  (+11/-2)
- `tile_engine/ops/gemm/gemm_instance_builder.py`  (+6/-5)
- `tile_engine/ops/gemm/gemm_profiler.hpp`  (+2/-2)
- `tile_engine/ops/gemm/codegen_utils.py`  (+3/-0)
- `example/ck_tile/18_flatmm/CMakeLists.txt`  (+0/-2)

## Key added lines (kernel files)

**`example/ck_tile/18_flatmm/flatmm_basic.cpp`**
```
typename FlatmmConfig,
using CodegenGemmTraits = ck_tile::TileGemmTraits<FlatmmConfig::kPadM,
const auto Run = [&](const auto memory_operation_) {
template <template <typename PreType> typename FlatmmConfig>
```

**`example/ck_tile/18_flatmm/flatmm_basic.hpp`**
```
template <typename DataType>
struct FlatmmConfig32
static constexpr ck_tile::index_t M_Tile = 128;
static constexpr ck_tile::index_t N_Tile = 128;
```

**`example/ck_tile/18_flatmm/run_flatmm_example.inc`**
```
typename FlatmmConfig,
float ave_time = flatmm_calc<ADataType,
BDataType,
AccDataType,
```

**`include/ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1.hpp`**
```
if constexpr(WG::kM == 16 && WG::kN == 16)
static_for<0, A_Buffer_Load_Inst_Num, 1>{}([&](auto i) {
ignore = i;
__builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
```

**`include/ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp`**
```
constexpr index_t MPerXdl = Problem::BlockGemmShape::WarpTile::at(I0);
constexpr index_t NPerXdl = Problem::BlockGemmShape::WarpTile::at(I1);
if constexpr(MPerXdl == 16 && NPerXdl == 16)
constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
```
