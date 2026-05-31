# Diff summary

- **files changed:** 24
- **lines:** +3288 / -16
- **kernel-ish files:** 11

## Files (by churn)

- `tutorial/ck_tile/01_naive_gemm/HOST_LEVEL_PIPELINE.md`  (+618/-0)
- `tutorial/ck_tile/01_naive_gemm/BLOCK_LEVEL_PIPELINE.md`  (+589/-0)
- `tutorial/ck_tile/01_naive_gemm/WALKTHROUGH.md`  (+506/-0)
- `tutorial/ck_tile/01_naive_gemm/KERNEL_ENTRY_POINT.md`  (+464/-0)
- `tutorial/ck_tile/01_naive_gemm/warp_level/practice_gemm_warp_pipeline_asmem_bsmem_creg.hpp`  (+195/-0)
- `tutorial/ck_tile/01_naive_gemm/block_level/practice_gemm_block_pipeline_agmem_bgmem_creg.hpp`  (+165/-0)
- `tutorial/ck_tile/01_naive_gemm/README.md`  (+150/-0)
- `tutorial/ck_tile/01_naive_gemm/block_level/practice_gemm_block_policy_agmem_bgmem_creg.hpp`  (+135/-0)
- `tutorial/ck_tile/01_naive_gemm/practice_gemm.cpp`  (+131/-0)
- `tutorial/ck_tile/01_naive_gemm/host_level/practice_gemm_host_pipeline_agmem_bgmem_creg.hpp`  (+92/-0)
- `tutorial/ck_tile/01_naive_gemm/practice_gemm.hpp`  (+69/-0)
- `tutorial/ck_tile/01_naive_gemm/host_level/practice_gemm_host_policy_agmem_bgmem_creg.hpp`  (+51/-0)
- `tutorial/ck_tile/01_naive_gemm/reference_gemm.hpp`  (+36/-0)
- `tutorial/ck_tile/01_naive_gemm/warp_level/practice_gemm_warp_policy_asmem_bsmem_creg.hpp`  (+35/-0)
- `tutorial/ck_tile/00_copy_kernel/copy_basic.cpp`  (+12/-12)

## Key added lines (kernel files)

**`tutorial/ck_tile/00_copy_kernel/copy_basic.cpp`**
```
using ThreadTile = ck_tile::sequence<1, 4>;  // per-thread tile size along M and N
using WaveTile   = ck_tile::sequence<64, 4>; // per-wave tile size along M and N dimension
using BlockWaves = ck_tile::sequence<4, 1>; // number of waves per block along M and N dimension
using BlockTile  = ck_tile::sequence<512, 4>; // per-block tile size along M and N dimension
```

**`tutorial/ck_tile/01_naive_gemm/block_level/practice_gemm_block_pipeline_agmem_bgmem_creg.hpp`**
```
namespace ck_tile {
template <typename Problem, typename Policy = PracticeGemmBlockPolicy>
struct PracticeGemmBlockPipelineAGmemBGmemCreg
using ADataType   = typename Problem::ADataType;
```

**`tutorial/ck_tile/01_naive_gemm/block_level/practice_gemm_block_policy_agmem_bgmem_creg.hpp`**
```
namespace ck_tile {
template <typename ADataType_,
typename BDataType_,
typename CDataType_,
```

**`tutorial/ck_tile/01_naive_gemm/host_level/practice_gemm_host_pipeline_agmem_bgmem_creg.hpp`**
```
namespace ck_tile {
template <typename Problem_, typename Policy_ = PracticeGemmHostPolicy>
struct PracticeGemmHostPipeline
using ADataType   = typename Problem_::ADataType;
```

**`tutorial/ck_tile/01_naive_gemm/host_level/practice_gemm_host_policy_agmem_bgmem_creg.hpp`**
```
namespace ck_tile {
template <typename ADataType_,
typename BDataType_,
typename CDataType_,
```
