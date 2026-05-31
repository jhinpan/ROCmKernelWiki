# Diff summary

- **files changed:** 8
- **lines:** +88 / -29
- **kernel-ish files:** 8

## Files (by churn)

- `include/ck_tile/ops/flatmm/kernel/moe_flatmm_kernel.hpp`  (+37/-9)
- `include/ck_tile/ops/flatmm/pipeline/mixed_prec_flatmm_pipeline_agmem_bgmem_creg_v1.hpp`  (+11/-4)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp`  (+9/-4)
- `include/ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1.hpp`  (+12/-0)
- `include/ck_tile/ops/flatmm/pipeline/mx_flatmm_pipeline_agmem_bgmem_creg_v1.hpp`  (+8/-4)
- `include/ck_tile/core/arch/amd_buffer_addressing.hpp`  (+4/-4)
- `include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp`  (+4/-4)
- `include/ck_tile/ops/flatmm/pipeline/moe_flatmm_pipeline_agmem_bgmem_creg.hpp`  (+3/-0)

## Key added lines (kernel files)

**`include/ck_tile/core/arch/amd_buffer_addressing.hpp`**
```
DEVICE_NT0 = 16,
DEVICE_NT1 = 18,
SYSTEM_NT0 = 17,
SYSTEM_NT1 = 19,
```

**`include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp`**
```
DEVICE_NT0 = 16,
DEVICE_NT1 = 18,
SYSTEM_NT0 = 17,
SYSTEM_NT1 = 19,
```

**`include/ck_tile/ops/flatmm/kernel/moe_flatmm_kernel.hpp`**
```
if constexpr(!FlatmmPipeline::BPreShufflePermute)
index_t kFlatK =
kargs.K * BlockGemmShape::WarpTile::at(I1); // TODO (support splitK)
index_t kFlatN = kargs.N * kargs.K / kFlatK;
```

**`include/ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1.hpp`**
```
CK_TILE_HOST static constexpr amd_buffer_coherence_enum
GetBMemNTType(index_t M, index_t N, index_t K)
ck_tile::ignore = N;
ck_tile::ignore = K;
```

**`include/ck_tile/ops/flatmm/pipeline/mixed_prec_flatmm_pipeline_agmem_bgmem_creg_v1.hpp`**
```
GemmPipelineScheduler Scheduler_      = GemmPipelineScheduler::Intrawave,
bool HasHotLoop_                      = true,
TailNumber TailNum_                   = TailNumber::Full,
amd_buffer_coherence_enum BMemNTType_ = amd_buffer_coherence_enum::coherence_default,
```
