# Diff summary

- **files changed:** 18
- **lines:** +403 / -366
- **kernel-ish files:** 18

## Files (by churn)

- `include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp`  (+146/-151)
- `include/ck_tile/ops/epilogue/default_2d_epilogue.hpp`  (+97/-4)
- `include/ck_tile/ops/gemm/kernel/gemm_kernel.hpp`  (+17/-32)
- `test/ck_tile/grouped_gemm/test_grouped_gemm_util.hpp`  (+20/-28)
- `example/ck_tile/17_grouped_gemm/grouped_gemm.cpp`  (+20/-26)
- `test/ck_tile/batched_gemm/test_batched_gemm_util.hpp`  (+17/-26)
- `example/ck_tile/16_batched_gemm/batched_gemm.cpp`  (+16/-26)
- `example/ck_tile/03_gemm/gemm_basic.cpp`  (+13/-24)
- `test/ck_tile/gemm/test_gemm_pipeline_util.hpp`  (+16/-5)
- `example/ck_tile/03_gemm/universal_gemm.cpp`  (+14/-4)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp`  (+7/-10)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_mem.hpp`  (+6/-9)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1.hpp`  (+5/-5)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp`  (+1/-5)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v2.hpp`  (+3/-3)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_basic.cpp`**
```
using GemmEpilogue        = ck_tile::CShuffleEpilogue<
ck_tile::CShuffleEpilogueProblem<AccDataType,
CDataType,
CodegenPipelineProblem::kBlockSize,
```

**`example/ck_tile/03_gemm/universal_gemm.cpp`**
```
using GemmEpilogue = ck_tile::CShuffleEpilogue<
ck_tile::CShuffleEpilogueProblem<AccDataType,
CDataType,
GemmPipelineProblem::kBlockSize,
```

**`example/ck_tile/16_batched_gemm/batched_gemm.cpp`**
```
constexpr bool kPadM = false;
constexpr bool kPadN = false;
constexpr bool kPadK = false;
using GemmEpilogue        = ck_tile::CShuffleEpilogue<
```

**`example/ck_tile/17_grouped_gemm/grouped_gemm.cpp`**
```
static const bool kPadM = false;
static const bool kPadN = false;
static const bool kPadK = false;
template <typename ALayout, typename BLayout, typename CLayout>
```

**`include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp`**
```
typename CLayout_,
index_t kBlockSize_,
index_t kM_,
index_t kN_,
```
