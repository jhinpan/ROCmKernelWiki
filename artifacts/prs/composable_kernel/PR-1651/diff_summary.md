# Diff summary

- **files changed:** 13
- **lines:** +781 / -222
- **kernel-ish files:** 12

## Files (by churn)

- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp`  (+267/-63)
- `include/ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp`  (+264/-52)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp`  (+116/-38)
- `include/ck_tile/ops/gemm/kernel/gemm_kernel.hpp`  (+50/-20)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1.hpp`  (+44/-19)
- `example/ck_tile/03_gemm/gemm_basic.cpp`  (+10/-9)
- `include/ck_tile/ops/gemm/pipeline/tile_gemm_traits.hpp`  (+10/-6)
- `test/ck_tile/gemm/test_gemm_mem_pipeline_util.hpp`  (+6/-6)
- `example/ck_tile/03_gemm/gemm_mem_pipeline.cpp`  (+5/-5)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_mem.hpp`  (+3/-3)
- `example/ck_tile/03_gemm/README.md`  (+3/-0)
- `include/ck_tile/core/tensor/shuffle_tile.hpp`  (+1/-1)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp`  (+2/-0)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_basic.cpp`**
```
constexpr bool kPadM = false;
constexpr bool kPadN = false;
constexpr bool kPadK = false;
ck_tile::Default2DEpilogueProblem<AccDataType, CDataType, kPadM, kPadN>>>;
```

**`example/ck_tile/03_gemm/gemm_mem_pipeline.cpp`**
```
constexpr bool kPadM = true;
constexpr bool kPadN = true;
constexpr bool kPadK = true;
ck_tile::Default2DEpilogueProblem<AccDataType, CDataType, kPadM, kPadN>>;
```

**`include/ck_tile/core/tensor/shuffle_tile.hpp`**
```
static_assert(false, "The shuffle should always happen!");
```

**`include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp`**
```
static_assert(N2 != 0, "N2 is zero, which will lead to a division by zero error.");
static_assert(N1 != 0, "N1 is zero, which will lead to a division by zero error.");
```

**`include/ck_tile/ops/gemm/kernel/gemm_kernel.hpp`**
```
auto a_pad_view = [&]() {
if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
return pad_tensor_view(
a_tensor_view,
```
