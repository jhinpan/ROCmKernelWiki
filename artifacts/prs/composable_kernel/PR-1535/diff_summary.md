# Diff summary

- **files changed:** 15
- **lines:** +447 / -142
- **kernel-ish files:** 15

## Files (by churn)

- `include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp`  (+171/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp`  (+82/-51)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp`  (+50/-31)
- `example/ck_tile/03_gemm/gemm_basic.cpp`  (+40/-15)
- `include/ck_tile/host/reference/reference_gemm.hpp`  (+37/-10)
- `include/ck_tile/ops/gemm/pipeline/tile_gemm_traits.hpp`  (+27/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp`  (+10/-7)
- `include/ck_tile/ops/gemm/kernel/gemm_kernel.hpp`  (+6/-9)
- `include/ck_tile/ops/gemm.hpp`  (+6/-5)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1.hpp`  (+7/-3)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v2_default_policy.hpp`  (+4/-5)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v2.hpp`  (+3/-3)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp`  (+2/-2)
- `include/ck_tile/core/container/thread_buffer.hpp`  (+1/-1)
- `include/ck_tile/ops/epilogue.hpp`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_basic.cpp`**
```
constexpr bool kPadA        = true;
constexpr bool kPadB        = true;
constexpr bool kTilePermute = false;
constexpr ck_tile::index_t kOutputRank = 2;
```

**`include/ck_tile/host/reference/reference_gemm.hpp`**
```
const int N = (std::is_same_v<LayoutB, tensor_layout::gemm::ColumnMajor>)
? b_n_k.mDesc.get_lengths()[0]
: b_n_k.mDesc.get_lengths()[1];
BDataType v_b = (std::is_same_v<LayoutB, tensor_layout::gemm::ColumnMajor>)
```

**`include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp`**
```
namespace ck_tile {
template <typename AccDataType_,
typename ODataType_,
bool kPadM_,
```

**`include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp`**
```
using GemmProblem =
GemmPipelineProblem<typename Problem::QDataType,
typename Problem::KDataType,
typename Problem::AccDataType,
```

**`include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp`**
```
using GemmProblem =
GemmPipelineProblem<typename Problem::QDataType,
typename Problem::KDataType,
typename Problem::SaccDataType,
```
