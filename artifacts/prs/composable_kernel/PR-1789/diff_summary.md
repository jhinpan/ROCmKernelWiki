# Diff summary

- **files changed:** 25
- **lines:** +207 / -97
- **kernel-ish files:** 25

## Files (by churn)

- `include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_kernel.hpp`  (+131/-47)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`  (+38/-15)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_problem.hpp`  (+11/-10)
- `include/ck_tile/ops/fmha/pipeline/tile_fmha_traits.hpp`  (+5/-3)
- `example/ck_tile/01_fmha/fmha_fwd.hpp`  (+2/-2)
- `include/ck_tile/core.hpp`  (+1/-1)
- `include/ck_tile/host.hpp`  (+1/-1)
- `include/ck_tile/ops/add_rmsnorm2d_rdquant.hpp`  (+1/-1)
- `include/ck_tile/ops/common.hpp`  (+1/-1)
- `include/ck_tile/ops/elementwise.hpp`  (+1/-1)
- `include/ck_tile/ops/epilogue.hpp`  (+1/-1)
- `include/ck_tile/ops/flatmm.hpp`  (+1/-1)
- `include/ck_tile/ops/fmha.hpp`  (+1/-1)
- `include/ck_tile/ops/fused_moe.hpp`  (+1/-1)
- `include/ck_tile/ops/gemm.hpp`  (+1/-1)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`**
```
template <bool kHasUnevenSplits, bool kMergeNumHeadGroupsSeqLenQ = false>
struct instance {{
kMergeNumHeadGroupsSeqLenQ,
namespace {{
```

**`example/ck_tile/01_fmha/fmha_fwd.hpp`**
```
dim3 grids = Kernel::GridSize(
args.batch, args.nhead_q, args.nhead_k, args.max_seqlen_q, args.hdim_v, args.num_splits);
```

**`include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_kernel.hpp`**
```
static constexpr bool kMergeNumHeadGroupsSeqLenQ =
FmhaPipeline::Problem::kMergeNumHeadGroupsSeqLenQ;
static_assert(!kMergeNumHeadGroupsSeqLenQ ||
(kMergeNumHeadGroupsSeqLenQ && BiasEnum == BlockAttentionBiasEnum::NO_BIAS &&
```

**`include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_problem.hpp`**
```
static constexpr bool kPadSeqLenQ                = Traits::kPadSeqLenQ;
static constexpr bool kPadSeqLenK                = Traits::kPadSeqLenK;
static constexpr bool kPadHeadDimQ               = Traits::kPadHeadDimQ;
static constexpr bool kPadHeadDimV               = Traits::kPadHeadDimV;
```

**`include/ck_tile/ops/fmha/pipeline/tile_fmha_traits.hpp`**
```
bool kMergeNumHeadGroupsSeqLenQ_ = false,
index_t kBlockPerCu_             = -1 /* overwrite occupancy if not -1 */>
static constexpr bool kHasUnevenSplits           = kHasUnevenSplits_;
static constexpr bool kMergeNumHeadGroupsSeqLenQ = kMergeNumHeadGroupsSeqLenQ_;
```
