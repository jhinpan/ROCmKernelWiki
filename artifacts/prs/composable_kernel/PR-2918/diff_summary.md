# Diff summary

- **files changed:** 12
- **lines:** +110 / -88
- **kernel-ish files:** 11

## Files (by churn)

- `example/ck_tile/01_fmha/codegen/ops/fmha_bwd.py`  (+32/-29)
- `include/ck_tile/ops/fmha/kernel/fmha_bwd_kernel.hpp`  (+15/-15)
- `include/ck_tile/ops/fmha/pipeline/tile_fmha_traits.hpp`  (+17/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp`  (+8/-8)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr_iglp.hpp`  (+8/-8)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_trload_kr_ktr_vr.hpp`  (+8/-8)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_trload_qr_qtr_dor.hpp`  (+8/-8)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_problem.hpp`  (+5/-7)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_selector.hpp`  (+3/-2)
- `example/ck_tile/01_fmha/fmha_bwd.hpp`  (+2/-2)
- `test/ck_tile/fmha/test_fmha_bwd.inc`  (+3/-0)
- `example/ck_tile/01_fmha/script/smoke_test_bwd.sh`  (+1/-1)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/ops/fmha_bwd.py`**
```
using fmha_bwd_trait_{F_idx} = ck_tile::TileFmhaBwdTraits<{F_dpad},
({F_dpad} > 0)>>;
({F_dvpad} > 0)>>;
({F_dpad} > 0)>>;
```

**`example/ck_tile/01_fmha/fmha_bwd.hpp`**
```
ck_tile::index_t kPadD_,
ck_tile::index_t kPadDv_,
```

**`include/ck_tile/ops/fmha/kernel/fmha_bwd_kernel.hpp`**
```
static constexpr bool kIsGroupMode    = FmhaPipeline::kIsGroupMode;
static constexpr index_t kPadHeadDimQ = FmhaPipeline::kPadHeadDimQ;
static constexpr index_t kPadHeadDimV = FmhaPipeline::kPadHeadDimV;
static constexpr auto BiasEnum        = FmhaPipeline::BiasEnum;
```

**`include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp`**
```
static constexpr index_t kPadHeadDimQ  = Problem::kPadHeadDimQ;
static constexpr index_t kPadHeadDimV  = Problem::kPadHeadDimV;
kPadHeadDimQ ? kPadHeadDimQ : Policy::template GetAlignmentQ<Problem>();
kPadHeadDimQ ? kPadHeadDimQ : Policy::template GetAlignmentK<Problem>();
```

**`include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr_iglp.hpp`**
```
static constexpr index_t kPadHeadDimQ  = Problem::kPadHeadDimQ;
static constexpr index_t kPadHeadDimV  = Problem::kPadHeadDimV;
kPadHeadDimQ ? kPadHeadDimQ : Policy::template GetAlignmentQ<Problem>();
kPadHeadDimQ ? kPadHeadDimQ : Policy::template GetAlignmentK<Problem>();
```
