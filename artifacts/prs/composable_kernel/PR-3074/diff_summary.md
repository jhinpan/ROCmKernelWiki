# Diff summary

- **files changed:** 24
- **lines:** +473 / -360
- **kernel-ish files:** 23

## Files (by churn)

- `example/ck_tile/38_block_scale_gemm/gemm_quant_basic.cpp`  (+114/-151)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_group_quant_utils.hpp`  (+82/-24)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_fixtures.hpp`  (+34/-31)
- `include/ck_tile/ops/gemm_quant/block/block_universal_gemm_as_bs_bquant_cr.hpp`  (+27/-31)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_typed.cpp`  (+42/-5)
- `example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`  (+28/-1)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_policy.hpp`  (+14/-12)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_quant_pipeline_problem.hpp`  (+12/-11)
- `include/ck_tile/ops/gemm_quant/block/block_universal_gemm_as_aquant_bs_cr.hpp`  (+11/-11)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_base.hpp`  (+11/-11)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_v3.hpp`  (+12/-9)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_mem.hpp`  (+11/-8)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_v3.hpp`  (+11/-8)
- `include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`  (+10/-7)
- `example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`  (+8/-8)

## Key added lines (kernel files)

**`example/ck_tile/38_block_scale_gemm/gemm_quant_basic.cpp`**
```
typename QuantGroupSize,
ck_tile::BaseBQuantGemmPipelineAgBgCrCompV3<GemmPipelineProblem>>;
typename QuantGroupSize,
template <template <typename PreType> typename GemmConfig, typename QuantGroupSize>
```

**`example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`**
```
X(1, 1, 64)    /* 1D */               \
X(1, 1, 128)   /* 1D */               \
X(1, 8, 128)   /* 2D N=8  */          \
X(1, 32, 128)  /* 2D N=32 */          \
```

**`example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`**
```
typename QuantGroupSize,
typename QuantGroupSize,
if(K % QuantGroupSize::kK != 0)
AQK = K / QuantGroupSize::kK; // Group quantization: AQK = K / GroupSize
```

**`include/ck_tile/host/reference/reference_gemm.hpp`**
```
typename QuantGroupSize,
if((k + 1) % QuantGroupSize::kK == 0)
index_t outer_dim = (aquant) ? (m / QuantGroupSize::kM) : (k / QuantGroupSize::kK);
index_t inner_dim = (aquant) ? (k / QuantGroupSize::kK) : (n / QuantGroupSize::kN);
```

**`include/ck_tile/ops/gemm_quant/block/block_universal_gemm_ar_flatbr_bquant_cr.hpp`**
```
using QuantGroupSize  = remove_cvref_t<typename Problem::QuantGroupSize>;
static_assert(QuantGroupSize::kM == 1, "only N/K blocks for BQuant preshuffle kernel!");
static_assert(QuantGroupSize::kN == 1, "no block for N supported yet!");
static constexpr index_t kBlockSize = Problem::kBlockSize;
```
