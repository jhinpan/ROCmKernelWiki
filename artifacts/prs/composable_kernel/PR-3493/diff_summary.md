# Diff summary

- **files changed:** 13
- **lines:** +777 / -940
- **kernel-ish files:** 13

## Files (by churn)

- `include/ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_v2.hpp`  (+249/-568)
- `include/ck_tile/ops/gemm/block/block_wp_asmem_breg_creg.hpp`  (+212/-0)
- `include/ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp`  (+61/-114)
- `include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`  (+15/-124)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp`  (+108/-19)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v4.hpp`  (+21/-29)
- `include/ck_tile/ops/gemm_quant/kernel/grouped_gemm_quant_kernel.hpp`  (+20/-29)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_wp_bquant_pipeline_ag_bg_cr_base_policy.hpp`  (+42/-0)
- `include/ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp`  (+11/-23)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp`  (+14/-15)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_wp_bquant_pipeline_ag_bg_cr_v2.hpp`  (+10/-13)
- `include/ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_base_policy.hpp`  (+13/-6)
- `include/ck_tile/ops/gemm.hpp`  (+1/-0)

## Key added lines (kernel files)

**`include/ck_tile/ops/gemm/block/block_wp_asmem_breg_creg.hpp`**
```
namespace ck_tile {
template <typename Problem_, typename BlockPolicy_>
struct BlockWeightPreshuffleASmemBRegCReg
using Problem        = remove_cvref_t<Problem_>;
```

**`include/ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp`**
```
__shared__ char smem_ptr[GetSmemSize()];
RunGemmWithPipelineSelection2LDS(
a_ptr, b_ptr, c_ptr, kargs.ds_ptr, smem_ptr, kargs, splitk_batch_offset, i_m, i_n);
smem_ptr,
```

**`include/ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp`**
```
void* smem_ptr,
as_block_window, AElementWise{}, bs_block_window, BElementWise{}, num_loop, smem_ptr);
EpiloguePipeline{}(c_block_window, c_block_tile, ds_block_window, smem_ptr);
EpiloguePipeline{}(c_block_window, c_block_tile, ds_block_window, smem_ptr);
```

**`include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp`**
```
template <typename SrcDataType = void,
typename DstDataType = void,
index_t UnaryOpSize  = 8,
typename DstBlockTile,
```

**`include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp`**
```
static_assert(DoubleSmemBuffer == true, "pipeline requires double smem buffer");
constexpr index_t smem_size = Policy::template GetSmemSize<Problem>();
return 2 * smem_size;
void* __restrict__ p_smem) const
```
