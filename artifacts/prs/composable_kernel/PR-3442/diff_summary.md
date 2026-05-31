# Diff summary

- **files changed:** 9
- **lines:** +959 / -406
- **kernel-ish files:** 8

## Files (by churn)

- `include/ck_tile/ops/fmha/pipeline/block_fmha_batch_prefill_pipeline_qr_ks_vs_async.hpp`  (+376/-234)
- `include/ck_tile/ops/fmha/kernel/fmha_batch_prefill_kernel.hpp`  (+194/-78)
- `example/ck_tile/01_fmha/codegen/ops/fmha_batch_prefill.py`  (+141/-75)
- `example/ck_tile/01_fmha/fmha_fwd.hpp`  (+105/-19)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_problem.hpp`  (+66/-0)
- `include/ck_tile/ops/fmha/pipeline/tile_fmha_traits.hpp`  (+43/-0)
- `include/ck_tile/ops/fmha/block/block_attention_kvcache_layout_enum.hpp`  (+32/-0)
- `CHANGELOG.md`  (+1/-0)
- `include/ck_tile/ops/fmha.hpp`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/ops/fmha_batch_prefill.py`**
```
SUPPORTED_PAGE_SIZE = [128, 256, 1024]
SUPPORTED_KV_MEMORY_LAYOUT = ["vectorized", "linear"]
SUPPORTED_KV_LOOKUP_TABLE = ["vllm", "sglang"]
KV_MEMORY_LAYOUT_ENUM_MAP = {
```

**`example/ck_tile/01_fmha/fmha_fwd.hpp`**
```
int32_t num_total_pages;          // total physical pages in KV cache (SGLang/vLLM)
ck_tile::index_t page_block_size; // tokens per page (SGLang/vLLM)
ck_tile::BlockAttentionKVCacheMemoryLayoutEnum
kv_memory_layout;                                          // KV memory layout (SGLang/vLLM)
```

**`include/ck_tile/ops/fmha/block/block_attention_kvcache_layout_enum.hpp`**
```
namespace ck_tile {
enum class BlockAttentionKVCacheMemoryLayoutEnum
VECTORIZED_LAYOUT = 0,
LINEAR_LAYOUT     = 1,
```

**`include/ck_tile/ops/fmha/kernel/fmha_batch_prefill_kernel.hpp`**
```
static constexpr auto kKVMemoryLayout   = FmhaPipeline::Problem::kKVMemoryLayout;
static constexpr auto kKVLookupTable    = FmhaPipeline::Problem::kKVLookupTable;
static constexpr index_t kPageBlockSize = FmhaPipeline::kPageBlockSize;
static constexpr index_t kVectorSize    = FmhaPipeline::kVectorSize;
```

**`include/ck_tile/ops/fmha/pipeline/block_fmha_batch_prefill_pipeline_qr_ks_vs_async.hpp`**
```
template <typename OffsetVecType,
typename CoordVecType,
index_t kCoordAxis,
index_t kPageBlockSize,
```
