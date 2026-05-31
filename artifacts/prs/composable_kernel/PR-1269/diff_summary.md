# Diff summary

- **files changed:** 24
- **lines:** +948 / -115
- **kernel-ish files:** 19

## Files (by churn)

- `test/position_embedding/position_embedding.cpp`  (+215/-0)
- `include/ck_tile/ops/fmha/block/block_position_encoding.hpp`  (+189/-0)
- `example/ck_tile/01_fmha/fmha_fwd.cpp`  (+97/-19)
- `example/ck_tile/01_fmha/bias.hpp`  (+100/-0)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`  (+72/-13)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async.hpp`  (+44/-14)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp`  (+40/-11)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qs_ks_vs.hpp`  (+40/-11)
- `include/ck_tile/ops/fmha/block/block_attention_bias_enum.hpp`  (+37/-0)
- `example/ck_tile/01_fmha/generate.py`  (+26/-10)
- `example/ck_tile/01_fmha/README.md`  (+14/-9)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_fp8.hpp`  (+13/-9)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_enum.hpp`  (+19/-0)
- `example/ck_tile/01_fmha/mask.hpp`  (+6/-8)
- `example/ck_tile/01_fmha/fmha_fwd.hpp`  (+6/-5)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/bias.hpp`**
```
enum class bias_enum
no_bias          = 0,
elementwise_bias = 1,
alibi            = 2,
```

**`example/ck_tile/01_fmha/fmha_fwd.cpp`**
```
"num of head, for k/v, -1 means equal to h\n"
.insert("s_k", "-1", "seqlen_k, -1 means equal to s")
.insert("d_v", "-1", "head dim for v, -1 means equal to d")
.insert("bias",
```

**`example/ck_tile/01_fmha/fmha_fwd.hpp`**
```
const void* bias_ptr; // bias or alibi_slope pointer
ck_tile::index_t stride_bias; // if alibi, b*h need set this to h, 1*h need set this to 0
ck_tile::BlockAttentionBiasEnum BiasEnum_,
static constexpr auto BiasEnum                   = BiasEnum_;
```

**`example/ck_tile/01_fmha/generate.py`**
```
BIAS_MAP = {
"no" : "ck_tile::BlockAttentionBiasEnum::NO_BIAS",
"bias"  : "ck_tile::BlockAttentionBiasEnum::ELEMENTWISE_BIAS",
"alibi" : "ck_tile::BlockAttentionBiasEnum::ALIBI"
```

**`example/ck_tile/01_fmha/mask.hpp`**
```
friend std::ostream& operator<<(std::ostream& os, const mask_info& mi)
mi.serialize(os);
return os;
```
