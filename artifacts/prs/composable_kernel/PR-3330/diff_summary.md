# Diff summary

- **files changed:** 14
- **lines:** +667 / -84
- **kernel-ish files:** 12

## Files (by churn)

- `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`  (+228/-14)
- `example/ck_tile/01_fmha/fmha_fwd_runner.hpp`  (+180/-50)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp`  (+75/-8)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async.hpp`  (+75/-8)
- `include/ck_tile/host/reference/reference_batched_gemm.hpp`  (+40/-0)
- `example/ck_tile/01_fmha/fmha_fwd.hpp`  (+26/-0)
- `include/ck_tile/core/utility/functional.hpp`  (+12/-0)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`  (+5/-2)
- `example/ck_tile/01_fmha/quant.hpp`  (+7/-0)
- `include/ck_tile/core/numeric/math.hpp`  (+7/-0)
- `include/ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp`  (+6/-0)
- `example/ck_tile/01_fmha/script/smoke_test_fwd.sh`  (+3/-2)
- `example/ck_tile/01_fmha/codegen/cpp_symbol_map.py`  (+2/-0)
- `CHANGELOG.md`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/cpp_symbol_map.py`**
```
"blockscale": "ck_tile::BlockAttentionQuantScaleEnum::BLOCKSCALE",
"blockscale": "quant_scale_enum::blockscale",
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`**
```
["no", "pertensor", "blockscale"],
["no", "pertensor", "blockscale"],
get_mask_map(mask_impl).keys(),
```

**`example/ck_tile/01_fmha/fmha_fwd.hpp`**
```
const void* block_scale_seqstart_q_ptr;
const void* block_scale_seqstart_k_ptr;
ck_tile::index_t nhead_stride_q_descale;
ck_tile::index_t nhead_stride_k_descale;
```

**`example/ck_tile/01_fmha/fmha_fwd_runner.hpp`**
```
constexpr ck_tile::index_t block_scale_size_q_  = 128;
constexpr ck_tile::index_t block_scale_size_kv_ = 128;
size_t i_block_scale_q                           = 0;
size_t i_block_scale_k                           = 0;
```

**`example/ck_tile/01_fmha/quant.hpp`**
```
blockscale,
else if(type == quant_scale_enum::blockscale)
os << "bs";
else if(str == "bs" || str == "2")
```
