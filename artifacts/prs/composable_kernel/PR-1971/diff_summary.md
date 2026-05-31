# Diff summary

- **files changed:** 10
- **lines:** +37 / -10
- **kernel-ish files:** 10

## Files (by churn)

- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`  (+7/-2)
- `include/ck_tile/core/config.hpp`  (+8/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async.hpp`  (+7/-0)
- `include/ck_tile/core.hpp`  (+1/-4)
- `include/ck_tile/core/arch/amd_buffer_addressing.hpp`  (+4/-0)
- `include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp`  (+4/-0)
- `include/ck_tile/ops/fmha.hpp`  (+2/-2)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd_appendkv.py`  (+1/-1)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`  (+1/-1)
- `include/ck_tile/ops/fmha/pipeline/tile_fmha_shape.hpp`  (+2/-0)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`**
```
FMHA_FWD_API_PER_HDIM_CASE="""        {F_if} (t.hdim_q <= {F_hdim} && t.hdim_v <= {F_hdim_v}) {{
per_hdim_case = per_hdim_case + FMHA_FWD_API_PER_HDIM_CASE.format(F_if=if_j, F_hdim=hdim, F_hdim_v=trait.bn1, F_inner_di
'192' : FmhaFwdTileSize(128, 128, 32, 128, 32,  192,  4, 1, 1,  4, 1, 1,  32, 32, 16,  32, 32, 16,  -1),
if hdim == 192 and tile.F_bn1 == 128:
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd_appendkv.py`**
```
per_hdim_case = per_hdim_case + FMHA_FWD_API_PER_HDIM_CASE.format(F_if=if_j, F_hdim=hdim, F_hdim_v=hdim, F_inner_dispatc
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`**
```
per_hdim_case = per_hdim_case + FMHA_FWD_API_PER_HDIM_CASE.format(F_if=if_j, F_hdim=hdim, F_hdim_v=hdim, F_inner_dispatc
```

**`include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async.hpp`**
```
else if constexpr(kQKHeaddim <= 192)
if constexpr(kPadSeqLenK && BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
return 1;
return 2;
```

**`include/ck_tile/ops/fmha/pipeline/tile_fmha_shape.hpp`**
```
if(len == 192)
return 192;
```
