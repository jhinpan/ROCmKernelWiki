# Diff summary

- **files changed:** 12
- **lines:** +153 / -107
- **kernel-ish files:** 12

## Files (by churn)

- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`  (+27/-15)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`  (+26/-15)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_qr_ks_vs.hpp`  (+12/-11)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp`  (+12/-11)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async.hpp`  (+12/-11)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qs_ks_vs.hpp`  (+12/-11)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_fp8.hpp`  (+11/-11)
- `include/ck_tile/ops/fmha/pipeline/tile_fmha_shape.hpp`  (+18/-2)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp`  (+9/-6)
- `include/ck_tile/core/numeric/math.hpp`  (+6/-6)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`  (+4/-4)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_kernel.hpp`  (+4/-4)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`**
```
K0_MAX_SUBMAX_MAP = {
96 : 128,
128: 128,
using fmha_block_tile_{F_idx} = ck_tile::sequence<{F_bm0}, {F_bn0}, {F_bk0}, {F_bn1}, {F_bk1}, {F_bk0max}>;
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`**
```
K0_MAX_SUBMAX_MAP = {
96 : 128,
128: 128,
using fmha_block_tile = ck_tile::sequence<{F_bm0}, {F_bn0}, {F_bk0}, {F_bn1}, {F_bk1}, {F_bk0max}>;
```

**`include/ck_tile/core/numeric/math.hpp`**
```
return -x;
return __ocml_sin_f16(x);
return __ocml_ceil_f16(x);
return __ocml_floor_f16(x);
```

**`include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`**
```
_SS_("fmha_fwd_d") + _TS_(bfs::kQKHeaddim) + "_" + _SS_(t2s<QDataType>::name) +
_TS_(bfs::kN1) + "x" + _TS_(bfs::kK1) + "x" + _TS_(bfs::kQKHeaddim) + "_" +
make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kSubQKHeaddim>{}),
number<FmhaPipeline::kSubQKHeaddim>{});
```

**`include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_kernel.hpp`**
```
_SS_("fmha_fwd_splitkv_d") + _TS_(bfs::kQKHeaddim) + "_" + _SS_(t2s<QDataType>::name) +
_TS_(bfs::kN1) + "x" + _TS_(bfs::kK1) + "x" + _TS_(bfs::kQKHeaddim) + "_" +
make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kSubQKHeaddim>{}),
number<FmhaPipeline::kSubQKHeaddim>{});
```
