# Diff summary

- **files changed:** 22
- **lines:** +422 / -199
- **kernel-ish files:** 22

## Files (by churn)

- `include/ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_one_warp_v1.hpp`  (+237/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp`  (+46/-85)
- `example/ck_tile/01_fmha/fmha_fwd.cpp`  (+25/-39)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`  (+23/-18)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`  (+18/-14)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_qr_ks_vs_default_policy.hpp`  (+15/-6)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_problem.hpp`  (+10/-4)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_qr_ks_vs.hpp`  (+8/-5)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp`  (+5/-5)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async.hpp`  (+5/-5)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_tile_partitioner.hpp`  (+5/-4)
- `include/ck_tile/ops/fmha/pipeline/tile_fmha_shape.hpp`  (+7/-2)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_kernel.hpp`  (+5/-3)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`  (+4/-2)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qs_ks_vs.hpp`  (+2/-1)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`**
```
ck_tile::sequence<{F_rm0}, {F_rn0}, {F_rk0}>,
ck_tile::sequence<{F_rm1}, {F_rn1}, {F_rk1}>,
F_rm0       : int  # number of warps for gemm0 along q seqlen
F_rn0       : int  # number of warps for gemm0 along k seqlen
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`**
```
ck_tile::sequence<{F_rm0}, {F_rn0}, {F_rk0}>,
ck_tile::sequence<{F_rm1}, {F_rn1}, {F_rk1}>,
false, false>>;
F_rm0           = self.F_tile.F_rm0,
```

**`example/ck_tile/01_fmha/fmha_fwd.cpp`**
```
static const auto get_lengths = [](bool permute,
ck_tile::index_t b /*batch*/,
ck_tile::index_t h /*nhead*/,
ck_tile::index_t s /*seqlen*/,
```

**`include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`**
```
using g0br = typename bfs::Gemm0BlockWarps;
using g1br = typename bfs::Gemm1BlockWarps;
"r" + _TS_(g0br::at(ck_tile::number<0>{})) + "x" + _TS_(g0br::at(ck_tile::number<1>{})) + "x" + _TS_(g0br::at(ck_tile::n
"r" + _TS_(g1br::at(ck_tile::number<0>{})) + "x" + _TS_(g1br::at(ck_tile::number<1>{})) + "x" + _TS_(g1br::at(ck_tile::n
```

**`include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_kernel.hpp`**
```
using g0br = typename bfs::Gemm0BlockWarps;
using g1br = typename bfs::Gemm1BlockWarps;
"r" + _TS_(g0br::at(ck_tile::number<0>{})) + "x" + _TS_(g0br::at(ck_tile::number<1>{})) + "x" + _TS_(g0br::at(ck_tile::n
"r" + _TS_(g1br::at(ck_tile::number<0>{})) + "x" + _TS_(g1br::at(ck_tile::number<1>{})) + "x" + _TS_(g1br::at(ck_tile::n
```
