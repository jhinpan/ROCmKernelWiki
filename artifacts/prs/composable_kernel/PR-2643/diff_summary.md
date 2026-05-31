# Diff summary

- **files changed:** 11
- **lines:** +1054 / -168
- **kernel-ish files:** 11

## Files (by churn)

- `include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_trload_qr_qtr_dor.hpp`  (+743/-0)
- `example/ck_tile/01_fmha/codegen/ops/fmha_bwd.py`  (+136/-75)
- `include/ck_tile/ops/fmha/kernel/fmha_bwd_kernel.hpp`  (+81/-40)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_trload_default_policy.hpp`  (+37/-28)
- `example/ck_tile/01_fmha/fmha_bwd.hpp`  (+17/-19)
- `example/ck_tile/01_fmha/fmha_bwd.cpp`  (+22/-0)
- `include/ck_tile/ops/fmha/pipeline/tile_fmha_shape.hpp`  (+7/-2)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_selector.hpp`  (+5/-1)
- `example/ck_tile/01_fmha/codegen/cpp_symbol_map.py`  (+3/-1)
- `include/ck_tile/ops/epilogue/default_2d_epilogue.hpp`  (+2/-2)
- `include/ck_tile/ops/fmha.hpp`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/cpp_symbol_map.py`**
```
"f" : "false",
True : "true",
False : "false",
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_bwd.py`**
```
from typing import List, Tuple, Dict, Literal, Any
using fmha_warp_tile2_{F_idx}   = ck_tile::sequence<{F_wm0}, {F_wn0}, ck_tile::min({F_wk0}, {F_bk4})>;
fmha_warp_tile2_{F_idx},
{F_maxq}>;
```

**`example/ck_tile/01_fmha/fmha_bwd.cpp`**
```
ck_tile::FillConstant<QGradDataType>{ck_tile::numeric<QGradDataType>::infinity()}(dq_host);
ck_tile::FillConstant<KGradDataType>{ck_tile::numeric<KGradDataType>::infinity()}(dk_host);
ck_tile::FillConstant<VGradDataType>{ck_tile::numeric<VGradDataType>::infinity()}(dv_host);
dq_buf.ToDevice(dq_host.data());
```

**`example/ck_tile/01_fmha/fmha_bwd.hpp`**
```
constexpr bool dq_uss_acc  = FmhaBwdDQDKDVKernel::kMaxSeqLenQ == 0;
const auto dq_ptr          = dq_uss_acc ? args.dq_acc_ptr : args.dq_ptr;
const auto stride_dq       = dq_uss_acc ? args.stride_dq_acc : args.stride_dq;
const auto nhead_stride_dq = dq_uss_acc ? args.nhead_stride_dq_acc : args.nhead_stride_dq;
```

**`include/ck_tile/ops/epilogue/default_2d_epilogue.hpp`**
```
operator()(ODramWindowTmp& o_dram_window_tmp, const OAccTile& o_acc_tile, void* = nullptr) const
void* = nullptr) const
```
