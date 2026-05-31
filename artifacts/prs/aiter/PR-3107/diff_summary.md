# Diff summary

- **files changed:** 9
- **lines:** +615 / -1195
- **kernel-ish files:** 7

## Files (by churn)

- `aiter/ops/flydsl/kernels/preshuffle_gemm.py`  (+235/-735)
- `aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv`  (+191/-191)
- `aiter/configs/model_configs/dsv3_a8w8_bpreshuffle_tuned_gemm.csv`  (+131/-131)
- `aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py`  (+31/-83)
- `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py`  (+17/-23)
- `aiter/ops/gemm_op_a8w8.py`  (+7/-8)
- `aiter/aot/flydsl/gemm.py`  (+3/-8)
- `csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py`  (+0/-9)
- `aiter/ops/flydsl/gemm_kernels.py`  (+0/-7)

## Key added lines (kernel files)

**`aiter/aot/flydsl/gemm.py`**
```
r"(?P<lds_stage>\d+)x(?P<cshuffle>\d+)x(?P<async_copy>\d+)x(?P<waves_per_eu>\d+)_"
r"(?P<scheduler>[A-Za-z0-9_]+)$"
_compile_executable_to_cache(exe, out, a, b, scale_a, scale_b, m, n, stream)
```

**`aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py`**
```
for lds in _LDS_STAGES:
for tm, tn, tk in tiles_by_lds[lds]:
if wpe > 0 and wpe > _estimate_max_wpe(tm, tn, total_vgpr):
kl[idx] = _ki(tm, tn, tk, lds, csh, acp, wpe)
```

**`aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py`**
```
layout_scale: object  # fly layout value (same as PreshuffleBLayout.layout_b)
stride_n0: object  # index-typed MLIR value (dynamic)
stride_k0: object  # index-typed MLIR value (= 64)
stride_klane: object  # index-typed MLIR value (= 16)
```

**`aiter/ops/flydsl/kernels/preshuffle_gemm.py`**
```
from flydsl.expr import range_constexpr, const_expr
from flydsl._mlir import ir
from flydsl.expr import arith, vector
from flydsl.expr import gpu
```

**`aiter/ops/gemm_op_a8w8.py`**
```
"""Parse tile config from flydsl kernelName, e.g.
'flydsl_bpreshuflle_128x64x256_F8_F8_B16_2x0x1x1_default'
-> (tile_m=128, tile_n=64, tile_k=256, lds_stage=2, cshuffle=0, async_copy=1, wpe=1)
Returns None on parse failure.
```
