# Diff summary

- **files changed:** 96
- **lines:** +140 / -93
- **kernel-ish files:** 2

## Files (by churn)

- `Tensile/AsmCaps.py`  (+44/-0)
- `Tensile/Common.py`  (+2/-0)
- `Tensile/Tests/disabled/direct_to_lds/dtl_dgemm.yaml`  (+1/-1)
- `Tensile/Tests/disabled/direct_to_lds/dtl_dgemm_lite.yaml`  (+1/-1)
- `Tensile/Tests/disabled/direct_to_lds/dtl_tsgr_dgemm.yaml`  (+1/-1)
- `Tensile/Tests/emulation/float8/b8f8gemm_hybrid_b8f8b8s_SR_gfx940.yaml`  (+1/-1)
- `Tensile/Tests/emulation/float8/b8f8gemm_hybrid_b8f8b8s_gfx940.yaml`  (+1/-1)
- `Tensile/Tests/emulation/float8/b8f8gemm_hybrid_b8f8hs_gfx940.yaml`  (+1/-1)
- `Tensile/Tests/emulation/float8/b8f8gemm_hybrid_b8f8ss_gfx940.yaml`  (+1/-1)
- `Tensile/Tests/emulation/float8/b8gemm_b8b8s_SR_gfx940.yaml`  (+1/-1)
- `Tensile/Tests/emulation/float8/b8gemm_b8b8s_gfx940.yaml`  (+1/-1)
- `Tensile/Tests/emulation/float8/b8gemm_b8hs_gfx940.yaml`  (+1/-1)
- `Tensile/Tests/emulation/float8/b8gemm_b8ss_gfx940.yaml`  (+1/-1)
- `Tensile/Tests/emulation/float8/f8b8gemm_hybrid_f8b8b8s_SR_gfx940.yaml`  (+1/-1)
- `Tensile/Tests/emulation/float8/f8b8gemm_hybrid_f8b8b8s_gfx940.yaml`  (+1/-1)

## Key added lines (kernel files)

**`Tensile/AsmCaps.py`**
```
(11, 5, 1): {'HasAddLshl': True,
'HasAtomicAdd': True,
'HasDirectToLdsDest': False,
'HasDirectToLdsNoDest': False,
```

**`Tensile/Common.py`**
```
(11,5,1),
'gfx1151':'gfx1151',
```
