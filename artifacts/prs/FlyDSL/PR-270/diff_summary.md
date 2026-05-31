# Diff summary

- **files changed:** 17
- **lines:** +382 / -542
- **kernel-ish files:** 12

## Files (by churn)

- `kernels/moe_blockscale_2stage.py`  (+131/-160)
- `kernels/moe_gemm_2stage.py`  (+119/-148)
- `kernels/layout_utils.py`  (+0/-131)
- `docs/kernel_authoring_guide.md`  (+24/-5)
- `kernels/preshuffle_gemm.py`  (+10/-17)
- `python/flydsl/expr/buffer_ops.py`  (+13/-13)
- `docs/prebuilt_kernels_guide.md`  (+16/-8)
- `kernels/blockscale_preshuffle_gemm.py`  (+10/-13)
- `CLAUDE.md`  (+16/-6)
- `python/flydsl/expr/primitive.py`  (+15/-3)
- `docs/layout_system_guide.md`  (+1/-15)
- `kernels/mfma_preshuffle_pipeline.py`  (+11/-4)
- `python/flydsl/expr/rocdl/universal.py`  (+5/-8)
- `python/flydsl/expr/rocdl.py`  (+4/-6)
- `.claude/skills/flydsl-kernel-authoring/SKILL.md`  (+5/-2)

## Key added lines (kernel files)

**`kernels/blockscale_preshuffle_gemm.py`**
```
coord_wave_lane = fx.idx2crd(tx, layout_wave_lane)
wave_id = fx.get(coord_wave_lane, 0)
lane_id = fx.get(coord_wave_lane, 1)
coord_lane16 = fx.idx2crd(lane_id, layout_lane16)
```

**`kernels/mfma_preshuffle_pipeline.py`**
```
def crd2idx(crd, layout):
"""crd2idx returning an index-type scalar (unwraps fly.int_tuple)."""
result = fx.crd2idx(crd, layout)
scalar = fx.get_scalar(result)
```

**`kernels/moe_blockscale_2stage.py`**
```
emu = rocdl.exp2(T.f32, t)
sig = rocdl.rcp(T.f32, den)
arith.constant_vector(0, T.i32x4)
else arith.constant_vector(0.0, T.f32x4)
```

**`kernels/moe_gemm_2stage.py`**
```
from kernels.mfma_preshuffle_pipeline import crd2idx
emu = rocdl.exp2(T.f32, t)
sig = rocdl.rcp(T.f32, den)
arith.constant_vector(0, T.i32x4)
```

**`kernels/preshuffle_gemm.py`**
```
coord_wave_lane = fx.idx2crd(tx, layout_wave_lane)
wave_id = fx.get(coord_wave_lane, 0)
lane_id = fx.get(coord_wave_lane, 1)
coord_lane16 = fx.idx2crd(lane_id, layout_lane16)
```
