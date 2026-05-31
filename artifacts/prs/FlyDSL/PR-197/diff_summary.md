# Diff summary

- **files changed:** 9 (diff was byte-capped; summary is partial)
- **lines:** +302 / -3786
- **kernel-ish files:** 9

## Files (by churn)

- `kernels/mixed_moe_gemm_2stage.py`  (+0/-2361)
- `kernels/mixed_preshuffle_gemm.py`  (+0/-1123)
- `kernels/moe_blockscale_2stage.py`  (+125/-130)
- `kernels/moe_gemm_2stage.py`  (+110/-110)
- `kernels/mfma_preshuffle_pipeline.py`  (+24/-24)
- `kernels/blockscale_preshuffle_gemm.py`  (+21/-21)
- `kernels/layernorm_kernel.py`  (+13/-13)
- `kernels/mfma_epilogues.py`  (+4/-3)
- `kernels/layout_utils.py`  (+5/-1)

## Key added lines (kernel files)

**`kernels/blockscale_preshuffle_gemm.py`**
```
c_m = arith.index_cast(T.index, i32_m)
c_n = arith.index_cast(T.index, i32_n)
rt_M = arith.index_cast(T.index, i32_m)
rt_N = arith.index_cast(T.index, i32_n)
```

**`kernels/layernorm_kernel.py`**
```
width_i32 = fx.Int32(WARP_SIZE)
off = fx.Int32(sh)
if lane == fx.Int32(0):
if wave == fx.Int32(0):
```

**`kernels/layout_utils.py`**
```
if not isinstance(cv, ir.Value) and hasattr(cv, 'ir_value'):
cv = cv.ir_value()
elif isinstance(cv, ArithValue):
```

**`kernels/mfma_epilogues.py`**
```
import flydsl.expr as fx
ii_idx_list = [fx.Index(ii) for ii in range(4)]
c_nlane = fx.Index(CShuffleNLane)
c_evec = fx.Index(EVec)
```

**`kernels/mfma_preshuffle_pipeline.py`**
```
c16 = fx.Index(16)
c64 = fx.Index(64)
c4 = fx.Index(4)
c_kpack = fx.Index(kpack_bytes)
```
