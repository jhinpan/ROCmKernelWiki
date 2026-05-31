# Diff summary

- **files changed:** 8
- **lines:** +456 / -20
- **kernel-ish files:** 7

## Files (by churn)

- `kernels/rdna3_f16_gemm.py`  (+358/-0)
- `tests/kernels/test_rdna_gemm.py`  (+39/-11)
- `tests/kernels/benchmark_common.py`  (+24/-4)
- `scripts/run_benchmark.sh`  (+8/-5)
- `kernels/rdna_fp8_preshuffle_gemm.py`  (+12/-0)
- `python/flydsl/expr/typing.py`  (+12/-0)
- `kernels/rdna_f16_gemm.py`  (+2/-0)
- `python/flydsl/utils/smem_allocator.py`  (+1/-0)

## Key added lines (kernel files)

**`kernels/rdna3_f16_gemm.py`**
```
"""WMMA GEMM kernel for RDNA3 / RDNA3.5 (gfx11*, wave32).
Ported from rdna_f16_gemm.py (gfx120x). Same algorithm (4-warp double-
buffered LDS ping-pong, 128x128x32 tiles, swizzled grid mapping) but
adapted for the legacy v16-operand WMMA ABI used by RDNA3/RDNA3.5:
```

**`kernels/rdna_f16_gemm.py`**
```
elif const_expr(out_dtype == "f16"):
val = val.to(fx.Float16)
```

**`kernels/rdna_fp8_preshuffle_gemm.py`**
```
from flydsl.runtime.device import get_rocm_arch
_arch = str(get_rocm_arch() or "")
if _arch.startswith("gfx11"):
raise RuntimeError(
```

**`python/flydsl/expr/typing.py`**
```
- gfx12*: FP8 E4M3FN (OCP)
Raises ``RuntimeError`` on gfx11* (RDNA3/RDNA3.5): these chips have no
native FP8 instructions, so FP8 compute would surface as a late LLVM
"cannot select" error. Fail early with a clear message instead.
```

**`python/flydsl/utils/smem_allocator.py`**
```
"gfx1151": 65536,  # RDNA3.5: 64KB LDS per WGP
```
