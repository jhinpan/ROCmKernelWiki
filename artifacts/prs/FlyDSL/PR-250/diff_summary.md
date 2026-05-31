# Diff summary

- **files changed:** 15
- **lines:** +1368 / -20
- **kernel-ish files:** 11

## Files (by churn)

- `kernels/rdna_fp8_preshuffle_gemm.py`  (+447/-0)
- `kernels/rdna_f16_gemm.py`  (+398/-0)
- `tests/kernels/test_rdna_gemm.py`  (+201/-0)
- `tests/kernels/benchmark_common.py`  (+160/-3)
- `scripts/run_benchmark.sh`  (+43/-4)
- `tests/kernels/conftest.py`  (+39/-0)
- `tests/arch_compat.py`  (+26/-0)
- `scripts/run_tests.sh`  (+22/-0)
- `kernels/layernorm_kernel.py`  (+6/-3)
- `kernels/rmsnorm_kernel.py`  (+6/-3)
- `kernels/softmax_kernel.py`  (+6/-3)
- `python/flydsl/runtime/device.py`  (+5/-1)
- `.github/workflows/flydsl.yaml`  (+3/-2)
- `.github/runner-config.yml`  (+4/-0)
- `python/flydsl/expr/rocdl/universal.py`  (+2/-1)

## Key added lines (kernel files)

**`kernels/layernorm_kernel.py`**
```
import math
from kernels.kernels_common import get_warp_size
WARP_SIZE = get_warp_size()
for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
```

**`kernels/rdna_f16_gemm.py`**
```
"""WMMA GEMM kernel for RDNA4 (gfx120x, wave32).
4-warp LDS kernel inspired by Triton's 93 TFLOPS approach.
Architecture:
- 128x128x32 tiles, 4 warps (128 threads), 2x2 warp layout
```

**`kernels/rdna_fp8_preshuffle_gemm.py`**
```
"""Fast Float8 Preshuffle GEMM for RDNA4 (gfx120x, wave32).
Optimized for M=32, N=8192, K=6144 (decode-phase inference shape).
C[M,N] = A[M,K] @ B[K,N]
Both A and B are fp8_e4m3fn with per-tensor scales.
```

**`kernels/rmsnorm_kernel.py`**
```
import math
from kernels.kernels_common import get_warp_size
WARP_SIZE = get_warp_size()
for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
```

**`kernels/softmax_kernel.py`**
```
import math
from kernels.kernels_common import get_warp_size
WARP_SIZE = get_warp_size()
for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
```
