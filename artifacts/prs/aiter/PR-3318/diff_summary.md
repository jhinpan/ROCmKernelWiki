# Diff summary

- **files changed:** 7
- **lines:** +220 / -696
- **kernel-ish files:** 6

## Files (by churn)

- `aiter/ops/triton/_gluon_kernels/gemm/basic/gemm_mxfp4.py`  (+0/-388)
- `op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4.py`  (+71/-137)
- `aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py`  (+46/-95)
- `aiter/ops/triton/configs/gemm/gfx1250-GEMM-AFP4WFP4_PRESHUFFLED.json`  (+84/-14)
- `aiter/ops/shuffle.py`  (+0/-38)
- `op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4.py`  (+15/-22)
- `aiter/ops/triton/gluon/gemm_afp4wfp4.py`  (+4/-2)

## Key added lines (kernel files)

**`aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py`**
```
_gemm_afp4wfp4_kernel,
_gemm_afp4wfp4_preshuffle_kernel,
_gemm_afp4wfp4_kernel_preshuffle_scales,
_gemm_afp4wfp4_kernel[grid](
```

**`aiter/ops/triton/gluon/gemm_afp4wfp4.py`**
```
if dev != "gfx950":
raise ValueError(
"Gluon implementation is not supported on this device (requires CDNA4)."
```

**`op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4.py`**
```
import sys
gemm_afp4wfp4_preshuffled_weight_scales,
M: int, N: int, K: int, metric: str, layout: str, shuffle: bool, gluon: bool
shuffle_scales_fg=shuffle,
```

**`op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4.py`**
```
from aiter.ops.triton.gluon.gemm_afp4wfp4 import gemm_afp4wfp4 as gluon_gemm_afp4wfp4
from aiter.ops.shuffle import shuffle_weight
if M >= 32:
x_scales_shuffled = shuffle_scales(x_scales)
```
