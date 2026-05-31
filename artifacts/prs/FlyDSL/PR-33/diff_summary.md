# Diff summary

- **files changed:** 12
- **lines:** +1324 / -854
- **kernel-ish files:** 11

## Files (by churn)

- `kernels/layernorm_kernel.py`  (+215/-252)
- `kernels/reduce.py`  (+195/-85)
- `tests/kernels/benchmark_common.py`  (+265/-0)
- `kernels/softmax_kernel.py`  (+101/-153)
- `tests/kernels/test_layernorm.py`  (+153/-87)
- `tests/kernels/test_rmsnorm.py`  (+139/-84)
- `kernels/rmsnorm_kernel.py`  (+117/-99)
- `tests/kernels/test_softmax.py`  (+137/-39)
- `tests/kernels/gpu_common.py`  (+0/-53)
- `flydsl/src/flydsl/dialects/ext/func.py`  (+1/-1)
- `flydsl/src/flydsl/dialects/ext/mlir_extras/util.py`  (+1/-1)
- `flir/build.sh`  (+0/-0)

## Key added lines (kernel files)

**`flydsl/src/flydsl/dialects/ext/func.py`**
```
if any(tok in src for tok in ("flir.", "arith.", "scf.", "gpu.", "memref.", "vector.")):
```

**`flydsl/src/flydsl/dialects/ext/mlir_extras/util.py`**
```
if any(tok in src for tok in ("flir.", "arith.", "scf.", "gpu.", "memref.", "vector.")):
```

**`kernels/layernorm_kernel.py`**
```
from flydsl.dialects.ext import flir, arith
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
arch = get_hip_arch()
USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or arch.startswith("gfx95")
```

**`kernels/reduce.py`**
```
from __future__ import annotations
from flydsl.dialects.ext.python_control_flow import lower_range_for_loops
return vector.reduction(compute_type, "maxnumf", vec_val)
return vector.reduction(compute_type, "add", vec_val, fastmath=fm_fast)
```

**`kernels/rmsnorm_kernel.py`**
```
from flydsl.dialects.ext import flir, arith
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
arch = get_hip_arch()
row = flir.const_index(flir.block_idx("x"))
```
