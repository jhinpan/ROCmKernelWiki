# Diff summary

- **files changed:** 18
- **lines:** +204 / -116
- **kernel-ish files:** 13

## Files (by churn)

- `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale.cu`  (+11/-31)
- `aiter/ops/gemm_op_a4w4.py`  (+23/-16)
- `csrc/py_itfs_cu/gemm_common.cu`  (+37/-0)
- `csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle.cu`  (+8/-27)
- `csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale.cu`  (+11/-21)
- `csrc/py_itfs_cu/asm_gemm_a4w4.cu`  (+17/-8)
- `op_tests/test_gemm_a4w4.py`  (+23/-2)
- `aiter/ops/gemm_op_a8w8.py`  (+13/-8)
- `aiter/jit/optCompilerConfig.json`  (+15/-0)
- `aiter/ops/gemm_op_common.py`  (+10/-0)
- `csrc/pybind/gemm_common_pybind.cu`  (+9/-0)
- `aiter/configs/a4w4_blockscale_tuned_gemm.csv`  (+8/-0)
- `csrc/include/rocm_ops.hpp`  (+7/-0)
- `csrc/include/gemm_common.h`  (+6/-0)
- `csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py`  (+3/-2)

## Key added lines (kernel files)

**`aiter/ops/gemm_op_a4w4.py`**
```
from ..ops.gemm_op_common import get_padded_m
def get_GEMM_config(M: int, N: int, K: int):
if not hasattr(get_GEMM_config, "gemm_dict"):
gemm_dict = pd.read_csv(
```

**`aiter/ops/gemm_op_a8w8.py`**
```
from ..ops.gemm_op_common import get_padded_m
padded_M = M
config = None
for gl in [None, 0, 1]:
```

**`aiter/ops/gemm_op_common.py`**
```
from ..jit.core import (
compile_ops,
@compile_ops("module_gemm_common")
def get_padded_m(M: int, N: int, K: int, gl: int) -> int: ...
```

**`csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale.cu`**
```
padded_m = getPaddedM(M, N, K, 0);
padded_m = getPaddedM(M, N, K, 1);
it = lookup.find({padded_m, N, K});
if (it != lookup.end())
```

**`csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py`**
```
if splitK is not None and splitK > 0:
asm_tiles = [key for key in asm_kernels.keys()]
```
