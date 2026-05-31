# Diff summary

- **files changed:** 15
- **lines:** +1066 / -98
- **kernel-ish files:** 5

## Files (by churn)

- `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w8_blockscale.py`  (+246/-6)
- `aiter/ops/triton/gemm/basic/gemm_a16w8_blockscale.py`  (+146/-12)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-A16W8_BLOCKSCALE_PRESHUFFLED-N=2112-K=7168.json`  (+86/-0)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-A16W8_BLOCKSCALE_PRESHUFFLED-N=7168-K=2048.json`  (+86/-0)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=4096-K=512.json`  (+86/-0)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=4608-K=7168.json`  (+86/-0)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=7168-K=2048.json`  (+86/-0)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=7168-K=2304.json`  (+86/-0)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=2112-K=7168.json`  (+26/-26)
- `op_tests/triton_tests/gemm/basic/test_gemm_a16w8_blockscale.py`  (+43/-9)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=3072-K=1536.json`  (+25/-25)
- `op_tests/triton_tests/gemm/basic/test_gemm_a8w8_blockscale.py`  (+22/-9)
- `aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py`  (+14/-11)
- `aiter/ops/triton/configs/gemm/gfx942-GEMM-A16W8_BLOCKSCALE_PRESHUFFLED.json`  (+14/-0)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-A16W8_BLOCKSCALE_PRESHUFFLED.json`  (+14/-0)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w8_blockscale.py`**
```
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
_gemm_a16w8_blockscale_repr = make_kernel_repr(
"_gemm_a16w8_blockscale_kernel",
"BLOCK_SIZE_M",
```

**`aiter/ops/triton/gemm/basic/gemm_a16w8_blockscale.py`**
```
_gemm_a16w8_blockscale_preshuffle_kernel,
from aiter.ops.triton.utils.gemm_config_utils import compute_splitk_params
prequant: Optional[bool] = False,
skip_reduce: Optional[bool] = False,
```

**`aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py`**
```
return_y_pp = num_ksplit > 1 and skip_reduce
if return_y_pp:
return_y_pp = config["NUM_KSPLIT"] > 1 and skip_reduce
if y is None and not return_y_pp:
```

**`op_tests/triton_tests/gemm/basic/test_gemm_a16w8_blockscale.py`**
```
from aiter.ops.triton.gemm.basic.gemm_a16w8_blockscale import (
gemm_a16w8_blockscale,
gemm_a16w8_blockscale_preshuffle,
from aiter.ops.shuffle import shuffle_weight
```

**`op_tests/triton_tests/gemm/basic/test_gemm_a8w8_blockscale.py`**
```
(16, 4608, 7168),
(32, 4608, 7168),
(64, 4608, 7168),
(128, 4608, 7168),
```
