# Diff summary

- **files changed:** 85 (diff was byte-capped; summary is partial)
- **lines:** +4618 / -327
- **kernel-ish files:** 59

## Files (by churn)

- `csrc/kernels/mla/metadata/v1_1_device.cuh`  (+686/-0)
- `csrc/kernels/mla/reduce.cu`  (+633/-0)
- `op_tests/test_mla_persistent.py`  (+547/-0)
- `op_tests/test_mla_sparse.py`  (+513/-0)
- `csrc/kernels/mla/metadata/v1_2_device.cuh`  (+397/-0)
- `csrc/kernels/mla/metadata/v1_comm.cuh`  (+364/-0)
- `aiter/mla.py`  (+212/-101)
- `csrc/py_itfs_cu/asm_mla.cu`  (+235/-37)
- `csrc/kernels/mla/metadata/v1_1_host.cuh`  (+264/-0)
- `op_tests/test_mla.py`  (+162/-65)
- `aiter/ops/attention.py`  (+180/-1)
- `csrc/kernels/mla/metadata.cu`  (+133/-0)
- `csrc/include/rocm_ops.hpp`  (+67/-26)
- `csrc/include/mla.h`  (+69/-0)
- `csrc/include/attention_asm_mla.h`  (+33/-24)

## Key added lines (kernel files)

**`aiter/mla.py`**
```
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype
num_kv_splits_indptr,
MAYBE_FINAL_OUT: tl.constexpr,
BATCH_NUM: tl.constexpr,
```

**`aiter/ops/attention.py`**
```
import math
from typing import Tuple, Optional
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype
num_kv_splits_indptr: Optional[torch.Tensor],
```

**`aiter/utility/base_tuner.py`**
```
"""obtain name of the kernel from its id"""
```

**`csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_common.py`**
```
12: kernelInstance(256,     64,   128,    128,  16,  16,  16,   16,   4,    2,     [8, 32, 1],      [8, 32, 1],         
```

**`csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_common.py`**
```
90: kernelInstance( 256,      224,  128,   128,  16,  16,  16,   16,   14,   2,    [8, 32, 1],        [8, 32, 1],       
91: kernelInstance( 256,      256,   128,   128,  16,  16,  16,   16,  16,   2,    [8, 32, 1],        [8, 32, 1],       
92: kernelInstance( 256,      32,    192,   128,  16,  16,  16,   16,   2,   3,    [8, 32, 1],        [8, 32, 1],       
93: kernelInstance( 256,      64,    192,   128,  16,  16,  16,   16,   4,   3,    [8, 32, 1],        [8, 32, 1],       
```
