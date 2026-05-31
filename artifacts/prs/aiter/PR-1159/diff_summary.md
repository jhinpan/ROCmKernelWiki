# Diff summary

- **files changed:** 60 (diff was byte-capped; summary is partial)
- **lines:** +1187 / -1208
- **kernel-ish files:** 59

## Files (by churn)

- `csrc/ck_gemm_a4w4_blockscale/include/gemm_a4w4_blockscale_common.cuh`  (+168/-168)
- `csrc/ck_gemm_a8w8_blockscale/include/gemm_a8w8_blockscale_common.cuh`  (+166/-166)
- `csrc/kernels/custom_all_reduce.cu`  (+135/-136)
- `csrc/kernels/moe_align_block_size_kernels.cu`  (+78/-82)
- `aiter/jit/optCompilerConfig.json`  (+47/-87)
- `csrc/py_itfs_ck/moe_ck_2stages_kernel.cu`  (+47/-47)
- `csrc/include/custom_all_reduce.cuh`  (+36/-53)
- `aiter/jit/core.py`  (+46/-28)
- `csrc/include/mha_common.h`  (+44/-27)
- `csrc/kernels/attention_ragged.cu`  (+49/-19)
- `csrc/kernels/rmsnorm_kernels.cu`  (+24/-34)
- `csrc/ck_gemm_a8w8_blockscale_bpreshuffle/include/gemm_a8w8_blockscale_bpreshuffle_common.cuh`  (+21/-21)
- `csrc/kernels/cache_kernels.cu`  (+21/-21)
- `csrc/ck_gemm_a8w8_bpreshuffle/include/gemm_a8w8_bpreshuffle_common.cuh`  (+19/-19)
- `csrc/kernels/custom_kernels.cu`  (+18/-18)

## Key added lines (kernel files)

**`aiter/jit/core.py`**
```
def rename_cpp_to_cu(els, dst, hipify, recursive=False):
if hipify:
if name.endswith(".cpp") or name.endswith(".cu"):
newName = name.replace(".cpp", ".cu")
```

**`aiter/jit/utils/cpp_extension.py`**
```
return True
```

**`aiter/ops/communication.py`**
```
import logging
from torch import Tensor
import aiter
destroy_distributed_environment,
```

**`aiter/ops/custom_all_reduce.py`**
```
import torch
from ..jit.core import compile_ops
```

**`aiter/utility/base_tuner.py`**
```
"""??kernel name"""
```
