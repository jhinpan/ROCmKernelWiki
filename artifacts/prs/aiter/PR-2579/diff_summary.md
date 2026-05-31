# Diff summary

- **files changed:** 10
- **lines:** +2779 / -249
- **kernel-ish files:** 10

## Files (by churn)

- `aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention.py`  (+559/-149)
- `op_tests/op_benchmarks/triton/bench_fav3_sage.py`  (+637/-27)
- `op_tests/op_benchmarks/triton/bench_fav3_sage_mxfp4.py`  (+441/-27)
- `op_tests/triton_tests/attention/test_fav3_sage.py`  (+386/-3)
- `aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention_mxfp4.py`  (+344/-23)
- `aiter/test_mha_common.py`  (+122/-0)
- `aiter/ops/triton/_triton_kernels/attention/block_lut.py`  (+104/-0)
- `aiter/ops/triton/attention/fav3_sage.py`  (+57/-18)
- `aiter/ops/triton/attention/utils.py`  (+72/-0)
- `aiter/ops/triton/attention/fav3_sage_attention_mxfp4_wrapper.py`  (+57/-2)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/attention/block_lut.py`**
```
Triton kernel to build the block-sparse LUT (kv_block_indices) from a 4D block
attention mask without using nonzero or argsort.
import torch
import triton
```

**`aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention.py`**
```
@triton.jit
def _sage_fwd_blocksparse_nomask(
k_base_ptrs,
v_base_ptrs,
```

**`aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention_mxfp4.py`**
```
@triton.jit
def _sage_fwd_blocksparse_nomask_mxfp4(
k_base_ptrs,
v_base_ptrs,
```

**`aiter/ops/triton/attention/fav3_sage.py`**
```
"num_stages": 5,
block_lut: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
num_q_blocks = (seqlen_q + BLKQ - 1) // BLKQ
num_k_blocks = (seqlen_k + BLKK - 1) // BLKK
```

**`aiter/ops/triton/attention/fav3_sage_attention_mxfp4_wrapper.py`**
```
from typing import Optional, Tuple
block_lut: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
if block_lut is not None:
kv_block_indices, lut_start, lut_count = block_lut
```
