# Diff summary

- **files changed:** 7
- **lines:** +174 / -100
- **kernel-ish files:** 7

## Files (by churn)

- `aiter/ops/triton/gemm_afp4wfp4.py`  (+78/-43)
- `aiter/ops/triton/gemm_a16wfp4.py`  (+29/-13)
- `aiter/ops/triton/gemm_a16w16_atomic.py`  (+16/-18)
- `aiter/ops/triton/batched_gemm_a16wfp4.py`  (+24/-9)
- `op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4.py`  (+20/-4)
- `aiter/ops/triton/batched_gemm_afp4wfp4_pre_quant.py`  (+5/-8)
- `aiter/ops/triton/gemm_afp4wfp4_pre_quant_atomic.py`  (+2/-5)

## Key added lines (kernel files)

**`aiter/ops/triton/batched_gemm_a16wfp4.py`**
```
from aiter.ops.triton.utils.common_utils import serialize_dict, deserialize_str
def batched_gemm_a16wfp4_(
Computes batched matrix multiplication Y[i] = X[i] @ W[i]^T with BF16 activations and FP4 weights.
x (torch.Tensor): BF16/FP16 input matrix with shape (B, M, K).
```

**`aiter/ops/triton/batched_gemm_afp4wfp4_pre_quant.py`**
```
x: torch.Tensor,
w: torch.Tensor,
w_scales: torch.Tensor,
) -> torch.Tensor:
```

**`aiter/ops/triton/gemm_a16w16_atomic.py`**
```
Computes 16 bit matrix multiplication Y = X @ W^T using atomic operations for split-K reduction.
x (torch.Tensor): BF16/FP16 input matrix  matrix with shape (M, K).
w (torch.Tensor): BF16/FP16 weight matrix with shape (N, K), internally transposed.
dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
```

**`aiter/ops/triton/gemm_a16wfp4.py`**
```
from aiter.ops.triton.utils.common_utils import serialize_dict, deserialize_str
atomic_add: Optional[bool] = False,
def gemm_a16wfp4_(
atomic_add: Optional[bool] = False,
```

**`aiter/ops/triton/gemm_afp4wfp4.py`**
```
NUM_KSPLIT = triton.cdiv(K, (SPLITK_BLOCK_SIZE // 2))
M, K = x.shape
N, _ = w.shape
config = deserialize_str(config)
```
