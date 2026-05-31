# Diff summary

- **files changed:** 32
- **lines:** +212 / -238
- **kernel-ish files:** 6

## Files (by churn)

- `csrc/py_itfs_cu/asm_gemm_a16w16.cu`  (+111/-182)
- `csrc/include/rocm_ops.hpp`  (+29/-28)
- `hsa/gfx950/bf16gemm/bf16gemm_fp32bf16.csv`  (+25/-13)
- `hsa/gfx942/bf16gemm/bf16gemm_fp32bf16.csv`  (+23/-11)
- `aiter/tuned_gemm.py`  (+12/-2)
- `aiter/ops/gemm_op_a16w16.py`  (+9/-1)
- `op_tests/test_gemm_a16w16.py`  (+2/-1)
- `csrc/include/asm_gemm_a16w16.h`  (+1/-0)
- `hsa/gfx942/bf16gemm/bf16gemm_fp32bf16_tn_128x64_bshuffle_splitk_clean.co`  (+0/-0)
- `hsa/gfx942/bf16gemm/bf16gemm_fp32bf16_tn_160x64_bshuffle_splitk_clean.co`  (+0/-0)
- `hsa/gfx942/bf16gemm/bf16gemm_fp32bf16_tn_32x64_bshuffle_splitk_clean.co`  (+0/-0)
- `hsa/gfx942/bf16gemm/bf16gemm_fp32bf16_tn_32x64_splitk_clean.co`  (+0/-0)
- `hsa/gfx942/bf16gemm/bf16gemm_fp32bf16_tn_48x64_bshuffle_splitk_clean.co`  (+0/-0)
- `hsa/gfx942/bf16gemm/bf16gemm_fp32bf16_tn_48x64_splitk_clean.co`  (+0/-0)
- `hsa/gfx942/bf16gemm/bf16gemm_fp32bf16_tn_64x64_bshuffle_splitk_clean.co`  (+0/-0)

## Key added lines (kernel files)

**`aiter/ops/gemm_op_a16w16.py`**
```
semaphore: Tensor,
semaphore: Tensor,
@functools.lru_cache(maxsize=1)
def get_semaphore_workspace(device: torch.device) -> Tensor:
```

**`aiter/tuned_gemm.py`**
```
from aiter import (
gemm_a16w16_asm,
get_semaphore_workspace,
hipb_create_extension,
```

**`csrc/include/asm_gemm_a16w16.h`**
```
torch::Tensor& semaphore,
```

**`csrc/include/rocm_ops.hpp`**
```
py::arg("semaphore"),                  \
m.def("top_k_per_row_prefill",     \
&top_k_per_row_prefill,      \
py::arg("logits"),           \
```

**`csrc/py_itfs_cu/asm_gemm_a16w16.cu`**
```
void* ptr_D;
void* ptr_C;
void* ptr_A;
void* ptr_B;
```
