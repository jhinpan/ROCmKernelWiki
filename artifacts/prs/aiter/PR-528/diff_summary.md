# Diff summary

- **files changed:** 18
- **lines:** +1921 / -4
- **kernel-ish files:** 14

## Files (by churn)

- `csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_common.py`  (+339/-0)
- `csrc/ck_gemm_a4w4_blockscale/gen_instances.py`  (+293/-0)
- `csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py`  (+228/-0)
- `aiter/configs/a4w4_blockscale_untuned_gemm.csv`  (+184/-0)
- `aiter/configs/a4w4_blockscale_tuned_gemm.csv`  (+182/-0)
- `csrc/ck_gemm_a4w4_blockscale/include/gemm_a4w4_blockscale_common.cuh`  (+167/-0)
- `op_tests/test_gemm_a4w4_blockscale.py`  (+154/-0)
- `csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale.cu`  (+117/-0)
- `csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.cu`  (+90/-0)
- `aiter/jit/optCompilerConfig.json`  (+41/-4)
- `csrc/ck_gemm_a4w4_blockscale/README.md`  (+28/-0)
- `aiter/ops/gemm_op_a4w4.py`  (+22/-0)
- `csrc/include/rocm_ops.hpp`  (+22/-0)
- `csrc/ck_gemm_a4w4_blockscale/include/gemm_a4w4_blockscale.h`  (+20/-0)
- `aiter/jit/core.py`  (+14/-0)

## Key added lines (kernel files)

**`aiter/jit/core.py`**
```
"hip_clang_path": None,
hip_clang_path = d_args.get("hip_clang_path", None)
prev_hip_clang_path = None
if hip_clang_path is not None and os.path.exists(hip_clang_path):
```

**`aiter/ops/gemm_op_a4w4.py`**
```
@compile_ops("module_gemm_a4w4_blockscale")
def gemm_a4w4_blockscale(
XQ: Tensor,  # XQ:[M, K/2] f4x2
WQ: Tensor,  # WQ:[N, K/2] f4x2
```

**`csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale.cu`**
```
using BlockwiseKernel = std::function<
torch::Tensor(torch::Tensor &, torch::Tensor &,
torch::Tensor &, torch::Tensor &,
torch::Tensor &)>;
```

**`csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_common.py`**
```
from dataclasses import dataclass
@dataclass
class kernelInstance:
BLOCK_SIZE: int
```

**`csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.cu`**
```
using BlockwiseKernel = std::function<
torch::Tensor(torch::Tensor &, torch::Tensor &,
torch::Tensor &, torch::Tensor &,
torch::Tensor &)>;
```
