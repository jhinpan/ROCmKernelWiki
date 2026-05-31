# Diff summary

- **files changed:** 11
- **lines:** +622 / -24
- **kernel-ish files:** 7

## Files (by churn)

- `aiter/ops/triton/_triton_kernels/gemm_a8w8_blockscale.py`  (+221/-4)
- `aiter/ops/triton/gemm_a8w8_blockscale.py`  (+153/-3)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=2112-K=7168.json`  (+86/-0)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=3072-K=1536.json`  (+86/-0)
- `op_tests/triton_tests/gemm/basic/test_gemm_a8w8_blockscale.py`  (+43/-14)
- `aiter/ops/triton/configs/gemm/gfx942-GEMM-A8W8_BLOCKSCALE_PRESHUFFLED.json`  (+14/-0)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-A8W8_BLOCKSCALE_PRESHUFFLED.json`  (+14/-0)
- `aiter/ops/triton/gluon/gemm_a8w8_blockscale.py`  (+1/-1)
- `op_tests/op_benchmarks/triton/bench_gemm_a8w8_blockscale.py`  (+1/-1)
- `op_tests/triton_tests/gemm/fused/test_fused_gemm_a8w8_blockscale_a16w16.py`  (+1/-1)
- `op_tests/triton_tests/gemm/fused/test_fused_gemm_a8w8_blockscale_mul_add.py`  (+2/-0)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/gemm_a8w8_blockscale.py`**
```
a_scale_ptrs += offs_ks_step * stride_ascale_k
b_scale_ptrs += offs_ks_step * stride_bscale_k
c = accumulator.to(c_ptr.type.element_ty)
offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
```

**`aiter/ops/triton/gemm_a8w8_blockscale.py`**
```
_gemm_a8w8_blockscale_preshuffle_kernel,
def gemm_a8w8_blockscale_preshuffle(
x: torch.Tensor,
w: torch.Tensor,
```

**`aiter/ops/triton/gluon/gemm_a8w8_blockscale.py`**
```
if int(dev.split("gfx")[1]) < 950:
```

**`op_tests/op_benchmarks/triton/bench_gemm_a8w8_blockscale.py`**
```
x, weight, _, x_scale, _, w_scale, y = generate_gemm_a8w8_blockscale_inputs(
```

**`op_tests/triton_tests/gemm/basic/test_gemm_a8w8_blockscale.py`**
```
gemm_a8w8_blockscale_preshuffle as triton_gemm_a8w8_blockscale_preshuffle,
from aiter.ops.shuffle import shuffle_weight
output: bool = False,
shuffle: bool = False,
```
