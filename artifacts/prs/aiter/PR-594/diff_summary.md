# Diff summary

- **files changed:** 26
- **lines:** +1410 / -1007
- **kernel-ish files:** 25

## Files (by churn)

- `op_tests/op_benchmarks/triton/bench_gemm_a16w16.py`  (+118/-153)
- `op_tests/op_benchmarks/triton/bench_gemm_a8w8.py`  (+123/-145)
- `op_tests/op_benchmarks/triton/bench_gemm_a8w8_blockscale.py`  (+121/-145)
- `op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4.py`  (+110/-154)
- `op_tests/op_benchmarks/triton/bench_batched_gemm_afp4wfp4_pre_quant.py`  (+123/-136)
- `op_tests/op_benchmarks/triton/bench_batched_gemm_afp4wfp4.py`  (+121/-132)
- `op_tests/op_benchmarks/triton/bench_topk.py`  (+121/-19)
- `op_tests/op_benchmarks/triton/utils/benchmark_utils.py`  (+130/-1)
- `op_tests/op_benchmarks/triton/bench_routing.py`  (+112/-5)
- `op_tests/op_benchmarks/triton/bench_mha.py`  (+36/-72)
- `op_tests/op_benchmarks/triton/utils/argparse.py`  (+103/-0)
- `op_tests/op_benchmarks/triton/bench_tests/test_kernel_benchmarks.py`  (+80/-0)
- `op_tests/op_benchmarks/triton/bench_rope.py`  (+30/-3)
- `op_tests/op_benchmarks/triton/bench_pa_prefill.py`  (+7/-22)
- `op_tests/op_benchmarks/triton/bench_extend_attention.py`  (+16/-11)

## Key added lines (kernel files)

**`aiter/ops/triton/batched_gemm_afp4wfp4.py`**
```
stride_ab = tl.cast(stride_ab, tl.int64)
stride_bb = tl.cast(stride_bb, tl.int64)
stride_cb = tl.cast(stride_cb, tl.int64)
pid_batch = tl.cast(pid_batch, tl.int64)
```

**`aiter/ops/triton/batched_gemm_afp4wfp4_pre_quant.py`**
```
stride_ab = tl.cast(stride_ab, tl.int64)
stride_bb = tl.cast(stride_bb, tl.int64)
stride_cb = tl.cast(stride_cb, tl.int64)
pid_batch = tl.cast(pid_batch, tl.int64)
```

**`aiter/ops/triton/mha.py`**
```
"""Use 64-bit integer strides to prevent integer overflows with very large tensors."""
) -> Tuple[torch.Tensor, torch.Tensor]:
Convert a tensor to FP8 format, returning an FP8 tensor and a descale factor.
- x (torch.Tensor): shape [batch, seq_len, heads, dim]
```

**`aiter/ops/triton/routing.py`**
```
"small": (16, 64, 8, 1, 0, 1),
"medium": (16, 64, 8, 1, 0, 2),
"large": (16, 64, 8, 1, 2, 2),
"very_large": (16, 64, 8, 2, 2, 2),
```

**`op_tests/op_benchmarks/triton/bench_batched_gemm_afp4wfp4.py`**
```
import math
from op_tests.op_benchmarks.triton.utils.argparse import (
get_parser,
add_argparse_ff,
```
