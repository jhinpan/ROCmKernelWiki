# Diff summary

- **files changed:** 18
- **lines:** +964 / -162
- **kernel-ish files:** 17

## Files (by churn)

- `op_tests/op_benchmarks/triton/bench_batched_gemm_a8w8.py`  (+155/-0)
- `op_tests/op_benchmarks/triton/bench_batched_gemm_a16w16.py`  (+149/-0)
- `op_tests/op_benchmarks/triton/bench_gemm_a8wfp4.py`  (+148/-0)
- `op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4_pre_quant_atomic.py`  (+136/-0)
- `op_tests/triton_tests/test_gemm_a8wfp4.py`  (+86/-35)
- `op_tests/op_benchmarks/triton/utils/benchmark_utils.py`  (+72/-24)
- `op_tests/op_benchmarks/triton/bench_batched_gemm_afp4wfp4_pre_quant.py`  (+32/-25)
- `op_tests/triton_tests/test_batched_gemm_a8w8.py`  (+45/-11)
- `op_tests/triton_tests/test_gemm_afp4wfp4_pre_quant_atomic.py`  (+31/-13)
- `op_tests/op_benchmarks/triton/bench_gemm_a16w16.py`  (+32/-10)
- `op_tests/triton_tests/test_batched_gemm_bf16.py`  (+32/-9)
- `op_tests/op_benchmarks/triton/bench_batched_gemm_afp4wfp4.py`  (+17/-21)
- `op_tests/triton_tests/test_batched_gemm_afp4wfp4_pre_quant.py`  (+8/-4)
- `op_tests/op_benchmarks/triton/bench_gemm_a8w8_blockscale.py`  (+8/-2)
- `op_tests/op_benchmarks/triton/utils/argparse.py`  (+7/-2)

## Key added lines (kernel files)

**`op_tests/op_benchmarks/triton/bench_batched_gemm_a16w16.py`**
```
import sys
import torch
import triton
import math
```

**`op_tests/op_benchmarks/triton/bench_batched_gemm_a8w8.py`**
```
import sys
import torch
import triton
import math
```

**`op_tests/op_benchmarks/triton/bench_batched_gemm_afp4wfp4.py`**
```
batched_model_benchmark_shapes,
print_vgpr,
model_benchmark_shapes_fn=batched_model_benchmark_shapes,
x_names=["batch", "M", "N", "K"],
```

**`op_tests/op_benchmarks/triton/bench_batched_gemm_afp4wfp4_pre_quant.py`**
```
batched_model_benchmark_shapes,
print_vgpr,
batch: int,
metric: str,
```

**`op_tests/op_benchmarks/triton/bench_gemm_a16w16.py`**
```
from aiter.ops.triton.gemm_a16w16_atomic import gemm_a16w16_atomic
print_vgpr,
def bench_gemm_fn(
M: int, N: int, K: int, metric: str, layout: str, atomic: bool = False, **kwargs
```
