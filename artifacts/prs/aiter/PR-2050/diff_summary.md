# Diff summary

- **files changed:** 16
- **lines:** +1474 / -168
- **kernel-ish files:** 14

## Files (by churn)

- `op_tests/op_benchmarks/triton/model_benchmarking_tool/model_shapes.json`  (+578/-0)
- `op_tests/op_benchmarks/triton/model_benchmarking_tool/bench_models.py`  (+541/-0)
- `op_tests/op_benchmarks/triton/model_benchmarking_tool/model_shapes.md`  (+117/-0)
- `op_tests/op_benchmarks/triton/bench_moe_gemm_a8w4.py`  (+37/-21)
- `op_tests/op_benchmarks/triton/bench_moe_gemm_a8w8.py`  (+37/-21)
- `op_tests/op_benchmarks/triton/bench_moe_gemm_a8w8_blockscale.py`  (+37/-21)
- `op_tests/op_benchmarks/triton/bench_moe_gemm_a4w4.py`  (+35/-19)
- `op_tests/op_benchmarks/triton/bench_rmsnorm.py`  (+21/-12)
- `op_tests/op_benchmarks/triton/bench_batched_gemm_a8w8.py`  (+12/-13)
- `op_tests/op_benchmarks/triton/bench_batched_gemm_afp4wfp4.py`  (+12/-12)
- `op_tests/op_benchmarks/triton/bench_gemm_a16w16.py`  (+9/-10)
- `op_tests/op_benchmarks/triton/bench_gemm_a8w8_blockscale.py`  (+9/-10)
- `op_tests/op_benchmarks/triton/bench_gemm_a8w8_per_token_scale.py`  (+9/-10)
- `op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4.py`  (+9/-9)
- `op_tests/op_benchmarks/triton/bench_rope.py`  (+7/-8)

## Key added lines (kernel files)

**`op_tests/op_benchmarks/triton/bench_batched_gemm_a8w8.py`**
```
from aiter.ops.triton.gemm.batched.batched_gemm_a8w8 import (
batched_gemm_a8w8 as batched_gemm_a8w8,
def parse_args(args: list[str] | None = None):
return get_ff_args(parser, args=args)
```

**`op_tests/op_benchmarks/triton/bench_batched_gemm_afp4wfp4.py`**
```
from aiter.ops.triton.gemm.batched.batched_gemm_afp4wfp4 import (
batched_gemm_afp4wfp4 as batched_gemm_afp4wfp4,
def parse_args(args: list[str] | None = None):
return get_ff_args(parser, args=args)
```

**`op_tests/op_benchmarks/triton/bench_gemm_a16w16.py`**
```
def parse_args(args: list[str] | None = None):
return get_ff_args(parser, args=args)
def main(args: list[str] | None = None) -> None:
parsed_args, defaults = parse_args(args=args)
```

**`op_tests/op_benchmarks/triton/bench_gemm_a8w8_blockscale.py`**
```
def parse_args(args: list[str] | None = None):
return get_ff_args(parser, args=args)
def main(args: list[str] | None = None) -> None:
parsed_args, defaults = parse_args(args=args)
```

**`op_tests/op_benchmarks/triton/bench_gemm_a8w8_per_token_scale.py`**
```
def parse_args(args: list[str] | None = None):
return get_ff_args(parser, args=args)
def main(args: list[str] | None = None) -> None:
parsed_args, defaults = parse_args(args=args)
```
