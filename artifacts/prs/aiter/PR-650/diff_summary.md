# Diff summary

- **files changed:** 31
- **lines:** +421 / -224
- **kernel-ish files:** 29

## Files (by churn)

- `aiter/ops/triton/configs/moe/MI350X-MOE_ROUTING_SIGMOID_TOPK1.json`  (+70/-0)
- `aiter/ops/triton/moe_routing_sigmoid_top1_fused.py`  (+0/-61)
- `op_tests/op_benchmarks/triton/bench_batched_gemm_afp4wfp4_pre_quant.py`  (+28/-21)
- `op_tests/op_benchmarks/triton/bench_batched_gemm_afp4wfp4.py`  (+29/-19)
- `op_tests/triton_tests/test_batched_gemm_afp4wfp4.py`  (+39/-9)
- `op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4.py`  (+26/-16)
- `op_tests/triton_tests/test_batched_gemm_afp4wfp4_pre_quant.py`  (+30/-10)
- `op_tests/triton_tests/test_gemm_a8w8_blockscale.py`  (+34/-5)
- `op_tests/triton_tests/test_gemm_a8w8.py`  (+26/-4)
- `op_tests/op_benchmarks/triton/bench_rmsnorm.py`  (+12/-14)
- `op_tests/op_benchmarks/triton/utils/benchmark_utils.py`  (+18/-6)
- `op_tests/triton_tests/test_gemm_afp4wfp4.py`  (+17/-7)
- `op_tests/op_benchmarks/triton/bench_gemm_a8w8.py`  (+11/-11)
- `op_tests/op_benchmarks/triton/bench_gemm_a8w8_blockscale.py`  (+9/-11)
- `op_tests/op_benchmarks/triton/bench_gemm_a16w16.py`  (+10/-8)

## Key added lines (kernel files)

**`op_tests/op_benchmarks/triton/bench_batched_gemm_afp4wfp4.py`**
```
import aiter.ops.triton.utils.arch_info as arch_info
M_list = [4096] if args.model == "all" else [2**i for i in range(0, 15)]
for model_name, config in configs.items():
(model_name, M, N, K, 16)
```

**`op_tests/op_benchmarks/triton/bench_batched_gemm_afp4wfp4_pre_quant.py`**
```
import aiter.ops.triton.utils.arch_info as arch_info
M_list = [4096] if args.M is not None else [2**i for i in range(0, 15)]
for model_name, config in configs.items():
(model_name, M, N, K, 16)
```

**`op_tests/op_benchmarks/triton/bench_extend_attention.py`**
```
bench_MLA.run(save_path="." if args.o else None, print_data=True, show_plots=False)
parser.add_argument(
action="store_true",
default=False,
```

**`op_tests/op_benchmarks/triton/bench_gemm_a16w16.py`**
```
def bench_gemm_fn(M: int, N: int, K: int, metric: str, layout: str, model_name=None):
M, N, K, c_dtype, layout=layout, output=True
def bench_gemm_a16w16(
M, hidden_dim, intermediate_dim, metric, layer, model_name=None, **kwargs
```

**`op_tests/op_benchmarks/triton/bench_gemm_a8w8.py`**
```
def bench_gemm_fn(M: int, N: int, K: int, metric: str, layout: str):
M, N, K, str_to_torch_dtype["fp8e4m3"], c_dtype, layout=layout, output=True
def bench_gemm_a8w8(
M, hidden_dim, intermediate_dim, metric, layer, model_name=None, **kwargs
```
