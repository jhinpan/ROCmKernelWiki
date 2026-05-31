# Diff summary

- **files changed:** 25
- **lines:** +306 / -106
- **kernel-ish files:** 23

## Files (by churn)

- `op_tests/op_benchmarks/triton/bench_schema.yaml`  (+67/-0)
- `op_tests/op_benchmarks/triton/bench_hstu_attn.py`  (+30/-7)
- `op_tests/op_benchmarks/triton/bench_pa_prefill.py`  (+21/-9)
- `op_tests/op_benchmarks/triton/bench_extend_attention.py`  (+12/-15)
- `op_tests/op_benchmarks/triton/bench_rmsnorm.py`  (+20/-7)
- `op_tests/op_benchmarks/triton/bench_la_paged_decode.py`  (+21/-5)
- `op_tests/op_benchmarks/triton/bench_moe_routing_sigmoid_top1_fused.py`  (+15/-4)
- `op_tests/op_benchmarks/triton/utils/benchmark_utils.py`  (+15/-3)
- `op_tests/op_benchmarks/triton/bench_pa_decode.py`  (+15/-2)
- `op_tests/op_benchmarks/triton/bench_mha.py`  (+8/-8)
- `op_tests/op_benchmarks/triton/bench_moe_align_block_size.py`  (+13/-2)
- `op_tests/op_benchmarks/triton/bench_moe_mx.py`  (+11/-3)
- `op_tests/op_benchmarks/triton/bench_batched_gemm_a16w16.py`  (+6/-5)
- `op_tests/op_benchmarks/triton/bench_batched_gemm_a8w8.py`  (+6/-5)
- `op_tests/op_benchmarks/triton/bench_gemm_a8w8_per_token_scale.py`  (+9/-2)

## Key added lines (kernel files)

**`op_tests/op_benchmarks/triton/bench_batched_gemm_a16w16.py`**
```
get_caller_name_no_ext,
plot_name=get_caller_name_no_ext(),
plot_name=get_caller_name_no_ext(),
def bench_batched_gemm_a8w8(batch, M, N, K, metric, **kwargs):
```

**`op_tests/op_benchmarks/triton/bench_batched_gemm_a8w8.py`**
```
get_caller_name_no_ext,
plot_name=get_caller_name_no_ext(),
plot_name=get_caller_name_no_ext(),
def bench_batched_gemm_a8w8(batch, M, N, K, metric, **kwargs):
```

**`op_tests/op_benchmarks/triton/bench_batched_gemm_afp4wfp4.py`**
```
get_caller_name_no_ext,
plot_name=get_caller_name_no_ext(),
plot_name=get_caller_name_no_ext(),
print_vgpr(fun, get_caller_name_no_ext())
```

**`op_tests/op_benchmarks/triton/bench_batched_gemm_afp4wfp4_pre_quant.py`**
```
get_caller_name_no_ext,
plot_name=get_caller_name_no_ext(),
plot_name=get_caller_name_no_ext(),
print_vgpr(fun, get_caller_name_no_ext())
```

**`op_tests/op_benchmarks/triton/bench_extend_attention.py`**
```
get_caller_name_no_ext,
line_vals = ["fwd_Time_(ms)"]
line_arg="metric",
plot_name=get_caller_name_no_ext(),
```
