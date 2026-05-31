# Diff summary

- **files changed:** 48
- **lines:** +505 / -895
- **kernel-ish files:** 21

## Files (by churn)

- `op_tests/triton_tests/test_moe_e2e.py`  (+0/-454)
- `op_tests/triton_tests/test_moe.py`  (+336/-39)
- `op_tests/triton_tests/test_moe_mx.py`  (+9/-174)
- `aiter/ops/triton/moe_op_gelu.py`  (+14/-82)
- `aiter/ops/triton/utils/types.py`  (+40/-0)
- `op_tests/op_benchmarks/triton/bench_moe.py`  (+12/-21)
- `aiter/ops/triton/moe_op_silu_fused.py`  (+5/-22)
- `aiter/ops/triton/moe_align_block_size.py`  (+10/-16)
- `aiter/ops/triton/moe_op.py`  (+3/-21)
- `aiter/ops/triton/utils/moe_common.py`  (+22/-0)
- `aiter/ops/triton/moe_op_mxfp4.py`  (+1/-20)
- `aiter/ops/triton/utils/arch_info.py`  (+16/-4)
- `aiter/ops/triton/utils/moe_config_utils.py`  (+12/-7)
- `op_tests/op_benchmarks/triton/bench_moe_mx.py`  (+6/-9)
- `op_tests/op_benchmarks/triton/bench_pa_decode.py`  (+7/-7)

## Key added lines (kernel files)

**`aiter/ops/triton/activation.py`**
```
@triton.jit
def _silu_exp2(x):
return x / (1.0 + tl.exp2(-(x * 1.44269504089)))
```

**`aiter/ops/triton/gemm_a16w16.py`**
```
fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM-A16W16.json"
```

**`aiter/ops/triton/gemm_a8w8.py`**
```
fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM-A8W8.json"
```

**`aiter/ops/triton/gemm_afp4wfp4.py`**
```
fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM-AFP4WFP4-N={N}-K={2*K}.json"
fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM-AFP4WFP4.json"
```

**`aiter/ops/triton/moe_align_block_size.py`**
```
def _moe_align_block_size_stage1_kernel(
def _moe_align_block_size_stage2_kernel(
def _moe_align_block_size_stage3_kernel(
def _moe_align_block_size_stage4_kernel(
```
