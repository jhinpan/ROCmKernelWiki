# Diff summary

- **files changed:** 13
- **lines:** +533 / -255
- **kernel-ish files:** 7

## Files (by churn)

- `aiter/ops/triton/moe_op.py`  (+180/-145)
- `aiter/ops/triton/moe_op_silu_fused.py`  (+62/-43)
- `aiter/ops/triton/moe_op_mxfp4.py`  (+31/-20)
- `op_tests/op_benchmarks/triton/bench_moe.py`  (+36/-10)
- `aiter/ops/triton/moe_op_gelu.py`  (+24/-18)
- `aiter/ops/triton/configs/moe/MI350X-MOE-DEFAULT.json`  (+35/-0)
- `aiter/ops/triton/configs/moe/MI350X-MOE-FP8_W8A8.json`  (+35/-0)
- `aiter/ops/triton/configs/moe/MI350X-MOE-INT4_W4A16.json`  (+35/-0)
- `aiter/ops/triton/configs/moe/MI350X-MOE-INT8_W8A16.json`  (+35/-0)
- `aiter/ops/triton/configs/moe/MI350X-MOE-INT8_W8A8.json`  (+35/-0)
- `op_tests/op_benchmarks/triton/utils/model_configs.json`  (+18/-6)
- `aiter/ops/triton/utils/moe_config_utils.py`  (+6/-11)
- `op_tests/op_benchmarks/triton/utils/benchmark_utils.py`  (+1/-2)

## Key added lines (kernel files)

**`aiter/ops/triton/moe_op.py`**
```
"EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
EVEN_K: tl.constexpr,
num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
```

**`aiter/ops/triton/moe_op_gelu.py`**
```
"EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
EVEN_K: tl.constexpr,
num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
```

**`aiter/ops/triton/moe_op_mxfp4.py`**
```
"EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
EVEN_K: tl.constexpr,
num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
```

**`aiter/ops/triton/moe_op_silu_fused.py`**
```
"EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
EVEN_K: tl.constexpr,
num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
```

**`aiter/ops/triton/utils/moe_config_utils.py`**
```
import warnings
dtype_str = "DEFAULT" if dtype is None else dtype
config_file_path = f"{AITER_TRITON_CONFIGS_PATH}/moe/{dev}-MOE-{dtype_str}.json"
warnings.warn(
```
