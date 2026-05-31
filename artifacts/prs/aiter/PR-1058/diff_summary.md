# Diff summary

- **files changed:** 22
- **lines:** +171 / -105
- **kernel-ish files:** 16

## Files (by churn)

- `op_tests/test_pa_mtp.py`  (+67/-25)
- `hsa/gfx942/pa/pa_asm.csv`  (+40/-37)
- `csrc/py_itfs_cu/asm_pa.cu`  (+36/-18)
- `hsa/gfx942/pa/codegen.py`  (+5/-4)
- `aiter/ops/batched_gemm_op_a8w8.py`  (+3/-3)
- `aiter/ops/batched_gemm_op_bf16.py`  (+3/-3)
- `aiter/ops/gemm_op_a8w8.py`  (+3/-2)
- `aiter/jit/core.py`  (+2/-2)
- `op_tests/test_mha.py`  (+2/-2)
- `setup.py`  (+2/-2)
- `aiter/fused_moe_dp_shared_expert.py`  (+1/-1)
- `aiter/jit/utils/cpp_extension.py`  (+1/-1)
- `op_tests/op_benchmarks/triton/bench_la_paged_decode.py`  (+1/-1)
- `op_tests/op_benchmarks/triton/bench_moe.py`  (+1/-1)
- `op_tests/test_mha_varlen.py`  (+1/-1)

## Key added lines (kernel files)

**`aiter/fused_moe_dp_shared_expert.py`**
```
get_padded_M(token_num),  # consider token_num > 1024 as prefill
```

**`aiter/jit/core.py`**
```
if max_jobs_env is not None:
elif data.get(ops_name) is None:
```

**`aiter/jit/utils/cpp_extension.py`**
```
if prebuild_thread_num is not None:
```

**`aiter/ops/batched_gemm_op_a8w8.py`**
```
if config is not None:
if splitK is None:
if ck_config is not None:
```

**`aiter/ops/batched_gemm_op_bf16.py`**
```
if config is not None:
if splitK is None:
if ck_config is not None:
```
