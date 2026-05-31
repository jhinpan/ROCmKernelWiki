# Diff summary

- **files changed:** 115
- **lines:** +948 / -421
- **kernel-ish files:** 57

## Files (by churn)

- `csrc/py_itfs_ck/moe_ck_2stages_kernel.cu`  (+219/-86)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm.hpp`  (+53/-40)
- `csrc/include/rocm_ops.hpp`  (+34/-32)
- `csrc/py_itfs_cu/asm_fmoe.cpp`  (+28/-28)
- `hsa/fmoe_2stages/fmoe_stage1_bf16_pertokenFp8_doweight_g1u1.csv`  (+55/-0)
- `aiter/test_common.py`  (+45/-8)
- `hsa/fmoe_2stages/tune.py`  (+40/-11)
- `aiter/fused_moe.py`  (+36/-9)
- `csrc/include/moe_op.h`  (+22/-22)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm_common.cuh`  (+20/-15)
- `csrc/py_itfs_cu/asm_moe_2stage.cpp`  (+24/-8)
- `aiter/ops/moe_op.py`  (+15/-12)
- `aiter/configs/tuned_fmoe.csv`  (+12/-11)
- `aiter/configs/untuned_fmoe.csv`  (+12/-11)
- `op_tests/test_moe_2stage.py`  (+14/-6)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
doweight_stage1=False,
doweight_stage1,
assert doweight_stage1 == False, "doweight_stage1 not support in fused_moe_1stage"
doweight_stage1=doweight_stage1,
```

**`aiter/fused_moe_bf16_asm.py`**
```
fc1_scale, a1_scale, block_size, sorted_weights)
sorted_expert_ids,
num_valid_ids, moe_buf, topk, fc2_scale, a2_scale, block_size, sorted_weights)
```

**`aiter/ops/moe_op.py`**
```
sorted_weights: Tensor,
sorted_weights: Tensor,
sorted_weights: Tensor,
sorted_weights: Tensor,
```

**`aiter/ops/quant.py`**
```
return y, scale.view(1)
```

**`aiter/test_common.py`**
```
iter_used_memory, inputSize, _, _ = device_memory_profiling(func, *args, **kwargs)
free_memory = torch.cuda.mem_get_info(gpu_id)[0]
(free_memory - iter_used_memory + inputSize) * 0.9,
cache_size = max(cache_size, 0)
```
