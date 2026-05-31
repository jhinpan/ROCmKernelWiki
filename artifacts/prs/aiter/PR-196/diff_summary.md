# Diff summary

- **files changed:** 7
- **lines:** +13 / -12
- **kernel-ish files:** 6

## Files (by churn)

- `op_tests/test_gemm_a8w8_blockscale.py`  (+5/-5)
- `aiter/fused_moe_bf16_asm.py`  (+3/-2)
- `3rdparty/composable_kernel`  (+1/-1)
- `csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py`  (+1/-1)
- `csrc/ck_gemm_a8w8/gemm_a8w8_tune.py`  (+1/-1)
- `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py`  (+1/-1)
- `csrc/py_itfs_cu/asm_fmoe.cpp`  (+1/-1)

## Key added lines (kernel files)

**`aiter/fused_moe_bf16_asm.py`**
```
global_E = E
global_E = expert_mask.numel()
sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting_ck(topk_ids, topk_weight, global_E,
```

**`csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py`**
```
print(f"Tuning result for B:{b}, M:{m}, N:{n}, K:{k} is kernelId={best_kernelId} {kernels_list[best_kernelId].name} {spl
```

**`csrc/ck_gemm_a8w8/gemm_a8w8_tune.py`**
```
print(f"Tuning result for M:{m}, N:{n}, K:{k} is kernelId={best_kernelId} {kernels_list[best_kernelId].name} {splitK=}, 
```

**`csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py`**
```
print(f"Tuning result for M:{m}, N:{n}, K:{k} is kernelId={best_kernelId} {kernels_list[best_kernelId].name} {splitK=}, 
```

**`csrc/py_itfs_cu/asm_fmoe.cpp`**
```
uint32_t local_round = (tg_num + num_cu - 1) / num_cu;
```
