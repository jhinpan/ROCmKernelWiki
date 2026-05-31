# Diff summary

- **files changed:** 19
- **lines:** +541 / -164
- **kernel-ish files:** 17

## Files (by churn)

- `op_tests/test_moe_ep.py`  (+296/-0)
- `op_tests/test_moe_sorting.py`  (+118/-0)
- `op_tests/test_moe.py`  (+7/-102)
- `aiter/fused_moe_bf16_asm.py`  (+29/-14)
- `csrc/py_itfs_ck/moe_kernels.cu`  (+20/-15)
- `csrc/kernels/topk_softmax_kernels.cu`  (+21/-8)
- `aiter/fused_moe_gelu.py`  (+12/-8)
- `csrc/py_itfs_ck/moe_sorting_kernels.cu`  (+8/-4)
- `csrc/rocm_ops.cpp`  (+7/-2)
- `csrc/pybind/moe_sorting_pybind.cu`  (+6/-1)
- `aiter/jit/core.py`  (+3/-2)
- `aiter/jit/optCompilerConfig.json`  (+3/-1)
- `aiter/ops/moe_op.py`  (+2/-1)
- `aiter/ops/moe_sorting.py`  (+2/-1)
- `csrc/include/moe_ck.h`  (+2/-1)

## Key added lines (kernel files)

**`aiter/fused_moe_bf16_asm.py`**
```
def moe_sorting_ck(topk_ids, topk_weights, num_experts, model_dim, moebuf_dtype, expert_mask = None):
num_tokens_post_pad, moe_buf, num_experts, BLOCK_SIZE_M, expert_mask)
w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
w2,  # [expert(local_expert:EP), dim, inter_dim]
```

**`aiter/fused_moe_gelu.py`**
```
topk_ids: Optional[torch.Tensor] = None,
topk_weights: Optional[torch.Tensor] = None,
if topk_weights is None:
topk_weights = torch.empty(M,
```

**`aiter/jit/core.py`**
```
if os.path.exists(f'{this_dir}/{md_name}.so'):
os.remove(f'{this_dir}/{md_name}.so')
```

**`aiter/ops/moe_op.py`**
```
block_m: Optional[int] = 32,
expert_mask: Optional[Tensor] = None
```

**`aiter/ops/moe_sorting.py`**
```
unit_size: int,
local_expert_mask: Optional[Tensor]=None): ...
```
