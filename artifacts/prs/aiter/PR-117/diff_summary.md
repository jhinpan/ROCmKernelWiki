# Diff summary

- **files changed:** 19
- **lines:** +164 / -541
- **kernel-ish files:** 17

## Files (by churn)

- `op_tests/test_moe_ep.py`  (+0/-296)
- `op_tests/test_moe_sorting.py`  (+0/-118)
- `op_tests/test_moe.py`  (+102/-7)
- `aiter/fused_moe_bf16_asm.py`  (+14/-29)
- `csrc/py_itfs_ck/moe_kernels.cu`  (+15/-20)
- `csrc/kernels/topk_softmax_kernels.cu`  (+8/-21)
- `aiter/fused_moe_gelu.py`  (+8/-12)
- `csrc/py_itfs_ck/moe_sorting_kernels.cu`  (+4/-8)
- `csrc/rocm_ops.cpp`  (+2/-7)
- `csrc/pybind/moe_sorting_pybind.cu`  (+1/-6)
- `aiter/jit/core.py`  (+2/-3)
- `aiter/jit/optCompilerConfig.json`  (+1/-3)
- `aiter/ops/moe_op.py`  (+1/-2)
- `aiter/ops/moe_sorting.py`  (+1/-2)
- `csrc/include/moe_ck.h`  (+1/-2)

## Key added lines (kernel files)

**`aiter/fused_moe_bf16_asm.py`**
```
def moe_sorting_ck(topk_ids, topk_weights, num_experts, model_dim, moebuf_dtype):
num_tokens_post_pad, moe_buf, num_experts, BLOCK_SIZE_M)
w1,  # [expert, inter_dim*2, dim] N,K
w2,  # [expert, dim, inter_dim]
```

**`aiter/fused_moe_gelu.py`**
```
topk_weights = torch.empty(M,
dtype=torch.float32,
device=hidden_states.device)
topk_ids = torch.empty(M,
```

**`aiter/jit/core.py`**
```
if os.path.exists(f'{this_dir}/{md_name}.so'):
os.remove(f'{this_dir}/{md_name}.so')
```

**`aiter/ops/moe_op.py`**
```
block_m: Optional[int] = 32
```

**`aiter/ops/moe_sorting.py`**
```
unit_size: int): ...
```
