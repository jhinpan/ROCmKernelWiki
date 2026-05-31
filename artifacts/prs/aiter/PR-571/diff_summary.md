# Diff summary

- **files changed:** 22
- **lines:** +438 / -384
- **kernel-ish files:** 20

## Files (by churn)

- `op_tests/multigpu_tests/test_dispatch_combine.py`  (+350/-0)
- `csrc/py_itfs_ck/moe_kernels.cu`  (+0/-127)
- `aiter/fused_moe.py`  (+39/-12)
- `csrc/include/moe_ck.h`  (+17/-29)
- `csrc/include/rocm_ops.hpp`  (+14/-28)
- `op_tests/test_moe.py`  (+2/-39)
- `csrc/cpp_itfs/moe/test_asm_moe.py`  (+1/-39)
- `op_tests/test_moe_tkw1.py`  (+2/-35)
- `op_tests/test_moe_ep.py`  (+1/-34)
- `aiter/ops/moe_op.py`  (+0/-16)
- `aiter/jit/optCompilerConfig.json`  (+0/-15)
- `csrc/py_itfs_ck/moe_sorting_kernels.cu`  (+3/-1)
- `aiter/ops/communication.py`  (+2/-1)
- `csrc/include/moe_sorting.h`  (+2/-1)
- `3rdparty/composable_kernel`  (+1/-1)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
num_local_tokens=None,
num_local_tokens,
expert_mask: Optional[torch.tensor] = None,  # EP
w1_scale: Optional[torch.tensor] = None,  # [expert(local_expert:EP), inter_dim, 1]
```

**`aiter/ops/communication.py`**
```
backend="cpu:gloo,cuda:nccl",
```

**`aiter/ops/moe_sorting.py`**
```
num_local_tokens: Optional[torch.Tensor] = None,
```

**`aiter/ops/quant.py`**
```
triton.quant.dynamic_per_token_quant_fp8_i8(y, x.view(-1, x.shape[-1]), scale)
```

**`csrc/include/moe_ck.h`**
```
void ck_moe_stage1(torch::Tensor& hidden_states, // [m, k], input token
torch::Tensor& w1, // [e, n, k]/[e, 2*n, k], pre-shuffle([e, nr, kr, w])
torch::Tensor& w2, // [e, n, k], pre-shuffle([e, nr, kr, w])
torch::Tensor& sorted_token_ids,  // [max_num_tokens_padded]
```
