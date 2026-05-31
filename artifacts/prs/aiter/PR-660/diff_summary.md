# Diff summary

- **files changed:** 8
- **lines:** +175 / -89
- **kernel-ish files:** 7

## Files (by churn)

- `op_tests/test_moe_sorting.py`  (+120/-52)
- `csrc/py_itfs_ck/moe_sorting_kernels.cu`  (+43/-33)
- `aiter/fused_moe.py`  (+5/-1)
- `csrc/include/moe_sorting.h`  (+2/-1)
- `csrc/include/rocm_ops.hpp`  (+2/-1)
- `3rdparty/composable_kernel`  (+1/-1)
- `aiter/ops/moe_sorting.py`  (+1/-0)
- `op_tests/test_moe_sorting_mxfp4.py`  (+1/-0)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
dispatch_policy=0,
num_valid_ids = torch.empty((2), dtype=dtypes.i32, device=device)
dispatch_policy,
moe_sorting_dispatch_policy=0,
```

**`aiter/ops/moe_sorting.py`**
```
dispatch_policy: int = 0,
```

**`csrc/include/moe_sorting.h`**
```
std::optional<torch::Tensor> num_local_tokens = std::nullopt,
int dispatch_policy = 0);
```

**`csrc/include/rocm_ops.hpp`**
```
py::arg("num_local_tokens")  = std::nullopt, \
py::arg("dispatch_policy") = 0);
```

**`csrc/py_itfs_ck/moe_sorting_kernels.cu`**
```
void moe_sorting_fwd(torch::Tensor& topk_ids,          // [m, topk]
torch::Tensor& topk_weights,      // [m, topk]
torch::Tensor& sorted_token_ids,  // [max_num_tokens_padded]
torch::Tensor& sorted_weights,    // [max_num_tokens_padded]
```
