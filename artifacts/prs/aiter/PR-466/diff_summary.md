# Diff summary

- **files changed:** 13
- **lines:** +122 / -59
- **kernel-ish files:** 13

## Files (by churn)

- `csrc/include/rocm_ops.hpp`  (+29/-29)
- `csrc/kernels/topk_softmax_kernels.cu`  (+17/-13)
- `csrc/include/moe_op.h`  (+16/-11)
- `csrc/kernels/cache_kernels.cu`  (+21/-0)
- `csrc/kernels/activation_kernels.cu`  (+8/-0)
- `csrc/include/cache.h`  (+5/-1)
- `csrc/kernels/quant_kernels.cu`  (+3/-3)
- `csrc/include/custom_all_reduce.h`  (+5/-0)
- `csrc/include/activation.h`  (+4/-0)
- `csrc/include/quant.h`  (+4/-0)
- `csrc/kernels/custom_all_reduce.cu`  (+4/-0)
- `csrc/kernels/moe_align_block_size_kernels.cu`  (+4/-0)
- `csrc/kernels/topk_softmax_kernels_group.cu`  (+2/-2)

## Key added lines (kernel files)

**`csrc/include/activation.h`**
```
namespace aiter {
} // namespace aiter
```

**`csrc/include/cache.h`**
```
namespace aiter {
const double scale, const std::string &kv_cache_dtype);
} // namespace aiter
```

**`csrc/include/custom_all_reduce.h`**
```
namespace aiter {
} // namespace aiter
```

**`csrc/include/moe_op.h`**
```
namespace aiter {
void topk_softmax(torch::Tensor &topk_weights, torch::Tensor &topk_indices,
torch::Tensor &token_expert_indices,
torch::Tensor &gating_output,
```

**`csrc/include/quant.h`**
```
namespace aiter {
} // namespace aiter
```
