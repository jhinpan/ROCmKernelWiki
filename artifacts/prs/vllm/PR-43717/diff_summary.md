# Diff summary

- **files changed:** 23
- **lines:** +911 / -722
- **kernel-ish files:** 22

## Files (by churn)

- `csrc/libtorch_stable/cache_kernels.cu`  (+360/-311)
- `csrc/libtorch_stable/cache_kernels_fused.cu`  (+102/-89)
- `csrc/torch_bindings.cpp`  (+4/-138)
- `csrc/libtorch_stable/torch_bindings.cpp`  (+141/-0)
- `csrc/libtorch_stable/ops.h`  (+129/-0)
- `csrc/libtorch_stable/attention/paged_attention_v2.cu`  (+47/-41)
- `csrc/libtorch_stable/attention/paged_attention_v1.cu`  (+41/-37)
- `csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu`  (+36/-34)
- `csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh`  (+17/-13)
- `csrc/ops.h`  (+0/-23)
- `csrc/quantization/w8a8/fp8/amd/quant_utils.cuh`  (+12/-9)
- `CMakeLists.txt`  (+9/-11)
- `csrc/libtorch_stable/attention/attention_kernels.cuh`  (+4/-7)
- `csrc/libtorch_stable/attention/attention_utils.cuh`  (+2/-2)
- `csrc/libtorch_stable/activation_kernels.cu`  (+1/-1)

## Key added lines (kernel files)

**`csrc/libtorch_stable/attention/paged_attention_v1.cu`**
```
torch::stable::Tensor& out, torch::stable::Tensor& query,
torch::stable::Tensor& key_cache, torch::stable::Tensor& value_cache,
int num_kv_heads, float scale, torch::stable::Tensor& block_tables,
torch::stable::Tensor& seq_lens, int max_seq_len,
```

**`csrc/libtorch_stable/attention/paged_attention_v2.cu`**
```
torch::stable::Tensor& out, torch::stable::Tensor& exp_sums,
torch::stable::Tensor& max_logits, torch::stable::Tensor& tmp_out,
torch::stable::Tensor& query, torch::stable::Tensor& key_cache,
torch::stable::Tensor& value_cache, int num_kv_heads, float scale,
```

**`csrc/libtorch_stable/cache_kernels.cu`**
```
void swap_blocks(torch::stable::Tensor& src, torch::stable::Tensor& dst,
const torch::stable::Tensor& block_mapping) {
torch::stable::Device src_device = src.device();
torch::stable::Device dst_device = dst.device();
```

**`csrc/libtorch_stable/cache_kernels_fused.cu`**
```
do {                                                                     \
VLLM_STABLE_DISPATCH_FLOATING_TYPES(                                   \
q_pe.scalar_type(), "qk_scalar_type", [&] {                        \
using qk_t = scalar_t;                                           \
```

**`csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu`**
```
void reshape_and_cache_nvfp4_dispatch(torch::stable::Tensor& key,
torch::stable::Tensor& value,
torch::stable::Tensor& key_cache,
torch::stable::Tensor& value_cache,
```
