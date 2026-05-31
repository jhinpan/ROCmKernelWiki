# Diff summary

- **files changed:** 32
- **lines:** +2523 / -129
- **kernel-ish files:** 30

## Files (by churn)

- `sgl-kernel/csrc/elementwise/dsv4_norm_rope.cu`  (+700/-0)
- `python/sglang/srt/layers/fused_qk_norm_rope_store.py`  (+380/-0)
- `sgl-kernel/csrc/elementwise/deepseek_v4_topk.cu`  (+372/-0)
- `python/sglang/jit_kernel/triton_store_cache.py`  (+237/-0)
- `python/sglang/srt/models/deepseek_v4.py`  (+153/-28)
- `python/sglang/srt/layers/attention/dsv4/compress_hip.py`  (+91/-20)
- `python/sglang/jit_kernel/triton/hash_topk.py`  (+99/-0)
- `sgl-kernel/python/sgl_kernel/elementwise.py`  (+79/-0)
- `python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/fp8_utils.cuh`  (+71/-2)
- `python/sglang/jit_kernel/dsv4/moe.py`  (+32/-19)
- `python/sglang/jit_kernel/include/sgl_kernel/warp.cuh`  (+20/-23)
- `python/sglang/srt/models/deepseek_v2.py`  (+39/-1)
- `sgl-kernel/include/sgl_kernel_ops.h`  (+37/-0)
- `sgl-kernel/python/sgl_kernel/top_k.py`  (+32/-0)
- `python/sglang/jit_kernel/include/sgl_kernel/runtime.cuh`  (+31/-0)

## Key added lines (kernel files)

**`python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh`**
```
device_.set_options<kDLGPU>();
device_.set_options<kDLGPU>();
```

**`python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh`**
```
device_.set_options<kDLGPU>();
device_.set_options<kDLGPU>();
```

**`python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh`**
```
uint32_t n = __shfl_up_sync(device::kFullMask, val, offset);
uint32_t n = __shfl_up(val, offset, 32);
val = max(val, __shfl_xor_sync(device::kFullMask, val, mask, 32));
val = max(val, __shfl_xor(val, mask, 32));
```

**`python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope_v2.cuh`**
```
const float other = __shfl_xor_sync(kFullMask, data[i], mask, kWarpThreads);
const float other = __shfl_xor(data[i], mask, kWarpThreads);
const auto x = cast<float>(cast<bf16_t>(data[0]));
const auto y = cast<float>(cast<bf16_t>(data[1]));
```

**`python/sglang/jit_kernel/dsv4/attn.py`**
```
is_hip_runtime,
if is_hip_runtime():
from sglang.jit_kernel.triton_store_cache import triton_fused_store_cache
triton_fused_store_cache(input, cache, indices, page_size=page_size, type=type)
```
