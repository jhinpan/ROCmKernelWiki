# Diff summary

- **files changed:** 24 (diff was byte-capped; summary is partial)
- **lines:** +3222 / -1648
- **kernel-ish files:** 23

## Files (by churn)

- `csrc/kernels/rope/rope_common.h`  (+1981/-650)
- `csrc/kernels/rope/pos_fwd_kernels.cu`  (+0/-327)
- `csrc/kernels/rope/general_bwd_kernels.cu`  (+0/-311)
- `csrc/kernels/rope/general_fwd_kernels.cu`  (+0/-310)
- `aiter/jit/optCompilerConfig.json`  (+178/-9)
- `csrc/kernels/rope/general_2c_cached_positions_offsets_fwd_kernels.cu`  (+99/-0)
- `csrc/kernels/rope/general_2c_cached_positions_fwd_kernels.cu`  (+94/-0)
- `csrc/kernels/rope/general_1c_cached_positions_offsets_fwd_kernels.cu`  (+81/-0)
- `csrc/kernels/rope/general_1c_cached_positions_fwd_kernels.cu`  (+76/-0)
- `csrc/kernels/rope/general_2c_cached_bwd_kernels.cu`  (+65/-0)
- `csrc/kernels/rope/general_2c_cached_fwd_kernels.cu`  (+65/-0)
- `csrc/kernels/rope/general_2c_uncached_bwd_kernels.cu`  (+63/-0)
- `csrc/kernels/rope/general_2c_uncached_fwd_kernels.cu`  (+63/-0)
- `csrc/kernels/rope/general_1c_2d_bwd_kernels.cu`  (+56/-0)
- `csrc/kernels/rope/general_1c_2d_fwd_kernels.cu`  (+56/-0)

## Key added lines (kernel files)

**`aiter/jit/core.py`**
```
"module_rope_1c_uncached_fwd",
"module_rope_1c_uncached_bwd",
"module_rope_2c_uncached_fwd",
"module_rope_2c_uncached_bwd",
```

**`aiter/ops/rope.py`**
```
@compile_ops("module_rope_1c_uncached_fwd")
@compile_ops("module_rope_1c_uncached_bwd")
@compile_ops("module_rope_2c_uncached_fwd")
@compile_ops("module_rope_2c_uncached_bwd")
```

**`csrc/include/rocm_ops.hpp`**
```
py::arg("nope_first"))
```

**`csrc/kernels/rope/general_1c_2d_bwd_kernels.cu`**
```
using namespace aiter;
void rope_2d_bwd_impl(
torch::Tensor&       input_grads,
const torch::Tensor& output_grads,
```

**`csrc/kernels/rope/general_1c_2d_fwd_kernels.cu`**
```
using namespace aiter;
void rope_2d_fwd_impl(
torch::Tensor&       output,
const torch::Tensor& input,
```
