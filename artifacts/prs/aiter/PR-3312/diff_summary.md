# Diff summary

- **files changed:** 6
- **lines:** +109 / -30
- **kernel-ish files:** 6

## Files (by churn)

- `aiter/ops/quant.py`  (+43/-3)
- `aiter/fused_moe.py`  (+32/-13)
- `csrc/kernels/quant_kernels.cu`  (+30/-10)
- `csrc/include/rocm_ops.hpp`  (+2/-2)
- `csrc/include/quant.h`  (+1/-1)
- `op_tests/test_moe_sorting_mxfp4.py`  (+1/-1)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
from aiter import (
fused_dynamic_mxfp4_quant_moe_sort,
fused_dynamic_mxfp8_quant_moe_sort,
mxfp4_moe_sort_fwd,
```

**`aiter/ops/quant.py`**
```
def fused_dynamic_mx_quant_moe_sort_hip(
HIP path for fused dynamic MX (fp4 or fp8) quantization and MoE scale
sorting. The output dtype of ``out`` selects the quant target: fp4x2/uint8
for MXFP4, fp8 for MXFP8.
```

**`csrc/include/quant.h`**
```
void fused_dynamic_mx_quant_moe_sort_hip(aiter_tensor_t& out,         // [token_num * topk, d] for fp8 or [token_num * t
```

**`csrc/include/rocm_ops.hpp`**
```
m.def("fused_dynamic_mx_quant_moe_sort_hip",                      \
&aiter::fused_dynamic_mx_quant_moe_sort_hip,                \
```

**`csrc/kernels/quant_kernels.cu`**
```
constexpr float fp8_power2_limit = 1.0f / 128.0f;
constexpr float fp8_power2_limit = 1.0f / 256.0f;
? 0.25f /* 1/4, fp4 max=6 */
: (std::is_same_v<DTYPE_O, opus::fp8_t>
```
