# Diff summary

- **files changed:** 48
- **lines:** +216 / -216
- **kernel-ish files:** 6

## Files (by churn)

- `csrc/py_itfs_cu/asm_fmoe.cpp`  (+136/-183)
- `csrc/include/moe_op.h`  (+23/-16)
- `csrc/include/rocm_ops.hpp`  (+18/-5)
- `op_tests/test_moe.py`  (+14/-7)
- `aiter/fused_moe_bf16_asm.py`  (+15/-5)
- `aiter/ops/moe_op.py`  (+10/-0)
- `hsa/fmoe/gelu/fmoe_fp8_g1u1_subGU_128_gelu.co`  (+0/-0)
- `hsa/fmoe/gelu/fmoe_fp8_g1u1_subGU_192_gelu.co`  (+0/-0)
- `hsa/fmoe/gelu/fmoe_fp8_g1u1_subGU_256_gelu.co`  (+0/-0)
- `hsa/fmoe/gelu/fmoe_fp8_g1u1_subGU_320_gelu.co`  (+0/-0)
- `hsa/fmoe/gelu/fmoe_fp8_g1u1_subGU_384_gelu.co`  (+0/-0)
- `hsa/fmoe/gelu/fmoe_fp8_g1u1_subGU_448_gelu.co`  (+0/-0)
- `hsa/fmoe/gelu/fmoe_fp8_g1u1_subGU_512_gelu.co`  (+0/-0)
- `hsa/fmoe/gelu/fmoe_int8_g1u0_subGU_128_gelu.co`  (+0/-0)
- `hsa/fmoe/gelu/fmoe_int8_g1u0_subGU_192_gelu.co`  (+0/-0)

## Key added lines (kernel files)

**`aiter/fused_moe_bf16_asm.py`**
```
from aiter import ActivationType
expert_mask=None,
activation = ActivationType.Silu
fc2_smooth_scale, activation)
```

**`aiter/ops/moe_op.py`**
```
@compile_ops("module_moe_asm", fc_name='ActivationType')
class _ActivationType():
ActivationType = _ActivationType(0)
activation: Optional[_ActivationType] = ActivationType.Silu,
```

**`csrc/include/moe_op.h`**
```
namespace py = pybind11;
enum class ActivationType : int
torch::Tensor &fc2_smooth_scale,  // [expert, 1, hidden_dim]
ActivationType activation = ActivationType::Silu);
```

**`csrc/include/rocm_ops.hpp`**
```
py::enum_<ActivationType>(m, "ActivationType")                             \
.value("Silu", ActivationType::Silu)                                   \
.value("Gelu", ActivationType::Gelu)                                   \
.export_values();                                                      \
```

**`csrc/py_itfs_cu/asm_fmoe.cpp`**
```
torch::Tensor &fc2_smooth_scale,  // [expert, 1, inter_dim],
ActivationType activation)
static FMoeKernel impl_int8_512("fmoe_int8_g1u0_subGU_512", "fmoe/silu/fmoe_int8_g1u0_subGU_512.co", 512);
static FMoeKernel impl_int8_448("fmoe_int8_g1u0_subGU_448", "fmoe/silu/fmoe_int8_g1u0_subGU_448.co", 448);
```
