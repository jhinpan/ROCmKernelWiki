# Diff summary

- **files changed:** 31
- **lines:** +867 / -427
- **kernel-ish files:** 29

## Files (by churn)

- `csrc/py_itfs_ck/moe_ck_2stages_kernel.cu`  (+204/-90)
- `op_tests/test_moe_2stage.py`  (+137/-72)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm.hpp`  (+94/-59)
- `aiter/fused_moe_bf16_asm.py`  (+68/-53)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm_common.cuh`  (+26/-26)
- `aiter/fused_moe.py`  (+23/-27)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertensor_b16_f8.cu`  (+21/-6)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertoken_b16_f8.cu`  (+21/-6)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertoken_f16_f8.cu`  (+21/-6)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertensor_b16_i8.cu`  (+20/-6)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertensor_f16_f8.cu`  (+20/-6)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertensor_f16_i8.cu`  (+20/-6)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertoken_b16_i8.cu`  (+20/-6)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertoken_f16_i8.cu`  (+20/-6)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm1_instance_pertensor_b16_f8_wint4.cu`  (+14/-6)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
if q_dtype_w in [torch.bfloat16, torch.float16, torch.uint32, torch.int8]:
@functools.lru_cache()
def get1tensor(device):
return torch.tensor(1.0, dtype=torch.float, device=device)
```

**`aiter/fused_moe_bf16_asm.py`**
```
from aiter import pertoken_quant, get_hip_quant, get_torch_quant
from aiter import ActivationType, QuantType
from aiter import pertoken_quant, get_hip_quant, get_torch_quant
from aiter import ActivationType, QuantType
```

**`aiter/ops/moe_op.py`**
```
act_op: Optional[int] = 0,
```

**`csrc/include/moe_ck.h`**
```
std::optional<torch::Tensor> sorted_weights,
std::optional<int> act_op);
```

**`csrc/include/rocm_ops.hpp`**
```
py::arg("sorted_weights") = std::nullopt,  \
py::arg("act_op") = 0);                    \
```
