# Diff summary

- **files changed:** 21
- **lines:** +589 / -1
- **kernel-ish files:** 7

## Files (by churn)

- `op_tests/test_moe_tkw1.py`  (+254/-0)
- `aiter/fused_moe_bf16_asm.py`  (+188/-0)
- `csrc/py_itfs_cu/asm_fmoe.cpp`  (+106/-0)
- `aiter/ops/moe_op.py`  (+17/-0)
- `csrc/include/moe_op.h`  (+15/-0)
- `csrc/include/rocm_ops.hpp`  (+9/-0)
- `op_tests/test_moe.py`  (+0/-1)
- `hsa/fmoe/gelu/fmoe_fp8_g1u1_subGU_128_gelu_tkw1.co`  (+0/-0)
- `hsa/fmoe/gelu/fmoe_fp8_g1u1_subGU_192_gelu_tkw1.co`  (+0/-0)
- `hsa/fmoe/gelu/fmoe_fp8_g1u1_subGU_256_gelu_tkw1.co`  (+0/-0)
- `hsa/fmoe/gelu/fmoe_fp8_g1u1_subGU_320_gelu_tkw1.co`  (+0/-0)
- `hsa/fmoe/gelu/fmoe_fp8_g1u1_subGU_384_gelu_tkw1.co`  (+0/-0)
- `hsa/fmoe/gelu/fmoe_fp8_g1u1_subGU_448_gelu_tkw1.co`  (+0/-0)
- `hsa/fmoe/gelu/fmoe_fp8_g1u1_subGU_512_gelu_tkw1.co`  (+0/-0)
- `hsa/fmoe/silu/fmoe_fp8_g1u1_subGU_128_silu_tkw1.co`  (+0/-0)

## Key added lines (kernel files)

**`aiter/fused_moe_bf16_asm.py`**
```
from aiter import pertoken_quant
def asm_moe_tkw1(hidden_states,
w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
w2,  # [expert(local_expert:EP), dim, inter_dim]
```

**`aiter/ops/moe_op.py`**
```
@compile_ops("module_moe_asm")
def fmoe_g1u1_tkw1(
out: Tensor,
input: Tensor,
```

**`csrc/include/moe_op.h`**
```
void fmoe_g1u1_tkw1(torch::Tensor &out,                                           // [token_cnt, dim]
torch::Tensor &input,                                         // [token_cnt, dim] M,K
torch::Tensor &gate,                                          // [expert, hidden_dim*2, dim] N,K
torch::Tensor &down,                                          // [expert, hidden_dim, dim]
```

**`csrc/include/rocm_ops.hpp`**
```
m.def("fmoe_g1u1_tkw1", &fmoe_g1u1_tkw1,                                   \
py::arg("out"), py::arg("input"),                                    \
py::arg("gate"), py::arg("down"),                                    \
py::arg("sorted_token_ids"), py::arg("sorted_weight_buf"),           \
```

**`csrc/py_itfs_cu/asm_fmoe.cpp`**
```
void fmoe_g1u1_tkw1(torch::Tensor &out,                            // [token_cnt, dim]
torch::Tensor &input,                          // [token_cnt, dim] M,K
torch::Tensor &gate,                           // [expert, inter_dim*2, dim] N,K
torch::Tensor &down,                           // [expert, dim, inter_dim]
```
