# Diff summary

- **files changed:** 8
- **lines:** +233 / -103
- **kernel-ish files:** 7

## Files (by churn)

- `csrc/py_itfs_cu/asm_fmoe.cpp`  (+135/-64)
- `aiter/fused_moe_bf16_asm.py`  (+40/-16)
- `op_tests/test_moe.py`  (+25/-22)
- `aiter/ops/moe_op.py`  (+17/-0)
- `csrc/include/moe_op.h`  (+14/-1)
- `csrc/pybind/moe_op_pybind.cu`  (+1/-0)
- `csrc/rocm_ops.cpp`  (+1/-0)
- `hsa/fmoe_fp8_g1u1_smf_subGU_512.co`  (+0/-0)

## Key added lines (kernel files)

**`aiter/fused_moe_bf16_asm.py`**
```
def asm_moe(hidden_states,
w1,  # [expert, inter_dim*2, dim] N,K
w2,  # [expert, dim, inter_dim]
topk_weight, topk_ids,
```

**`aiter/ops/moe_op.py`**
```
@compile_ops("module_moe_asm")
def fmoe_fp8_g1u1_a16(
out: Tensor,
input: Tensor,  # bf16
```

**`csrc/include/moe_op.h`**
```
void fmoe_fp8_g1u1_a16(torch::Tensor &out,                    // [token_cnt, dim]
torch::Tensor &input,                  // [token_cnt, dim] M,K
torch::Tensor &gate,                   // [expert, inter_dim, dim] N,K
torch::Tensor &down,                   // [expert, dim, inter_dim]
```

**`csrc/py_itfs_cu/asm_fmoe.cpp`**
```
unsigned int inter_dim;
int inter_dim = w2.size(2);
int stride_D = inter_dim * I_elemSize;
int stride_expert_GU = stride_GU * inter_dim;
```

**`csrc/pybind/moe_op_pybind.cu`**
```
m.def("fmoe_fp8_g1u1_a16", &fmoe_fp8_g1u1_a16);
```
