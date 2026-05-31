# Diff summary

- **files changed:** 21
- **lines:** +147 / -106
- **kernel-ish files:** 7

## Files (by churn)

- `op_tests/test_moe.py`  (+60/-31)
- `csrc/py_itfs_cu/asm_fmoe.cpp`  (+40/-23)
- `csrc/py_itfs_ck/moe_kernels.cu`  (+13/-27)
- `csrc/include/moe_op.h`  (+13/-13)
- `csrc/rocm_ops.cpp`  (+13/-6)
- `csrc/pybind/moe_op_pybind.cu`  (+8/-1)
- `ater/ops/moe_op.py`  (+0/-5)
- `hsa/fmoe_fp8_g1u1_subGU_128.co`  (+0/-0)
- `hsa/fmoe_fp8_g1u1_subGU_192.co`  (+0/-0)
- `hsa/fmoe_fp8_g1u1_subGU_256.co`  (+0/-0)
- `hsa/fmoe_fp8_g1u1_subGU_320.co`  (+0/-0)
- `hsa/fmoe_fp8_g1u1_subGU_384.co`  (+0/-0)
- `hsa/fmoe_fp8_g1u1_subGU_448.co`  (+0/-0)
- `hsa/fmoe_fp8_g1u1_subGU_512.co`  (+0/-0)
- `hsa/fmoe_int8_g1u1_subGU_128.co`  (+0/-0)

## Key added lines (kernel files)

**`csrc/include/moe_op.h`**
```
void fmoe_g1u1(torch::Tensor &out,                           // [token_cnt, dim]
torch::Tensor &input,                         // [token_cnt, dim] M,K
torch::Tensor &gate,                          // [expert, hidden_dim*2, dim] N,K
torch::Tensor &down,                          // [expert, hidden_dim, dim]
```

**`csrc/py_itfs_ck/moe_kernels.cu`**
```
int shared_intermediate_size = w2.size(-1);
auto out = torch::empty({tokens, hidden_size}, torch::TensorOptions().dtype(hidden_states.dtype()).device(device));
int gate_only = 1;
if (shared_intermediate_size_0 == 2 * shared_intermediate_size)
```

**`csrc/py_itfs_cu/asm_fmoe.cpp`**
```
TORCH_CHECK(false, __func__, " Unsupported hidden_dim " + std::to_string(hidden_dim) + ", which should be divisible by 1
TORCH_CHECK(false, __func__, " Input only supput Int8!");
void fmoe_g1u1(torch::Tensor &out,                                          // [token_cnt, dim]
torch::Tensor &input,                                        // [token_cnt, dim] M,K
```

**`csrc/pybind/moe_op_pybind.cu`**
```
m.def("fmoe_g1u1", &fmoe_g1u1,
py::arg("out"), py::arg("input"),
py::arg("gate"), py::arg("down"),
py::arg("sorted_token_ids"), py::arg("sorted_weight_buf"),
```

**`csrc/rocm_ops.cpp`**
```
m.def("fmoe_g1u1", &fmoe_g1u1,
py::arg("out"), py::arg("input"),
py::arg("gate"), py::arg("down"),
py::arg("sorted_token_ids"), py::arg("sorted_weight_buf"),
```
