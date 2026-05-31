# Diff summary

- **files changed:** 9
- **lines:** +355 / -16
- **kernel-ish files:** 8

## Files (by churn)

- `op_tests/test_moe_blockscale.py`  (+226/-0)
- `csrc/py_itfs_cu/asm_fmoe.cpp`  (+59/-11)
- `aiter/ops/moe_op.py`  (+20/-0)
- `csrc/rocm_ops.cpp`  (+14/-5)
- `csrc/include/moe_op.h`  (+18/-0)
- `csrc/pybind/moe_op_pybind.cu`  (+10/-0)
- `aiter/test_common.py`  (+7/-0)
- `aiter/fused_moe_bf16_asm.py`  (+1/-0)
- `hsa/fmoe_fp8_blockscale_g1u1_subGU_256.co`  (+0/-0)

## Key added lines (kernel files)

**`aiter/ops/moe_op.py`**
```
@compile_ops("module_moe_asm")
def fmoe_fp8_blockscale_g1u1(
out: Tensor,
input: Tensor,
```

**`aiter/test_common.py`**
```
def run_perftest(func, *args, num_iters=101, num_warmup=10, **kwargs):
@perftest(num_iters=num_iters, num_warmup=num_warmup)
def worker():
return func(*args, **kwargs)
```

**`csrc/include/moe_op.h`**
```
void fmoe_fp8_blockscale_g1u1(torch::Tensor &out,                            // [token_cnt, dim]
torch::Tensor &input,                          // [token_cnt, dim] M,K
torch::Tensor &gate,                           // [expert, inter_dim*2, dim] N,K
torch::Tensor &down,                           // [expert, dim, inter_dim]
```

**`csrc/py_itfs_cu/asm_fmoe.cpp`**
```
int stride_expert_GUDQN = w1_dqn.has_value() ? w1_dqn.value().size(1) * sizeof(float) : 0;
int stride_expert_DDQN = w2_dqn.has_value() ? w2_dqn.value().size(1) * sizeof(float) : 0;
void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
&arg_size, HIP_LAUNCH_PARAM_END};
```

**`csrc/pybind/moe_op_pybind.cu`**
```
m.def("fmoe_fp8_blockscale_g1u1", &fmoe_fp8_blockscale_g1u1,
py::arg("out"), py::arg("input"),
py::arg("gate"), py::arg("down"),
py::arg("sorted_token_ids"), py::arg("sorted_weight_buf"),
```
