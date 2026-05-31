# Diff summary

- **files changed:** 8
- **lines:** +135 / -26
- **kernel-ish files:** 7

## Files (by churn)

- `csrc/py_itfs_cu/asm_fmoe.cpp`  (+48/-6)
- `ater/test_common.py`  (+30/-12)
- `op_tests/test_moe.py`  (+13/-7)
- `ater/ops/moe_op.py`  (+18/-0)
- `csrc/include/moe_op.h`  (+15/-0)
- `ater/fused_moe_bf16_asm.py`  (+10/-1)
- `csrc/pybind/moe_op_pybind.cu`  (+1/-0)
- `hsa/fmoe_int8_g1u0_smf.co`  (+0/-0)

## Key added lines (kernel files)

**`ater/fused_moe_bf16_asm.py`**
```
num_tokens_post_pad, moe_buf, num_experts, BLOCK_SIZE_M)
a16=False
elif a16:
ater.fmoe_int8_g1u0_a16(moe_buf, hidden_states, w1, w2, sorted_ids,
```

**`ater/ops/moe_op.py`**
```
@compile_ops(**compile_ops_)
def fmoe_int8_g1u0_a16(
out: Tensor,
input: Tensor,  # bf16
```

**`ater/test_common.py`**
```
def perftest(num_iters=100, num_warmup=20, testGraph=False):
for _ in range(num_warmup):
data = func(*args, **kwargs)
for _ in range(num_iters):
```

**`csrc/include/moe_op.h`**
```
void fmoe_int8_g1u0_a16(torch::Tensor &out,                    // [token_cnt, dim]
torch::Tensor &input,                  // [token_cnt, dim] M,K
torch::Tensor &gate,                   // [expert, inter_dim, dim] N,K
torch::Tensor &down,                   // [expert, dim, inter_dim]
```

**`csrc/py_itfs_cu/asm_fmoe.cpp`**
```
template <typename T, typename T_O, bool switchGxy = false>
int stride_X = input.stride(0) * input.element_size();
if constexpr (switchGxy)
HIP_CALL(hipModuleLaunchKernel(kernel_func,
```
