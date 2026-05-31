# Diff summary

- **files changed:** 34
- **lines:** +1052 / -67
- **kernel-ish files:** 18

## Files (by churn)

- `csrc/include/quant_common.cuh`  (+174/-0)
- `csrc/py_itfs_cu/asm_fmoe.cpp`  (+148/-17)
- `csrc/kernels/quant_kernels.cu`  (+164/-0)
- `csrc/py_itfs_ck/moe_kernels.cu`  (+125/-0)
- `op_tests/test_quant.py`  (+80/-0)
- `op_tests/test_moe.py`  (+51/-18)
- `ater/ops/moe_op.py`  (+48/-8)
- `ater/fused_moe_bf16_asm.py`  (+39/-14)
- `csrc/rocm_ops.cpp`  (+43/-6)
- `csrc/include/vectorization.cuh`  (+48/-0)
- `ater/jit/optCompilerConfig.json`  (+33/-2)
- `ater/ops/quant.py`  (+22/-2)
- `csrc/include/quant.h`  (+18/-0)
- `csrc/include/moe_ck.h`  (+16/-0)
- `csrc/include/moe_op.h`  (+15/-0)

## Key added lines (kernel files)

**`ater/fused_moe_bf16_asm.py`**
```
E, inter_dim, model_dim = w1.shape
if fc1_smooth_scale is not None:
a8 = torch.empty((topk * M, model_dim),
dtype=torch.int8, device=device)
```

**`ater/ops/moe_op.py`**
```
@compile_ops("module_moe_asm")
@compile_ops("module_moe_asm")
@compile_ops("module_moe_asm")
@compile_ops("module_moe_asm")
```

**`ater/ops/quant.py`**
```
@compile_ops("module_quant")
def static_scaled_fp8_quant(
out: Tensor, input: Tensor, scale: Tensor
@compile_ops("module_quant")
```

**`csrc/include/ater_hip_common.h`**
```
std::cout << "hipModuleLoad: " << (std::string(ATER_ASM_DIR) + hsaco).c_str() << "GetFunction: " << hsaco;
std::cout << " Success" << std::endl;
```

**`csrc/include/moe_ck.h`**
```
torch::Tensor ck_moe(torch::Tensor &hidden_states,          // [m, k], input token
torch::Tensor &w1,                     // [e, n, k]/[e, 2*n, k], pre-shuffle([e, nr, kr, w])
torch::Tensor &w2,                     // [e, n, k], pre-shuffle([e, nr, kr, w])
torch::Tensor &topk_weights,           // [tokens, topk]
```
