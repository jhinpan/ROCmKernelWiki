# Diff summary

- **files changed:** 11
- **lines:** +480 / -301
- **kernel-ish files:** 11

## Files (by churn)

- `csrc/ck_gemm_moe_2stages_codegen/gen_instances.py`  (+390/-281)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common.py`  (+29/-6)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.h`  (+30/-0)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu`  (+10/-6)
- `aiter/ops/moe_op.py`  (+6/-1)
- `csrc/include/py_itfs_common.h`  (+6/-0)
- `aiter/jit/core.py`  (+5/-0)
- `csrc/kernels/quant_kernels.cu`  (+2/-2)
- `csrc/py_itfs_cu/asm_fmoe.cu`  (+1/-2)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common_blockscale.cuh`  (+0/-2)
- `csrc/py_itfs_cu/asm_gemm_a4w4.cu`  (+1/-1)

## Key added lines (kernel files)

**`aiter/jit/core.py`**
```
import torch
if hasattr(torch, "float4_e2m1fn_x2"):
flags_hip += ["-DTORCH_Float4_e2m1fn_x2"]
```

**`aiter/ops/moe_op.py`**
```
md_name = "module_moe_ck2stages"
blob_gen_cmd = [
f"{AITER_CSRC_DIR}/ck_gemm_moe_2stages_codegen/gen_instances.py -w {{}}"
```

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu`**
```
MoeKernel moe_dispatch(std::string &kernelName, int block_m, int inter_dim, at::ScalarType x_dtype, at::ScalarType w_dty
return moe_stage1_heuristic_dispatch(block_m, x_dtype, w_dtype, y_dtype, act_op, quant_type, mul_routed_weight);
return moe_stage2_heuristic_dispatch(block_m, inter_dim, x_dtype, w_dtype, y_dtype, act_op, quant_type, mul_routed_weigh
bool MulRoutedWeight = sorted_weights.has_value();
```

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.h`**
```
using CK_Dtype = std::variant<I4, I8, I32, F16, B16, F8, F32, FP4X2>;
struct CK_DTypeVisitor
at::ScalarType operator()(I4) { return torch::kUInt32; }
at::ScalarType operator()(I8) { return torch::kInt8; }
```

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common.py`**
```
QuantType: int = 1
Adtype: str = ""
Bdtype: str = ""
Cdtype: str = ""
```
