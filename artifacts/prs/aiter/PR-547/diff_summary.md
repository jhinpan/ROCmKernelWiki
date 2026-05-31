# Diff summary

- **files changed:** 11
- **lines:** +145 / -219
- **kernel-ish files:** 10

## Files (by churn)

- `op_tests/test_gemm_a4w4_blockscale.py`  (+0/-154)
- `csrc/py_itfs_cu/asm_gemm_a4w4.cu`  (+55/-42)
- `op_tests/test_gemm_a4w4.py`  (+73/-12)
- `csrc/include/asm_gemm_a4w4.h`  (+9/-8)
- `csrc/include/rocm_ops.hpp`  (+2/-1)
- `aiter/jit/utils/chip_info.py`  (+2/-0)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu`  (+1/-1)
- `csrc/kernels/quant_kernels.cu`  (+1/-1)
- `aiter/ops/gemm_op_a4w4.py`  (+1/-0)
- `hsa/gfx942/fmoe_2stages/tune.py`  (+1/-0)
- `hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_tn_bpreshuffle_256x256.co`  (+0/-0)

## Key added lines (kernel files)

**`aiter/jit/utils/chip_info.py`**
```
elif ";" in gfx:
gfx = gfx.split(";")[-1]
```

**`aiter/ops/gemm_op_a4w4.py`**
```
bpreshuffle: Optional[bool] = True,
```

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu`**
```
void *w1_scale_ptr = w1_scale.has_value() ? w1_scale.value().data_ptr() : nullptr;
```

**`csrc/include/asm_gemm_a4w4.h`**
```
torch::Tensor gemm_a4w4_asm(torch::Tensor& A,       // A:[M, K/2] f4x2
torch::Tensor& B,       // B:[N, K/2] f4x2
torch::Tensor& A_scale, // A_scale:[M, K/32] e8m0 paded
torch::Tensor& B_scale, // B_scale:[N, K/32] e8m0 paded
```

**`csrc/include/rocm_ops.hpp`**
```
py::arg("beta") = 0.0, \
py::arg("bpreshuffle")  = true);
```
