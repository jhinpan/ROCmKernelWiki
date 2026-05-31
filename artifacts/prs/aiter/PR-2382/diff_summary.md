# Diff summary

- **files changed:** 17
- **lines:** +835 / -998
- **kernel-ish files:** 15

## Files (by churn)

- `aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv`  (+479/-514)
- `csrc/py_itfs_cu/asm_a8w8_blockscale_bpreshuffle.cu`  (+97/-100)
- `csrc/py_itfs_cu/asm_gemm_a8w8.cu`  (+72/-76)
- `csrc/py_itfs_cu/asm_gemm_a4w4.cu`  (+63/-75)
- `aiter/ops/gemm_op_a8w8.py`  (+65/-39)
- `csrc/py_itfs_cu/asm_mi350_a8w8_blockscale.cu`  (+28/-41)
- `aiter/ops/gemm_op_a4w4.py`  (+29/-12)
- `csrc/include/rocm_ops.hpp`  (+0/-41)
- `csrc/include/asm_gemm_a4w4.h`  (+0/-16)
- `csrc/include/asm_a8w8_blockscale_bpreshuffle.h`  (+0/-15)
- `csrc/include/asm_gemm_a8w8.h`  (+0/-14)
- `csrc/pybind/asm_mi350_a8w8_blockscale_asm_pybind.cu`  (+0/-13)
- `csrc/include/asm_mi350_a8w8_blockscale.h`  (+0/-12)
- `csrc/pybind/gemm_a4w4_asm_pybind.cu`  (+0/-9)
- `csrc/pybind/gemm_a8w8_asm_pybind.cu`  (+0/-9)

## Key added lines (kernel files)

**`aiter/ops/gemm_op_a4w4.py`**
```
@compile_ops(
"module_gemm_a4w4_asm",
fc_name="gemm_a4w4_asm",
ffi_type="ctypes",
```

**`aiter/ops/gemm_op_a8w8.py`**
```
ffi_type="ctypes",
def _gemm_a8w8_asm(
kernelName: Optional[str] = None,
bias: Optional[Tensor] = None,  # bias:[1, N] f32
```

**`csrc/py_itfs_cu/asm_a8w8_blockscale_bpreshuffle.cu`**
```
static CFG* get_cfg(AiterDtype inp_dtype, AiterDtype out_dtype) {
if (inp_dtype == AITER_DTYPE_fp8 && out_dtype == AITER_DTYPE_bf16) {
AITER_CHECK(false, __func__, " Unsupported input_type: ", AiterDtype_to_str(inp_dtype),
", out_type: ", AiterDtype_to_str(out_dtype), ". Expected FP8 input and BFloat16 output.");
```

**`csrc/py_itfs_cu/asm_gemm_a4w4.cu`**
```
static CFG* get_cfg(AiterDtype inp_dtype, AiterDtype out_dtype)
if((inp_dtype == AITER_DTYPE_fp4x2 || inp_dtype == AITER_DTYPE_u8) &&
out_dtype == AITER_DTYPE_bf16)
AITER_CHECK(false,
```

**`csrc/py_itfs_cu/asm_gemm_a8w8.cu`**
```
static CFG* get_cfg(AiterDtype inp_dtype, AiterDtype out_dtype)
if(inp_dtype == AITER_DTYPE_i8 && out_dtype == AITER_DTYPE_bf16)
AITER_CHECK(false,
" Unsupported input_type: ", AiterDtype_to_str(inp_dtype),
```
