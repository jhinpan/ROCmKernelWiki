# Diff summary

- **files changed:** 64 (diff was byte-capped; summary is partial)
- **lines:** +2170 / -2403
- **kernel-ish files:** 60

## Files (by churn)

- `csrc/cpp_itfs/pa/pa.cuh`  (+604/-1822)
- `csrc/cpp_itfs/pa/pa_kernels.cuh`  (+673/-0)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.h`  (+153/-223)
- `csrc/cpp_itfs/pa/pa_common.cuh`  (+298/-0)
- `aiter/ops/attention.py`  (+119/-22)
- `csrc/cpp_itfs/pa/pa.cpp.jinja`  (+41/-60)
- `csrc/cpp_itfs/mla/asm_mla_decode_fwd.cpp`  (+1/-90)
- `csrc/cpp_itfs/mla/asm_mla_decode_fwd.h`  (+89/-0)
- `csrc/cpp_itfs/pa/pa.py`  (+35/-20)
- `aiter/aot/test/test_matmul.cpp`  (+19/-19)
- `aiter/aot/triton/norm.py`  (+20/-17)
- `csrc/ck_gemm_a4w4_blockscale/include/gemm_a4w4_blockscale.h`  (+13/-15)
- `csrc/cpp_itfs/mla/asm_mla_decode_fwd.py`  (+17/-10)
- `aiter/aot/triton/decode_mla.py`  (+10/-12)
- `aiter/aot/test/matmul_fp16.py`  (+10/-9)

## Key added lines (kernel files)

**`aiter/aot/test/matmul_fp16.py`**
```
from triton.tools.compile import compile_kernel, CompileArgs
compile_args = CompileArgs(
path=__file__,
kernel_name="matmul_fp16",
```

**`aiter/aot/test/test_matmul.cpp`**
```
ASSERT_EQ(hipSuccess, matmul_fp16_b3a5a34c_0d1d2d34567c89c10d11c121314(
```

**`aiter/aot/triton/decode_mla.py`**
```
from triton.tools.compile import compile_kernel, CompileArgs
compile_args = CompileArgs(
path=f"{AITER_ROOT_DIR}/aiter/mla.py",
kernel_name="_fwd_kernel_stage2_asm",
```

**`aiter/aot/triton/norm.py`**
```
from triton.tools.compile import compile_kernel, CompileArgs
compile_args = CompileArgs(
path=f"{AITER_ROOT_DIR}/aiter/ops/triton/norm.py",
kernel_name="_layernorm_kernel",
```

**`aiter/jit/core.py`**
```
archs = [arch.strip() for arch in archs]
```
