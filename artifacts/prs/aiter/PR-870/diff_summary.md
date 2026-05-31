# Diff summary

- **files changed:** 54
- **lines:** +1149 / -171
- **kernel-ish files:** 16

## Files (by churn)

- `csrc/py_itfs_cu/asm_mha_varlen_fwd.cu`  (+397/-0)
- `hsa/gfx942/fmha_v3_fwd/codegen.py`  (+268/-59)
- `aiter/ops/mha.py`  (+173/-29)
- `hsa/gfx950/fmha_v3_fwd/codegen.py`  (+98/-24)
- `csrc/include/mha_fwd.h`  (+63/-15)
- `csrc/include/torch/mha_v3_varlen_fwd.h`  (+33/-0)
- `op_tests/cpp/mha/smoke_test_fwd_v3.sh`  (+22/-10)
- `csrc/include/rocm_ops.hpp`  (+27/-0)
- `op_tests/test_mha_varlen.py`  (+7/-18)
- `csrc/cpp_itfs/mha_fwd_generate.py`  (+18/-6)
- `aiter/jit/optCompilerConfig.json`  (+20/-0)
- `csrc/pybind/mha_varlen_fwd_asm_pybind.cu`  (+9/-0)
- `op_tests/cpp/mha/benchmark_mha_fwd.cpp`  (+6/-3)
- `csrc/py_itfs_ck/mha_fwd_kernels.cu`  (+2/-2)
- `csrc/py_itfs_ck/mha_varlen_fwd_kernels.cu`  (+2/-2)

## Key added lines (kernel files)

**`aiter/jit/core.py`**
```
"fmha_v3_varlen_fwd",
```

**`aiter/ops/mha.py`**
```
def gen_fmha_v3_varlen_fwd_fake_tensor(
q: torch.Tensor,
k: torch.Tensor,
v: torch.Tensor,
```

**`csrc/cpp_itfs/mha_fwd_generate.py`**
```
int how_v3_bf16_cvt = 1,
how_v3_bf16_cvt,
bool use_ext_asm,
int how_v3_bf16_cvt,
```

**`csrc/include/mha_fwd.h`**
```
int how_v3_bf16_cvt,
use_ext_asm(use_ext_asm),
how_v3_bf16_cvt(how_v3_bf16_cvt)
int how_v3_bf16_cvt;
```

**`csrc/include/rocm_ops.hpp`**
```
m.def("fmha_v3_varlen_fwd",                   \
&aiter::torch_itfs::fmha_v3_varlen_fwd, \
py::arg("q"),                           \
py::arg("k"),                           \
```
