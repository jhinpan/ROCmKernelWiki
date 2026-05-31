# Diff summary

- **files changed:** 15
- **lines:** +422 / -89
- **kernel-ish files:** 8

## Files (by churn)

- `csrc/py_itfs_cu/asm_gemm_a4w4.cu`  (+189/-32)
- `op_tests/test_gemm_a4w4.py`  (+65/-37)
- `hsa/gfx942/f4gemm/codegen.py`  (+64/-0)
- `hsa/gfx950/f4gemm/codegen.py`  (+64/-0)
- `csrc/include/rocm_ops.hpp`  (+15/-13)
- `aiter/ops/gemm_op_a4w4.py`  (+11/-2)
- `csrc/include/asm_gemm_a4w4.h`  (+6/-4)
- `hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4.csv`  (+5/-0)
- `aiter/jit/optCompilerConfig.json`  (+1/-1)
- `hsa/gfx942/f4gemm/f4gemm_bf16_per1x32Fp4.csv`  (+1/-0)
- `setup.py`  (+1/-0)
- `hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512.co`  (+0/-0)
- `hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co`  (+0/-0)
- `hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_KSplit_128x512.co`  (+0/-0)
- `hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_noBpreShuffle_256x256.co`  (+0/-0)

## Key added lines (kernel files)

**`aiter/ops/gemm_op_a4w4.py`**
```
from ..jit.utils.chip_info import get_gfx
gfx_arch = get_gfx()
if gfx_arch in ["gfx942"]:
raise RuntimeError(
```

**`csrc/include/asm_gemm_a4w4.h`**
```
std::string& kernelName,
std::optional<torch::Tensor>& bias, // bias:[M, N] f32
std::optional<float> alpha      = 1.0,
std::optional<float> beta       = 0.0,
```

**`csrc/include/rocm_ops.hpp`**
```
m.def("gemm_a4w4_asm",                        \
&gemm_a4w4_asm,                         \
"Asm gemm a4w4",                        \
py::arg("A"),                           \
```

**`csrc/py_itfs_cu/asm_gemm_a4w4.cu`**
```
int log2_k_split;
static CFG* get_cfg(torch::Tensor& inp, torch::Tensor& out)
if((inp.dtype() == torch::kFloat4_e2m1fn_x2 || inp.dtype() == torch::kUInt8) &&
out.scalar_type() == at::ScalarType::BFloat16)
```

**`hsa/gfx942/f4gemm/codegen.py`**
```
import os
import argparse
import glob
import pandas as pd
```
