# Diff summary

- **files changed:** 21
- **lines:** +1585 / -664
- **kernel-ish files:** 12

## Files (by churn)

- `csrc/py_itfs_cu/fmha_bwd_pre_post_kernel.py`  (+313/-561)
- `csrc/py_itfs_cu/asm_mha_varlen_bwd.cpp`  (+426/-0)
- `csrc/py_itfs_cu/asm_mha_bwd.cpp`  (+397/-0)
- `aiter/ops/mha.py`  (+155/-38)
- `csrc/cpp_itfs/generate.py`  (+114/-0)
- `aiter/jit/optCompilerConfig.json`  (+40/-7)
- `csrc/cpp_itfs/aiter_fmha_bwd.h`  (+47/-0)
- `csrc/py_itfs_ck/mha_bwd_kernels.cu`  (+13/-29)
- `csrc/py_itfs_ck/mha_varlen_bwd_kernels.cu`  (+14/-26)
- `csrc/include/fmha_v3_varlen_bwd.h`  (+31/-0)
- `csrc/include/rocm_ops.hpp`  (+26/-0)
- `csrc/pybind/mha_varlen_bwd_asm_pybind.cu`  (+9/-0)
- `csrc/pybind/mha_varlen_bwd_pybind.cu`  (+0/-3)
- `hsa/fmha_v3_bwd/bwd_hd64_bf16_a32_rtna_pssk_group.co`  (+0/-0)
- `hsa/fmha_v3_bwd/bwd_hd64_bf16_a32_rtne_pssk_group.co`  (+0/-0)

## Key added lines (kernel files)

**`aiter/ops/mha.py`**
```
@compile_ops("module_fmha_v3_varlen_bwd", fc_name="fmha_v3_varlen_bwd")
def fmha_v3_varlen_bwd(
dout: Tensor,
q: Tensor,
```

**`csrc/cpp_itfs/aiter_fmha_bwd.h`**
```
struct fmha_bwd_traits_all: public fmha_bwd_traits
fmha_bwd_traits_all(const mask_info &mask,
std::string dtype,
int head_size_q,
```

**`csrc/cpp_itfs/generate.py`**
```
import argparse
from pathlib import Path
from typing import List, Optional
GEN_DIR = ""    # in Cmake, have to generate files in same folder
```

**`csrc/include/fmha_v3_varlen_bwd.h`**
```
std::vector<at::Tensor>
fmha_v3_varlen_bwd(const at::Tensor &dout,         // [total_q, hq, d_v]
const at::Tensor &q,            // [total_q, hq, d_q]
const at::Tensor &k,            // [total_k, hk, d_q]
```

**`csrc/include/rocm_ops.hpp`**
```
m.def("fmha_v3_varlen_bwd", &fmha_v3_varlen_bwd,\
py::arg("dout"),                          \
py::arg("q"), py::arg("k"), py::arg("v"), \
py::arg("out"),                           \
```
