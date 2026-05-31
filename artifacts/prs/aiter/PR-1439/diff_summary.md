# Diff summary

- **files changed:** 116
- **lines:** +344 / -940
- **kernel-ish files:** 26

## Files (by churn)

- `hsa/codegen.py`  (+131/-0)
- `hsa/gfx950/i8gemm/codegen.py`  (+0/-76)
- `hsa/gfx942/fmoe/codegen.py`  (+0/-70)
- `hsa/gfx950/fmoe/codegen.py`  (+0/-70)
- `hsa/gfx942/pa/codegen.py`  (+0/-68)
- `hsa/gfx950/pa/codegen.py`  (+0/-68)
- `hsa/gfx942/bf16gemm/codegen.py`  (+0/-66)
- `hsa/gfx950/bf16gemm/codegen.py`  (+0/-66)
- `hsa/gfx942/f4gemm/codegen.py`  (+0/-64)
- `hsa/gfx942/i8gemm/codegen.py`  (+0/-64)
- `hsa/gfx950/f4gemm/codegen.py`  (+0/-64)
- `hsa/gfx942/fmoe_2stages/codegen.py`  (+0/-62)
- `hsa/gfx950/fmoe_2stages/codegen.py`  (+0/-62)
- `csrc/py_itfs_cu/asm_gemm_a16w16.cu`  (+29/-12)
- `csrc/py_itfs_cu/asm_gemm_a8w8.cu`  (+13/-10)

## Key added lines (kernel files)

**`aiter/jit/core.py`**
```
from chip_info import get_gfx, get_gfx_list
gfx = get_gfx_list()
if len(gfx) == 1:
AITER_ASM_DIR = f"{AITER_META_DIR}/hsa/{gfx[0]}/"
```

**`aiter/ops/gemm_op_a16w16.py`**
```
bpreshuffle: bool = False,
bpreshuffle: bool = False,
```

**`csrc/include/asm_gemm_a16w16.h`**
```
std::optional<std::string> kernelName,
bool bpreshuffle = false);
```

**`csrc/include/rocm_ops.hpp`**
```
py::arg("kernelName") = std::nullopt, \
py::arg("bpreshuffle")      = false);
```

**`csrc/py_itfs_cu/asm_fmoe.cu`**
```
std::string arch_id = get_gpu_arch();
std::string selectedKl = kernel_name.empty() ? "" : arch_id + kernel_name;
if (el.first.find(arch_id) != 0)
continue;
```
