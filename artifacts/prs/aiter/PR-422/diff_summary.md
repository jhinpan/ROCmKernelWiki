# Diff summary

- **files changed:** 19 (diff was byte-capped; summary is partial)
- **lines:** +715 / -700
- **kernel-ish files:** 19

## Files (by churn)

- `csrc/py_itfs_cu/fmha_bwd_pre_post_kernel.py`  (+655/-649)
- `csrc/cpp_itfs/mha_fwd_generate.py`  (+11/-11)
- `aiter/jit/utils/chip_info.py`  (+11/-4)
- `csrc/include/mha_fwd.h`  (+7/-7)
- `csrc/cpp_itfs/mha_bwd_generate.py`  (+6/-6)
- `csrc/include/mha_bwd.h`  (+3/-3)
- `csrc/include/aiter_hip_common.h`  (+3/-1)
- `csrc/py_itfs_ck/mha_batch_prefill_kernels.cu`  (+2/-2)
- `csrc/py_itfs_ck/mha_bwd_kernels.cu`  (+2/-2)
- `csrc/py_itfs_ck/mha_fwd_kernels.cu`  (+2/-2)
- `csrc/py_itfs_ck/mha_varlen_bwd_kernels.cu`  (+2/-2)
- `csrc/py_itfs_ck/mha_varlen_fwd_kernels.cu`  (+2/-2)
- `csrc/py_itfs_cu/asm_mha_bwd.cpp`  (+2/-2)
- `csrc/py_itfs_cu/asm_mha_varlen_bwd.cpp`  (+2/-2)
- `aiter/__init__.py`  (+1/-1)

## Key added lines (kernel files)

**`aiter/jit/utils/chip_info.py`**
```
import subprocess
result = subprocess.run(
["rocminfo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
output = result.stdout
```

**`csrc/cpp_itfs/mha_bwd_generate.py`**
```
mask_enum mask_type,
mask_type,
mask_enum mask_type,
auto traits = get_mha_bwd_traits(head_size_q,
```

**`csrc/cpp_itfs/mha_fwd_generate.py`**
```
mask_enum mask_type,
mask_type,
mask_enum mask_type,
mask_type,
```

**`csrc/include/mha_bwd.h`**
```
mask_enum mask_type,
mask_type,
mask_enum mask_type,
```

**`csrc/include/mha_fwd.h`**
```
mask_enum mask_type,
mask_type,
mask_enum mask_type,
mask_type,
```
