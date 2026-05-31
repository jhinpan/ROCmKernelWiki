# Diff summary

- **files changed:** 27
- **lines:** +747 / -1903
- **kernel-ish files:** 15

## Files (by churn)

- `hsa/gfx942/fmha_v3_fwd/codegen.py`  (+0/-921)
- `hsa/gfx950/fmha_v3_fwd/codegen.py`  (+0/-514)
- `csrc/cpp_itfs/mha_fwd.cpp`  (+332/-0)
- `csrc/cpp_itfs/mha_fwd_generate.py`  (+0/-271)
- `csrc/include/mha_fwd.h`  (+122/-69)
- `csrc/cpp_itfs/mha_fwd_batch_prefill.cpp`  (+63/-0)
- `aiter/jit/optCompilerConfig.json`  (+34/-23)
- `csrc/cpp_itfs/mha_fwd_split.cpp`  (+49/-0)
- `csrc/py_itfs_cu/asm_mha_varlen_fwd.cu`  (+21/-16)
- `csrc/py_itfs_ck/mha_fwd_kernels.cu`  (+20/-15)
- `csrc/py_itfs_cu/asm_mha_fwd.cu`  (+20/-15)
- `csrc/py_itfs_ck/mha_varlen_fwd_kernels.cu`  (+19/-14)
- `op_tests/cpp/mha/benchmark_mha_fwd.cpp`  (+14/-16)
- `op_tests/cpp/mha/compile.py`  (+12/-16)
- `hsa/gfx942/fmha_v3_fwd/fmha_fwd.csv`  (+25/-0)

## Key added lines (kernel files)

**`aiter/ops/mha.py`**
```
from ..jit.core import CK_DIR, AITER_META_DIR, compile_ops
```

**`csrc/cpp_itfs/mha_fwd.cpp`**
```
namespace aiter {
int get_cfg_mask_type(const mha_fwd_args& a)
if(a.mask_type == 0)
return 0;
```

**`csrc/cpp_itfs/mha_fwd_batch_prefill.cpp`**
```
namespace aiter {
mha_fwd_traits get_mha_fwd_traits(int head_size_q,
int head_size_v,
std::string dtype,
```

**`csrc/cpp_itfs/mha_fwd_split.cpp`**
```
namespace aiter {
mha_fwd_splitkv_traits get_mha_fwd_splitkv_traits(int head_size_q,
int head_size_v,
std::string dtype,
```

**`csrc/include/mha_fwd.h`**
```
struct mha_fwd_args
bool use_asm_v3;
bool v3_api_check;
int how_v3_bf16_cvt;
```
