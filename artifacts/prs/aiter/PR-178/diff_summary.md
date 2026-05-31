# Diff summary

- **files changed:** 55
- **lines:** +2696 / -33
- **kernel-ish files:** 6

## Files (by churn)

- `csrc/py_itfs_cu/fmha_bwd_pre_post_kernel.py`  (+2373/-0)
- `aiter/ops/mha.py`  (+243/-28)
- `csrc/pybind/mha_bwd_asm_pybind.cu`  (+27/-0)
- `csrc/include/fmha_v3_bwd.h`  (+26/-0)
- `aiter/jit/optCompilerConfig.json`  (+18/-0)
- `csrc/py_itfs_ck/mha_bwd_kernels.cu`  (+5/-5)
- `csrc/include/aiter_hip_common.h`  (+4/-0)
- `hsa/fmha_v3_bwd/bwd_bf16_a16_rtna.co`  (+0/-0)
- `hsa/fmha_v3_bwd/bwd_bf16_a16_rtna_pddv.co`  (+0/-0)
- `hsa/fmha_v3_bwd/bwd_bf16_a16_rtne.co`  (+0/-0)
- `hsa/fmha_v3_bwd/bwd_bf16_a16_rtne_pddv.co`  (+0/-0)
- `hsa/fmha_v3_bwd/bwd_bf16_a16_rtz.co`  (+0/-0)
- `hsa/fmha_v3_bwd/bwd_bf16_a16_rtz_pddv.co`  (+0/-0)
- `hsa/fmha_v3_bwd/bwd_bf16_a32_rtna.co`  (+0/-0)
- `hsa/fmha_v3_bwd/bwd_bf16_a32_rtna_psskddv.co`  (+0/-0)

## Key added lines (kernel files)

**`aiter/ops/mha.py`**
```
@compile_ops("module_fmha_v3_bwd", fc_name="fmha_v3_bwd")
def fmha_v3_bwd(
dout: Tensor,
q: Tensor,
```

**`csrc/include/aiter_hip_common.h`**
```
struct p1
unsigned int _p0;
```

**`csrc/include/fmha_v3_bwd.h`**
```
std::vector<at::Tensor>
fmha_v3_bwd(const at::Tensor &dout, // [b, sq, hq, d]
const at::Tensor &q,    // [b, sq, hq, d]
const at::Tensor &k,    // [b, sk, hk, d]
```

**`csrc/py_itfs_ck/mha_bwd_kernels.cu`**
```
dk = dk_.value();
TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
CHECK_DEVICE(dk);
TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
```

**`csrc/py_itfs_cu/fmha_bwd_pre_post_kernel.py`**
```
from dataclasses import dataclass
import argparse
import fnmatch
import itertools
```
