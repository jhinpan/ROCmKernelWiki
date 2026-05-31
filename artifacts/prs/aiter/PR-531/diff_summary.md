# Diff summary

- **files changed:** 18
- **lines:** +2546 / -274
- **kernel-ish files:** 14

## Files (by churn)

- `csrc/ck_gemm_moe_2stages_codegen/gen_instances.py`  (+441/-0)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.h`  (+440/-0)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common.cuh`  (+318/-0)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common_blockscale.cuh`  (+318/-0)
- `hsa/gfx942/fmoe_2stages/tune.py`  (+234/-48)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common.py`  (+257/-0)
- `op_tests/test_moe_2stage.py`  (+137/-77)
- `aiter/ops/moe_op.py`  (+145/-5)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu`  (+138/-0)
- `csrc/include/rocm_ops.hpp`  (+57/-56)
- `aiter/fused_moe.py`  (+28/-56)
- `aiter/configs/tuned_fmoe.csv`  (+6/-12)
- `aiter/utility/mp_tuner.py`  (+12/-3)
- `aiter/configs/untuned_fmoe.csv`  (+4/-9)
- `aiter/jit/optCompilerConfig.json`  (+6/-4)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
kernelName1 = ""
kernelName2 = ""
kernelName1 = cfg["kernelName1"]
kernelName2 = cfg["kernelName2"]
```

**`aiter/ops/moe_op.py`**
```
from ..jit.core import compile_ops, AITER_CSRC_DIR
from ..utility import dtypes
import functools
torch.int4 = getattr(torch, "int4", torch.uint32)
```

**`aiter/utility/mp_tuner.py`**
```
def worker(
gpuIDMap, tag, func, args, kwargs, ref=None, rtol=1e-2, atol=1e-2, printLog=False
atol=atol,
rtol=rtol,
```

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu`**
```
using MoeKernelMap = std::unordered_map<std::string, MoeKernel>;
template <int stage = 1>
MoeKernel moe_dispatch(std::string &kernelName, int block_m)
static const auto lookup = []
```

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.h`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using I4 = ck::pk_i4_t;
using I8 = int8_t;
```
