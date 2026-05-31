# Diff summary

- **files changed:** 23
- **lines:** +3483 / -480
- **kernel-ish files:** 22

## Files (by churn)

- `csrc/kernels/mla/hk/hk_mla_buffer_managers.cuh`  (+1546/-0)
- `csrc/kernels/mla/hk/mi3xx_v32_fwd_decode_h128_fp8_fp8.cuh`  (+812/-0)
- `csrc/include/custom_all_reduce.cuh`  (+424/-378)
- `csrc/kernels/mla/hk/hk_mla_softmax.cuh`  (+272/-0)
- `aiter/jit/core.py`  (+156/-11)
- `csrc/include/rocm_ops.hpp`  (+65/-47)
- `aiter/mla.py`  (+47/-20)
- `csrc/kernels/mla/hk_decode_fwd.cu`  (+48/-0)
- `csrc/kernels/mla/metadata/v1_2_device.cuh`  (+15/-12)
- `aiter/ops/attention.py`  (+26/-0)
- `aiter/jit/optCompilerConfig.json`  (+18/-0)
- `csrc/kernels/mla/hk/hk_mla_utils.cuh`  (+16/-0)
- `csrc/include/mla.h`  (+15/-0)
- `csrc/pybind/mla_hk_pybind.cu`  (+10/-0)
- `csrc/include/custom_all_reduce.h`  (+4/-4)

## Key added lines (kernel files)

**`aiter/jit/core.py`**
```
HIP_KITTENS_DIR = os.environ.get(
"HIP_KITTENS_DIR", f"{AITER_META_DIR}/3rdparty/HipKittens"
def clone_3rdparty(third_party: str) -> None:
def MainFunc():
```

**`aiter/mla.py`**
```
import os
use_hk = (
nhead == 128
and q.dtype == dtypes.fp8
```

**`aiter/ops/attention.py`**
```
@compile_ops("module_hk_mla")
def hk_mla_decode_fwd(
query: torch.Tensor,
kv_buffer: torch.Tensor,
```

**`csrc/include/custom_all_reduce.cuh`**
```
using T         = typename opus::vector_traits<V>::dtype;
using T         = typename opus::vector_traits<O>::dtype;
packed_assign_add<typename opus::vector_traits<A>::dtype, opus::vector_traits<A>::size()>(
tmp, upcast(ptrs[i][idx]));
```

**`csrc/include/custom_all_reduce.h`**
```
torch::Tensor& t,
const std::vector<torch::Tensor>& handles,
const std::vector<int64_t>& offsets);
```
