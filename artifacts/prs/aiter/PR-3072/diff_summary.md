# Diff summary

- **files changed:** 10 (diff was byte-capped; summary is partial)
- **lines:** +3835 / -536
- **kernel-ish files:** 9

## Files (by churn)

- `csrc/kernels/mla/hk/mi35x_v32_fwd_decode_m16x4_fp8_fp8.cuh`  (+1381/-0)
- `csrc/kernels/mla/hk/mi35x_v32_fwd_decode_m16x8_fp8_fp8.cuh`  (+1313/-0)
- `csrc/kernels/mla/hk/hk_mla_buffer_managers.cuh`  (+604/-165)
- `csrc/kernels/mla/hk/mi3xx_v32_fwd_decode_h128_fp8_fp8.cuh`  (+0/-351)
- `csrc/kernels/mla/hk/hk_mla_utils.cuh`  (+263/-2)
- `csrc/kernels/mla/hk/hk_mla_softmax.cuh`  (+205/-9)
- `aiter/mla.py`  (+27/-5)
- `aiter/ops/attention.py`  (+25/-1)
- `aiter/jit/core.py`  (+14/-2)
- `aiter/jit/optCompilerConfig.json`  (+3/-1)

## Key added lines (kernel files)

**`aiter/jit/core.py`**
```
def is_experimental_enabled() -> bool:
val = os.environ.get("AITER_ENABLE_EXPERIMENTAL", "0")
return int(val) != 0
except ValueError:
```

**`aiter/mla.py`**
```
from aiter.jit.core import is_experimental_enabled
get_gfx() == "gfx950"
and nhead == 128
and q.dtype == dtypes.fp8
```

**`aiter/ops/attention.py`**
```
from ..jit.core import compile_ops, is_experimental_enabled
is_hk_m16x4 = (
get_gfx() == "gfx950"
and q_dtype == dtypes.fp8
```

**`csrc/kernels/mla/hk/hk_mla_buffer_managers.cuh`**
```
class QManager8bitsV1
class QManager8bitsV2
__device__ QManager8bitsV2()
const uint32_t lane_idx = opus::lane_id();
```

**`csrc/kernels/mla/hk/hk_mla_softmax.cuh`**
```
template <bool kCheckBoundary, uint32_t GPR_4>
__device__ __forceinline__ void
softmax_scale_p_quad(const uint32_t col_start_idx, const uint32_t kv_end, const float softmax_scale)
constexpr uint32_t minus_inf_f32     = 0xff800000;
```
