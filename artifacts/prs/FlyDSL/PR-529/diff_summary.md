# Diff summary

- **files changed:** 13 (diff was byte-capped; summary is partial)
- **lines:** +1005 / -1504
- **kernel-ish files:** 13

## Files (by churn)

- `kernels/mixed_moe_gemm_2stage.py`  (+292/-857)
- `kernels/flash_attn_func.py`  (+204/-149)
- `kernels/blockscale_preshuffle_gemm.py`  (+148/-128)
- `kernels/hgemm_splitk.py`  (+106/-94)
- `kernels/custom_all_reduce.py`  (+86/-47)
- `.claude/skills/kernel-trace-analysis/scripts/hotspot_analyzer.py`  (+70/-48)
- `kernels/mla_fwd_decode_m16x8_fp8_fp8.py`  (+19/-61)
- `kernels/mla_fwd_decode.py`  (+31/-39)
- `kernels/mfma_epilogues.py`  (+12/-45)
- `kernels/mfma_preshuffle_pipeline.py`  (+32/-19)
- `kernels/kernels_common.py`  (+3/-9)
- `kernels/layout_utils.py`  (+2/-6)
- `kernels/__init__.py`  (+0/-2)

## Key added lines (kernel files)

**`.claude/skills/kernel-trace-analysis/scripts/hotspot_analyzer.py`**
```
if "vmcnt" in asm:
return "VMEM-wait"
if "lgkmcnt" in asm:
return "LDS/SMEM-wait"
```

**`kernels/blockscale_preshuffle_gemm.py`**
```
from flydsl._mlir import ir
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
```

**`kernels/custom_all_reduce.py`**
```
return int(storage.nbytes()) - int(t.storage_offset()) * int(t.element_size()) == int(t.numel()) * int(
t.element_size()
def init_custom_ar(
meta, rank_data, handles, offsets, rank: int, full_nvlink: bool, out=None, max_size: int = _DEFAULT_MAX_SIZE
```

**`kernels/flash_attn_func.py`**
```
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw
```

**`kernels/hgemm_splitk.py`**
```
if self.dtype == "bf16":
c_frag_new = rocdl.mfma_f32_16x16x16f16(
T.vec(self.WMMA_C_FRAG_VALUES, T.f32), [a_frag, b_frag, c_frag, 0, 0, 0]
if self.dtype == "bf16":
```
