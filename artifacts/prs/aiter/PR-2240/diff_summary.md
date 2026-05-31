# Diff summary

- **files changed:** 8
- **lines:** +1788 / -2049
- **kernel-ish files:** 8

## Files (by churn)

- `aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention_mxfp4.py`  (+15/-1762)
- `aiter/ops/triton/_triton_kernels/quant/sage_attention_quant.py`  (+987/-0)
- `aiter/ops/triton/quant/sage_attention_quant_wrappers.py`  (+677/-0)
- `aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention.py`  (+3/-264)
- `op_tests/op_benchmarks/triton/bench_fav3_sage_mxfp4.py`  (+76/-16)
- `aiter/ops/triton/attention/fav3_sage_attention_mxfp4_wrapper.py`  (+26/-5)
- `aiter/ops/triton/attention/fav3_sage.py`  (+3/-1)
- `op_tests/triton_tests/attention/test_fav3_sage.py`  (+1/-1)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention.py`**
```
start_m = tl.program_id(0).to(tl.int64)
off_h_q = tl.program_id(1).to(tl.int64)
off_z = tl.program_id(2).to(tl.int64)
```

**`aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention_mxfp4.py`**
```
if USE_BIAS:
m_diff = tl.where(m_ij == float("-inf"), float("-inf"), m_i - m_ij)
m_diff = m_i - m_ij
alpha = tl.math.exp2(m_diff)
```

**`aiter/ops/triton/_triton_kernels/quant/sage_attention_quant.py`**
```
import triton
import triton.language as tl
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid_3d
@triton.jit
```

**`aiter/ops/triton/attention/fav3_sage.py`**
```
from aiter.ops.triton.quant.sage_attention_quant_wrappers import sage_quant
```

**`aiter/ops/triton/attention/fav3_sage_attention_mxfp4_wrapper.py`**
```
from aiter.ops.triton.quant.sage_attention_quant_wrappers import sage_quant_mxfp4
import aiter
FP8_TYPE = aiter.dtypes.fp8
FP8_MAX = torch.finfo(FP8_TYPE).max
```
