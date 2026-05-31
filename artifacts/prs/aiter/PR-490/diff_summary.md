# Diff summary

- **files changed:** 20
- **lines:** +68 / -51
- **kernel-ish files:** 2

## Files (by churn)

- `op_tests/triton_tests/test_gemm_afp4wfp4.py`  (+15/-10)
- `op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4.py`  (+18/-6)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=2304-K=16384.json`  (+5/-5)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=16384-K=4096.json`  (+4/-4)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=18432-K=16384.json`  (+4/-4)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=9216-K=16384.json`  (+4/-4)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=16384-K=2048.json`  (+3/-3)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=16384-K=6656.json`  (+3/-3)
- `aiter/ops/triton/configs/MI300X-GEMM-AFP4WFP4.json`  (+1/-1)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=106496-K=16384.json`  (+1/-1)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=13312-K=16384.json`  (+1/-1)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=16384-K=13312.json`  (+1/-1)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=16384-K=16384.json`  (+1/-1)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=16384-K=26624.json`  (+1/-1)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=16384-K=53248.json`  (+1/-1)

## Key added lines (kernel files)

**`op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4.py`**
```
from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4, gemm_afp4wfp4_preshuffled_scales
import os
TRITON_HIP_PRESHUFFLE_SCALES = (
os.environ.get("TRITON_HIP_PRESHUFFLE_SCALES", "0") == "1"
```

**`op_tests/triton_tests/test_gemm_afp4wfp4.py`**
```
from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4, gemm_afp4wfp4_preshuffled_scales
if M >= 32:
x_scales_shuffled = shuffle_scales(x_scales)
x_scales_shuffled = x_scales
```
