# Diff summary

- **files changed:** 19
- **lines:** +38 / -38
- **kernel-ish files:** 1

## Files (by churn)

- `op_tests/triton_tests/test_gemm_afp4wfp4.py`  (+3/-3)
- `aiter/ops/triton/configs/MI300X-GEMM-AFP4WFP4.json`  (+2/-2)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=106496-K=16384.json`  (+2/-2)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=13312-K=16384.json`  (+2/-2)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=16384-K=13312.json`  (+2/-2)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=16384-K=16384.json`  (+2/-2)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=16384-K=2048.json`  (+2/-2)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=16384-K=26624.json`  (+2/-2)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=16384-K=4096.json`  (+2/-2)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=16384-K=53248.json`  (+2/-2)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=16384-K=6656.json`  (+2/-2)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=16384-K=8192.json`  (+2/-2)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=18432-K=16384.json`  (+2/-2)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=2304-K=16384.json`  (+2/-2)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=26624-K=16384.json`  (+2/-2)

## Key added lines (kernel files)

**`op_tests/triton_tests/test_gemm_afp4wfp4.py`**
```
from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4_preshuffled_scales
triton_out = gemm_afp4wfp4_preshuffled_scales(x, w, x_scales_triton, w_scales_triton, dtype, y)
triton_out = gemm_afp4wfp4_preshuffled_scales(x, w, x_scales_triton, w_scales_triton, dtype)
```
