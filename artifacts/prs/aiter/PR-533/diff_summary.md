# Diff summary

- **files changed:** 26
- **lines:** +278 / -251
- **kernel-ish files:** 3

## Files (by churn)

- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4-N=512-K=7168.json`  (+27/-27)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4-N=7168-K=2048.json`  (+26/-26)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4-N=2112-K=7168.json`  (+25/-25)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4-N=3072-K=1536.json`  (+23/-23)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4-N=7168-K=256.json`  (+23/-23)
- `aiter/ops/triton/utils/pid_preprocessing.py`  (+24/-15)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4-N=16384-K=16384.json`  (+19/-19)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4-N=16384-K=4096.json`  (+13/-13)
- `aiter/ops/triton/gemm_afp4wfp4.py`  (+15/-9)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4-N=16384-K=26624.json`  (+11/-11)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4-N=2304-K=16384.json`  (+11/-11)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4-N=16384-K=8192.json`  (+9/-9)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4-N=16384-K=13312.json`  (+8/-8)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4-N=16384-K=2048.json`  (+7/-7)
- `op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4.py`  (+13/-1)

## Key added lines (kernel files)

**`aiter/ops/triton/gemm_afp4wfp4.py`**
```
from aiter.ops.triton.utils.pid_preprocessing import (
pid_grid,
remap_xcd,
pid_unified = remap_xcd(pid_unified, GRID_MN * NUM_KSPLIT, NUM_XCDS=8)
```

**`aiter/ops/triton/utils/pid_preprocessing.py`**
```
def remap_xcd_chunked(
pid, GRID_MN, NUM_XCDS: tl.constexpr = 8, CHUNK_SIZE: tl.constexpr = 2
xcd = pid % NUM_XCDS
if pid > (GRID_MN // (NUM_XCDS * CHUNK_SIZE)) * (NUM_XCDS * CHUNK_SIZE):
```

**`op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4.py`**
```
from utils.benchmark_utils import get_model_configs, get_available_models, print_vgpr
parser.add_argument(
"--print_vgpr",
action="store_true",
```
