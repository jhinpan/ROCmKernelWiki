# Diff summary

- **files changed:** 15
- **lines:** +879 / -558
- **kernel-ish files:** 13

## Files (by churn)

- `aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv`  (+173/-173)
- `csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py`  (+97/-65)
- `csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py`  (+91/-62)
- `aiter/ops/gemm_op_a8w8.py`  (+96/-55)
- `csrc/ck_gemm_a8w8/gemm_a8w8_tune.py`  (+82/-65)
- `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py`  (+70/-60)
- `aiter/utility/mp_tuner.py`  (+116/-11)
- `gradlib/gradlib/GemmTuner.py`  (+49/-21)
- `aiter/ops/gemm_op_a4w4.py`  (+42/-15)
- `aiter/configs/a8w8_tuned_gemm.csv`  (+27/-27)
- `hsa/gfx942/fmoe_2stages/tune.py`  (+13/-1)
- `op_tests/test_gemm_a8w8.py`  (+10/-2)
- `csrc/ck_gemm_a8w8/gen_instances.py`  (+6/-0)
- `csrc/ck_gemm_a8w8_bpreshuffle/gen_instances.py`  (+6/-0)
- `op_tests/test_gemm_a8w8_blockscale.py`  (+1/-1)

## Key added lines (kernel files)

**`aiter/ops/gemm_op_a4w4.py`**
```
from aiter import logger
AITER_ROOT_DIR,
import functools
import pandas as pd
```

**`aiter/ops/gemm_op_a8w8.py`**
```
from aiter import logger
def gemm_a8w8_ck(
def gemm_a8w8_bpreshuffle_ck(
def gemm_a8w8_blockscale_ck(
```

**`aiter/utility/mp_tuner.py`**
```
gpuIDMap,
ref=None,
rtol=1e-2,
atol=1e-2,
```

**`csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py`**
```
from aiter.utility.mp_tuner import mp_tuner
def run_gemm_a4w4_blockscale(x, weight, x_scale, w_scale, out, kernel_id, splitK):
m, k = x.shape
n, k = weight.shape
```

**`csrc/ck_gemm_a8w8/gemm_a8w8_tune.py`**
```
from aiter.utility.mp_tuner import mp_tuner
filtered_df = untunedf.drop_duplicates().reset_index(drop=True)
return filtered_df
def generate_data(m, n, k):
```
