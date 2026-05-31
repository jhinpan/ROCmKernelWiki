# Diff summary

- **files changed:** 21
- **lines:** +306 / -89
- **kernel-ish files:** 6

## Files (by churn)

- `gradlib/gradlib/GemmTuner.py`  (+114/-27)
- `aiter/tuned_gemm.py`  (+121/-12)
- `op_tests/test_gemm_a16w16.py`  (+18/-9)
- `aiter/configs/bf16_untuned_gemm.csv`  (+11/-11)
- `hsa/gfx942/bf16gemm/bf16gemm_fp32bf16.csv`  (+10/-10)
- `aiter/configs/bf16_tuned_gemm.csv`  (+8/-9)
- `hsa/gfx950/bf16gemm/bf16gemm_fp32bf16.csv`  (+13/-3)
- `aiter/utility/mp_tuner.py`  (+4/-3)
- `gradlib/README.md`  (+4/-3)
- `aiter/jit/core.py`  (+2/-1)
- `csrc/py_itfs_cu/asm_gemm_a16w16.cu`  (+1/-1)
- `hsa/gfx950/bf16gemm/bf16gemm_fp32bf16_tn_128x64_bshuffle_splitk.co`  (+0/-0)
- `hsa/gfx950/bf16gemm/bf16gemm_fp32bf16_tn_160x64_bshuffle_splitk.co`  (+0/-0)
- `hsa/gfx950/bf16gemm/bf16gemm_fp32bf16_tn_32x64_bshuffle_splitk.co`  (+0/-0)
- `hsa/gfx950/bf16gemm/bf16gemm_fp32bf16_tn_32x64_pf3_splitk.co`  (+0/-0)

## Key added lines (kernel files)

**`aiter/jit/core.py`**
```
keys = untunedf.columns.to_list()
keys.append("cu_num")
```

**`aiter/tuned_gemm.py`**
```
extensions_created = False
untune_path = f"{this_dir}/configs/bf16_untuned_gemm.csv"
tune_path = AITER_CONFIGS.AITER_CONFIG_GEMM_BF16_FILE
tuned_df = pd.DataFrame(
```

**`aiter/utility/mp_tuner.py`**
```
if not tasks:
return []
```

**`csrc/py_itfs_cu/asm_gemm_a16w16.cu`**
```
else if (cfg.splitK == 1)// auto select
```

**`gradlib/gradlib/GemmTuner.py`**
```
from aiter.ops.shuffle import shuffle_weight
from aiter.jit.utils.chip_info import get_gfx
def call_hipb_mm(
input, weight, bias, scale_a, scale_b, solidx, out_dtype, bpreshuffle=False
```
