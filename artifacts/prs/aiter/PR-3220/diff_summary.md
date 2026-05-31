# Diff summary

- **files changed:** 25
- **lines:** +1210 / -469
- **kernel-ish files:** 21

## Files (by churn)

- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`  (+486/-218)
- `gradlib/gradlib/GemmTuner.py`  (+103/-45)
- `aiter/test_common.py`  (+114/-5)
- `csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py`  (+64/-26)
- `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py`  (+61/-23)
- `csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py`  (+49/-33)
- `op_tests/tuning_tests/test_tune_pipeline.py`  (+60/-8)
- `aiter/utility/base_tuner.py`  (+65/-1)
- `aiter/configs/a4w4_blockscale_tuned_gemm.csv`  (+32/-32)
- `csrc/ck_gemm_a8w8/gemm_a8w8_tune.py`  (+31/-14)
- `aiter/utility/mp_tuner.py`  (+36/-7)
- `csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py`  (+31/-12)
- `csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.py`  (+19/-11)
- `op_tests/test_gemm_a8w8.py`  (+20/-8)
- `op_tests/test_gemm_a16w16.py`  (+9/-5)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
import re
lock_name = re.sub(r"[^\w.\-]", "_", str(keys))
lock_path = os.path.join(bd_dir, f"lock_fmoe_tune_{lock_name}")
```

**`aiter/test_common.py`**
```
_ = func(*args, **kwargs)
_CATASTROPHIC_REL_THRESHOLD = 0.5
def _relmag_catastrophic(actual_max_delta, b):
"""Relative-magnitude catastrophic heuristic.
```

**`aiter/utility/base_tuner.py`**
```
help="Tolerable error ratio (default 0.05). During tuning, kernels "
"with observed error above this are rejected. During --run_config, "
"the effective threshold per shape is max(this value, the observed "
"errRatio stored in the tuned CSV), so kernels that were tuned with "
```

**`aiter/utility/mp_tuner.py`**
```
max_abs_delta=None,
output_keys=None,
_arg_key_list=None,
catastrophic_check=True,
```

**`csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py`**
```
"weight": weight,
"x_scale": x_scale,
"w_scale": w_scale,
"out": out,
```
